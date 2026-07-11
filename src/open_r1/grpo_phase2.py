#!/usr/bin/env python
"""
Phase 2: Entropy-based hint resample + GRPO training.

Usage:
  accelerate launch src/open_r1/grpo_phase2.py --config config_limo_phase2.yaml
"""
import json
import logging
import os
import re
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from open_r1.hint_resample import compute_deltas, apply_alpha_filter, apply_alpha_filter_absolute, choose_truncation_step

logger = logging.getLogger(__name__)


def _build_phase2_prefix(
    tokenizer, system_prompt, question, completion_text,
    entropy_steps, alpha, beta, insert_hint, hint_text,
    use_absolute_h=False,
):
    """Build Phase 2 prompt prefix with entropy-based truncation.

    Args:
        use_absolute_h: If True, use absolute entropy H > alpha instead of
                       ΔH = H_t - H_{t-1} > alpha to find truncation points.
                       This is the "w/o ΔH trigger" ablation.
    """
    steps = entropy_steps
    if len(steps) < 2:
        return None

    metrics = {
        "chosen_truncation_index": None,
        "spike_delta_H": None,
        "spike_absolute_H": None,
        "truncated_suffix": "",
        "clean_prompt": "",
    }

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    clean_prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    metrics["clean_prompt"] = clean_prompt

    if use_absolute_h:
        # Ablation: use absolute entropy H instead of ΔH
        qualifying = apply_alpha_filter_absolute(steps, alpha)
    else:
        # Standard DERPO: use ΔH = H_t - H_{t-1}
        deltas = compute_deltas(steps)
        qualifying = apply_alpha_filter(deltas, alpha)

    if not qualifying:
        metrics["prefix_text"] = clean_prompt
    else:
        chosen_step_idx = choose_truncation_step(qualifying, beta)
        chosen_step = steps[chosen_step_idx]

        metrics["chosen_truncation_index"] = chosen_step["step"]
        metrics["spike_absolute_H"] = chosen_step["entropy"]

        if not use_absolute_h:
            # ΔH only defined when using deltas
            deltas = compute_deltas(steps)
            chosen_delta = deltas[chosen_step_idx - 1]
            metrics["spike_delta_H"] = chosen_delta

        rel_pos = chosen_step.get("rel_pos", 0)
        comp_ids = tokenizer(completion_text, return_tensors="pt").input_ids[0]
        if rel_pos >= len(comp_ids):
            rel_pos = len(comp_ids) - 1
        truncated_ids = comp_ids[:rel_pos + 1]
        truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        metrics["discarded_tokens_len"] = comp_ids.size(0) - rel_pos - 1

        # truncated_suffix: old reasoning up to spike + hint
        # Prepended to completion_ids before forward pass → participates in loss.
        truncated_suffix = truncated_text
        if insert_hint:
            truncated_suffix += hint_text
        metrics["truncated_suffix"] = truncated_suffix

        # prompt for vLLM: truncated prefix so model continues from the spike
        base = tokenizer.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        metrics["prefix_text"] = base + truncated_text + (hint_text if insert_hint else "")

    return metrics


def build_phase2_dataset(input_json, tokenizer, system_prompt, alpha, beta, insert_hint, hint_text, skip_samples=0, use_absolute_h=False):
    with open(input_json) as f:
        phase1_data = json.load(f)
    if isinstance(phase1_data, dict):
        phase1_data = [phase1_data]

    rows = []
    skipped_no_completion = 0
    skipped_no_entropy = 0
    skipped_prefix_failed = 0
    for entry_idx, entry in enumerate(phase1_data):
        q_text = entry.get("q", "")
        ground_truth = str(entry.get("ground_truth", "")).strip()
        for sample_idx, o_i_entry in enumerate(entry.get("o_i", [])):
            if isinstance(o_i_entry, str):
                skipped_no_completion += 1
                continue
            comp_text = o_i_entry.get("completion_text", "")
            entropy_steps = o_i_entry.get("entropy_steps", [])
            if not comp_text:
                skipped_no_completion += 1
                continue
            if len(entropy_steps) < 2:
                skipped_no_entropy += 1
                logger.debug(f"  skip entry={entry_idx} sample={sample_idx+1}: "
                             f"only {len(entropy_steps)} entropy steps (need >=2)")
                continue
            info = _build_phase2_prefix(
                tokenizer, system_prompt, q_text, comp_text, entropy_steps,
                alpha, beta, insert_hint, hint_text,
                use_absolute_h=use_absolute_h,
            )
            if info is None:
                skipped_prefix_failed += 1
                logger.debug(f"  skip entry={entry_idx} sample={sample_idx+1}: "
                             f"_build_phase2_prefix returned None")
                continue
            rows.append({
                "prompt": info["prefix_text"],
                "question": q_text,
                "sample_id": sample_idx + 1,
                "ground_truth": ground_truth,
                "solution": ground_truth,   # cosine_scaled_reward expects 'solution'
                "answer": ground_truth,     # accuracy_reward can use 'answer'
                "orig_completion_text": comp_text,
                "entropy_steps": entropy_steps,
                "chosen_truncation_index": info["chosen_truncation_index"],
                "spike_delta_H": info["spike_delta_H"],
                "spike_absolute_H": info["spike_absolute_H"],
                "discarded_tokens_len": info.get("discarded_tokens_len", 0),
                "truncated_suffix": info.get("truncated_suffix", ""),
                "clean_prompt": info.get("clean_prompt", ""),
            })

    ds = datasets.Dataset.from_list(rows)
    logger.info(
        f"Phase 2 dataset: {len(rows)} valid samples from {len(phase1_data)} questions. "
        f"Skipped: {skipped_no_completion} no-completion, "
        f"{skipped_no_entropy} insufficient-entropy-steps, "
        f"{skipped_prefix_failed} prefix-build-failed."
    )
    if skip_samples > 0:
        ds = ds.select(range(skip_samples, len(ds)))
        logger.info(f"Phase 2 dataset: {len(ds)} samples after skipping {skip_samples} (resume)")
    return ds


class Phase2Trainer(GRPOTrainer):
    """Light wrapper around GRPOTrainer for Phase 2 metrics recording.

    Metrics are collected via the _on_completions_generated hook, which fires
    right after completions are generated and scored — no dependency on the
    deque-backed _textual_logs (maxlen=128) that silently drops old entries.
    """

    def __init__(self, script_args, **kwargs):
        super().__init__(**kwargs)
        self._p2_args = script_args
        self._p2_metrics: list[dict] = []
        self._p2_output = script_args.phase2_output_json
        if os.path.exists(self._p2_output):
            try:
                with open(self._p2_output) as f:
                    self._p2_metrics = [json.loads(line) for line in f if line.strip()]
            except Exception:
                pass
        # Prevent _remove_unused_columns from stripping Phase 2 metadata columns
        # (question, sample_id, ground_truth, etc.). Must be set before get_train_dataloader().
        self._signature_columns = list(self.train_dataset.column_names)

    # ── wandb display: clean prompt + full completion ────────────────────
    def _get_display_texts(self, inputs, prompts_text, completions_text):
        """Show clean system+user as prompt, truncated_suffix+generated as completion."""
        clean_prompts = [row.get("clean_prompt", pt) for row, pt in zip(inputs, prompts_text)]
        return clean_prompts, completions_text  # completions_text already includes truncated_suffix

    # ── prepend truncated suffix to completions ──────────────────────────
    def _prepend_phase2_suffix(self, inputs, prompt_ids, completion_ids, completion_mask):
        """Prepend truncated_suffix + hint tokens to each completion so they
        participate in the GRPO loss alongside the newly generated tokens."""
        device = completion_ids.device
        pad_token_id = self.processing_class.pad_token_id
        batch_size = completion_ids.size(0)

        # Collect the truncated suffix text for each row in the batch
        suffix_texts = [row.get("truncated_suffix", "") for row in inputs]
        if not any(suffix_texts):
            return prompt_ids, completion_ids, completion_mask

        # Tokenize each suffix (no padding — we'll pad manually later)
        suffix_encodings = []
        for text in suffix_texts:
            if text:
                ids = self.processing_class(text, add_special_tokens=False).input_ids
                suffix_encodings.append(torch.tensor(ids, dtype=completion_ids.dtype, device=device))
            else:
                suffix_encodings.append(torch.tensor([], dtype=completion_ids.dtype, device=device))

        # Build new completion_ids by concatenating suffix + generated completion
        new_completions = []
        new_masks = []
        new_max_len = 0
        for i in range(batch_size):
            suffix = suffix_encodings[i]
            # Trim padding from the original completion using the mask
            orig_len = completion_mask[i].sum().item()
            orig_comp = completion_ids[i, :orig_len]
            combined = torch.cat([suffix, orig_comp])
            combined_mask = torch.ones(len(combined), dtype=completion_mask.dtype, device=device)
            new_completions.append(combined)
            new_masks.append(combined_mask)
            new_max_len = max(new_max_len, len(combined))

        # Re-pad to uniform length
        padded_comp = []
        padded_mask = []
        for i in range(batch_size):
            comp = new_completions[i]
            mask = new_masks[i]
            pad_len = new_max_len - len(comp)
            if pad_len > 0:
                comp = torch.cat([comp, torch.full((pad_len,), pad_token_id, dtype=comp.dtype, device=device)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype, device=device)])
            padded_comp.append(comp)
            padded_mask.append(mask)

        new_completion_ids = torch.stack(padded_comp)
        new_completion_mask = torch.stack(padded_mask)

        return prompt_ids, new_completion_ids, new_completion_mask

    def _flush_metrics(self, new_entries: list[dict] | None = None):
        """Append new entries to the output file in JSONL format (one JSON object per line).
        Appending means a crash mid-training won't lose previously written data."""
        if new_entries and self.accelerator.is_main_process:
            with open(self._p2_output, "a") as f:
                for r in new_entries:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── hook: called from _generate_and_score_completions ────────────────
    def _on_completions_generated(self, inputs, prompts_text, completions_text):
        """Record per-sample metrics immediately after generation.

        Called once per generation batch (every steps_per_generation steps).
        A generation batch may contain multiple unique prompts; each unique
        prompt is repeated G times (once per generation).  We iterate over
        every unique prompt so that each (question, sample_id) gets its own
        output record.
        """
        if not self.accelerator.is_main_process:
            return

        G = self.num_generations
        if len(inputs) == 0:
            return

        # Each unique prompt occupies a contiguous block of G rows in the batch
        batch_entries = []
        for base in range(0, len(inputs), G):
            sample = inputs[base]
            prompt_text = sample.get("prompt", "")
            q = sample.get("question", "")
            sid = sample.get("sample_id", 0)
            gt = str(sample.get("ground_truth", "")).strip()

            # Completions belonging to this sample (G entries)
            sample_completions = completions_text[base:base + G]

            # Check ALL G completions: is_success=1 if any is correct
            is_success = 0
            best_comp = sample_completions[0] if sample_completions else ""
            for comp in sample_completions:
                matches = re.findall(r"\\boxed\{([^}]*)\}", comp)
                pred = matches[-1].strip() if matches else None
                if pred is not None and pred == gt:
                    is_success = 1
                    best_comp = comp
                    break

            # Prepend truncated suffix to get the full completion for entropy/metrics
            truncated_suffix = sample.get("truncated_suffix", "")
            full_comp = truncated_suffix + best_comp if truncated_suffix else best_comp

            # Compute entropy sequence for the full completion (suffix + new generation)
            try:
                h_seq = self._compute_entropy_for_completion(prompt_text, full_comp)
            except Exception:
                h_seq = []

            # Token length of the full resampled completion (suffix + generated)
            try:
                r_tokens_len = self._count_resampled_tokens(full_comp)
            except Exception:
                r_tokens_len = 0

            entry = {
                "alpha_used": self._p2_args.phase2_alpha,
                "beta_used": self._p2_args.phase2_beta,
                "is_insert_hint": self._p2_args.phase2_insert_hint,
                "question": q,
                "sample_id": sid,
                "is_success": is_success,
                "chosen_truncation_index": sample.get("chosen_truncation_index"),
                "spike_delta_H": sample.get("spike_delta_H"),
                "spike_absolute_H": sample.get("spike_absolute_H"),
                "resampled_H_sequence": h_seq,
                "discarded_tokens_len": sample.get("discarded_tokens_len", 0),
                "resampled_tokens_len": r_tokens_len,
            }
            self._p2_metrics.append(entry)
            batch_entries.append(entry)

        self._flush_metrics(batch_entries)  # append new lines to file immediately

    # ── terminal display ─────────────────────────────────────────────────

    # ── entropy helpers ──────────────────────────────────────────────────
    def _compute_entropy_for_completion(self, prompt_text: str, completion_text: str) -> list[float]:
        """Forward pass to extract Shannon entropy at each .\\n\\n / ?\\n\\n token in the completion suffix."""
        full_text = prompt_text + completion_text
        enc = self.processing_class(full_text, return_tensors="pt")
        input_ids = enc.input_ids.to(self.model.device)

        prompt_enc = self.processing_class(prompt_text, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.size(1)

        model = self.model
        was_training = model.training
        if was_training:
            model.eval()

        with torch.no_grad():
            logits = model(input_ids).logits[0].float()  # [seq_len, vocab]

        if was_training:
            model.train()

        seq_len = logits.size(0)
        entropies = []
        for pos in range(prompt_len, seq_len):
            if pos == 0:
                continue
            token_id = input_ids[0, pos].item()
            token_text = self.processing_class.decode([token_id], skip_special_tokens=True)
            if not (token_text.endswith(".\n\n") or token_text.endswith("?\n\n")):
                continue
            logit = logits[pos - 1]
            probs = torch.softmax(logit, dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)

        return entropies

    def _count_resampled_tokens(self, completion_text: str) -> int:
        """Count tokens in the resampled completion suffix."""
        enc = self.processing_class(completion_text, return_tensors="pt")
        return enc.input_ids.size(1)


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Phase 2 script parameters {script_args}")

    init_wandb_training(training_args)

    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    use_absolute_h = getattr(script_args, 'phase2_use_absolute_h', False)
    if use_absolute_h:
        logger.info("Ablation: using absolute entropy H (w/o ΔH trigger)")
    if not script_args.phase2_insert_hint:
        logger.info("Ablation: resampling without hint (w/o hint v)")

    logger.info(f"Building Phase 2 dataset from {script_args.phase2_input_json}")
    dataset = build_phase2_dataset(
        input_json=script_args.phase2_input_json,
        tokenizer=tokenizer,
        system_prompt=training_args.system_prompt or "",
        alpha=script_args.phase2_alpha,
        beta=script_args.phase2_beta,
        insert_hint=script_args.phase2_insert_hint,
        hint_text=script_args.phase2_hint_text,
        skip_samples=script_args.phase2_skip_samples,
        use_absolute_h=use_absolute_h,
    )

    reward_funcs = get_reward_funcs(script_args)

    trainer = Phase2Trainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        script_args=script_args,
    )

    logger.info("*** Phase 2 Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name="phase2", tags=["open-r1", "phase2"])
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    trainer._flush_metrics()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name="phase2", tags=["open-r1", "phase2"])


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
