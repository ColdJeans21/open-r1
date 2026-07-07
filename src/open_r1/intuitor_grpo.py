"""
Intuitor-style GRPO training using self-certainty as the advantage signal.
Monkey-patches GRPOTrainer._compute_loss to use self-certainty instead of
external reward advantages when reward_weights sum to 0.

Usage: same as grpo.py, set reward_weights: [0.0] for pure self-certainty mode.
"""
import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


def compute_self_certainty(model, input_ids, attention_mask, logits_to_keep, batch_size=1):
    """Compute per-token self-certainty = logsumexp(logits) - mean(logits)."""
    all_sce = []
    for i in range(0, input_ids.size(0), batch_size):
        ids = input_ids[i:i + batch_size]
        mask = attention_mask[i:i + batch_size]
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        sce_chunk = logits[:, -logits_to_keep:, :]
        sce = torch.logsumexp(sce_chunk, dim=-1) - sce_chunk.mean(dim=-1)
        all_sce.append(sce)
    return torch.cat(all_sce, dim=0)


def make_intuitor_compute_loss():
    """Return a replacement _compute_loss that uses self-certainty advantages."""

    def intuitor_compute_loss(self, model, inputs):
        # Extract completion data (same as parent)
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # 1. Compute per-token log probabilities (same as parent)
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # 2. Compute self-certainty from the model
        sce_per_token = compute_self_certainty(
            model, input_ids, attention_mask, logits_to_keep, batch_size=1
        )

        # 3. Average over valid tokens → per-completion score
        output_lengths = completion_mask.sum(dim=1, dtype=torch.float32)
        masked_sce = sce_per_token * completion_mask.float()
        sce_scores = masked_sce.sum(dim=1) / output_lengths.clamp(min=1)

        # 4. Group-normalize within each prompt's G completions
        G = self.num_generations
        B = sce_scores.size(0) // G
        if B > 0 and G > 1:
            reshaped = sce_scores.view(B, G)
            mean_s = reshaped.mean(dim=1, keepdim=True)
            std_s = reshaped.std(dim=1, keepdim=True)
            sce_advantages = ((reshaped - mean_s) / (std_s + 1e-4)).view(-1)
        else:
            sce_advantages = sce_scores

        # 5. Replace advantages in inputs with self-certainty
        inputs = dict(inputs)  # shallow copy
        inputs["advantages"] = sce_advantages

        # 6. Compute KL divergence if needed
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # 7. Compute advantages and policy loss (same as parent, but with SCE advantages)
        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach()
            if inputs.get("old_per_token_logps") is None
            else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * per_token_loss.size(1))
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log metrics
        mode = "train" if model.training else "eval"
        self._metrics[mode]["sce_advantage"].append(
            self.accelerator.gather(sce_advantages).mean().item()
        )

        # Log clip ratio
        clip_ratio = (coef_2 < coef_1).float() * completion_mask.float()
        clip_ratio = clip_ratio.sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather(clip_ratio).mean().item()
        )

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).mean().item()
            )

        return loss

    return intuitor_compute_loss


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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    dataset = get_dataset(script_args)
    reward_funcs = get_reward_funcs(script_args)
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt_column = getattr(script_args, 'dataset_prompt_column', 'prompt')
        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")
        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    rw = getattr(training_args, 'reward_weights', [1.0])
    use_intuitor = len(rw) > 0 and sum(rw) == 0.0

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
        ),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    # Monkey-patch for Intuitor mode
    if use_intuitor:
        logger.info("*** Intuitor mode: replacing advantages with self-certainty ***")
        fn = make_intuitor_compute_loss()
        trainer._compute_loss = fn.__get__(trainer, type(trainer))

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.create_model_card()
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
