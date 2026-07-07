"""
Beyond-the-80-20-Rule: High-Entropy GRPO training.
Only top entropy_top_ratio (default 0.2) response tokens contribute to the loss.

Monkey-patches GRPOTrainer._compute_loss to add entropy-based token masking.
Usage: same as grpo.py. Add `entropy_top_ratio: 0.2` to the YAML config.
"""
import logging
import os
import sys

import datasets
import torch
import torch.nn.functional as F
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


def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy H = -sum(p * log(p)) from logits over vocab dim."""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum(dim=-1)  # (B, L)


def get_global_entropy_top_mask(entropies, completion_mask, top_ratio=0.2):
    """Create a binary mask selecting top `top_ratio` fraction of tokens by entropy.

    Args:
        entropies: (B, L) per-token Shannon entropy
        completion_mask: (B, L) bool mask for completion tokens
        top_ratio: fraction of tokens to keep (0.0-1.0)
    Returns:
        entropy_top_mask: (B, L) bool mask
    """
    device = entropies.device
    # Collect all completion token entropies
    masked_entropies = entropies * completion_mask.float()
    valid_entropies = masked_entropies[completion_mask.bool()]

    if valid_entropies.numel() == 0:
        return completion_mask.bool()

    # Find the threshold for top `top_ratio` fraction
    k = max(1, int(valid_entropies.numel() * top_ratio))
    threshold = torch.topk(valid_entropies, k, largest=True).values[-1]

    # Create mask: token is selected if its entropy >= threshold AND it's a completion token
    return (masked_entropies >= threshold) & completion_mask.bool()


def make_high_entropy_compute_loss():
    """Return a replacement _compute_loss that only uses high-entropy tokens."""

    def high_entropy_compute_loss(self, model, inputs):
        # Extract data (same as parent)
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # 1. Compute per-token log probabilities (standard)
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # 2. Compute entropy from logits (additional forward pass for entropy)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            completion_logits = logits[:, -logits_to_keep:, :]
            entropies = compute_entropy_from_logits(completion_logits)  # (B, L_keep)

        # 3. Get entropy_top_ratio from config (default 0.2)
        top_ratio = getattr(self.args, 'entropy_top_ratio', 0.2)

        # 4. Create mask for top-entropy tokens only
        entropy_top_mask = get_global_entropy_top_mask(entropies, completion_mask, top_ratio)
        # Combine with completion_mask — only train on high-entropy completion tokens
        effective_mask = entropy_top_mask.float() * completion_mask.float()  # (B, L_keep)

        # 5. Compute KL divergence if needed
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
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # 6. Compute policy loss with effective_mask (high-entropy tokens only)
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

        # Apply entropy mask — only high-entropy tokens contribute to loss
        if self.loss_type == "grpo":
            loss = ((per_token_loss * effective_mask).sum(-1) / effective_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * effective_mask).sum() / effective_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * effective_mask).sum() / (per_token_loss.size(0) * per_token_loss.size(1))
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * effective_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log metrics
        mode = "train" if model.training else "eval"
        if "reward" not in self._metrics[mode]:
            self._metrics[mode]["reward"] = []
        self._metrics[mode]["reward"].append(advantages.mean().item())

        # Log what fraction of tokens are being used
        frac_used = effective_mask.sum() / completion_mask.sum().clamp(min=1.0)
        if "entropy_frac_used" not in self._metrics[mode]:
            self._metrics[mode]["entropy_frac_used"] = []
        self._metrics[mode]["entropy_frac_used"].append(
            self.accelerator.gather(frac_used).mean().item()
        )

        # Log clip ratio
        clip_ratio = (coef_2 < coef_1).float() * effective_mask
        clip_ratio = clip_ratio.sum() / effective_mask.sum().clamp(min=1.0)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather(clip_ratio).mean().item()
        )

        if self.beta != 0.0:
            mean_kl = (per_token_kl * effective_mask).sum() / effective_mask.sum().clamp(min=1.0)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())

        return loss

    return high_entropy_compute_loss


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

    # Always monkey-patch for high-entropy token masking (top_ratio=0.2)
    logger.info("*** High-Entropy mode: only top 20% tokens contribute to loss ***")
    fn = make_high_entropy_compute_loss()
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
