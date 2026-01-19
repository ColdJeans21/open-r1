"""
Hint-based regeneration module for GRPO training.
Memory-optimized version.

Structure after regeneration:
    New Prompt: [Original Prompt] + [Hint]
    New Completion: [Truncated Part] + [Newly Generated Completion]
    
The Hint is part of the prompt and does NOT participate in advantage computation.
The Truncated Part + Newly Generated Completion form the new completion and DO participate in advantage computation.
"""

import re
from contextlib import nullcontext
from typing import Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ..models import unwrap_model_for_generation


class HintRegenerator:
    """
    Handles hint-based regeneration for samples where all generations have zero accuracy.
    """
    
    def __init__(
        self,
        processing_class,
        generation_config,
        num_generations: int,
        truncate_ratio: float = 0.15,
        hint_template: str = "\n\nWait, I think I made a mistake. Let me reconsider the problem step by step and verify my calculations carefully.\n\n",
        regeneration_count: int = 1,
    ):
        self.processing_class = processing_class
        self.generation_config = generation_config
        self.num_generations = num_generations
        self.truncate_ratio = truncate_ratio
        self.hint_template = hint_template
        self.regeneration_count = regeneration_count
    
    # def extract_answer_value(self, answer: str) -> str:
    #     """Extract the numerical answer from various formats."""
    #     if not answer:
    #         return ""
        
    #     match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", str(answer))
    #     if match:
    #         return match.group(1).replace(",", "")
        
    #     match = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$", str(answer))
    #     if match:
    #         return match.group(1).replace(",", "")
        
    #     return str(answer)
    
    def detect_all_zero_groups(
        self,
        rewards_per_func: torch.Tensor,
        reward_func_names: list[str],
    ) -> tuple[torch.Tensor, Optional[int], torch.Tensor]:
        """Detect groups where all samples have zero accuracy."""
        device = rewards_per_func.device
        batch_size = rewards_per_func.size(0)
        
        acc_reward_idx = None
        for i, name in enumerate(reward_func_names):
            name_lower = name.lower()
            if "accuracy" in name_lower or "acc" in name_lower:
                acc_reward_idx = i
                break
        
        if acc_reward_idx is None:
            return (
                torch.zeros(batch_size, dtype=torch.bool, device=device),
                None,
                torch.arange(batch_size, device=device) // self.num_generations
            )
        
        acc_rewards = rewards_per_func[:, acc_reward_idx]
        num_groups = batch_size // self.num_generations
        acc_rewards_grouped = acc_rewards.view(num_groups, self.num_generations)
        all_zero_groups = (acc_rewards_grouped == 0).all(dim=1)
        all_zero_samples = all_zero_groups.repeat_interleave(self.num_generations)
        group_indices = torch.arange(batch_size, device=device) // self.num_generations
        
        return all_zero_samples, acc_reward_idx, group_indices
    
    def select_samples_to_regenerate(
        self,
        all_zero_samples: torch.Tensor,
        group_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select which samples to regenerate based on regeneration_count."""
        device = all_zero_samples.device
        
        if not all_zero_samples.any():
            return torch.tensor([], dtype=torch.long, device=device)
        
        all_zero_indices = all_zero_samples.nonzero(as_tuple=True)[0]
        
        if self.regeneration_count == -1:
            return all_zero_indices
        
        unique_groups = group_indices[all_zero_indices].unique()
        selected_indices = []
        
        for group_id in unique_groups:
            group_mask = group_indices[all_zero_indices] == group_id
            group_sample_indices = all_zero_indices[group_mask]
            num_to_select = min(self.regeneration_count, len(group_sample_indices))
            selected_indices.append(group_sample_indices[:num_to_select])
        
        if selected_indices:
            return torch.cat(selected_indices)
        return torch.tensor([], dtype=torch.long, device=device)
    
    def build_hint_prompts_and_get_truncated(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        # correct_answers: list[str],
        regenerate_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[torch.Tensor]]:
        """
        Build new prompts: [Original Prompt] + [Hint]
        Also return the truncated completion parts to be prepended to new completions.
        
        Returns:
            - padded_prompt_ids: New prompts with hint
            - padded_prompt_mask: Attention mask for new prompts
            - hint_lengths: Length of hint for each sample
            - truncated_completions: List of truncated completion tensors
        """
        device = prompt_ids.device
        num_regenerate = len(regenerate_indices)
        
        new_prompt_ids_list = []
        new_prompt_mask_list = []
        hint_lengths = []
        truncated_completions = []  # Store truncated parts
        
        hint_ids = self.processing_class.encode(
            self.hint_template,
            add_special_tokens=False,
            return_tensors="pt"
        ).squeeze(0).to(device)
        hint_length = hint_ids.size(0)

        # for i, idx in enumerate(regenerate_indices):
        #     idx = idx.item()
            
        #     # Get original prompt tokens (non-padded, left-padded format)
        #     orig_prompt = prompt_ids[idx][prompt_mask[idx] == 1]
            
        #     # Calculate truncation position
        #     completion_length = completion_mask[idx].sum().item()
        #     truncate_pos = max(1, int(completion_length * self.truncate_ratio))
            
        #     # Get truncated completion (this will be prepended to new completion later)
        #     truncated_completion = completion_ids[idx][:truncate_pos]
        #     truncated_completions.append(truncated_completion)
            
        #     # Build hint text
        #     answer = correct_answers[i] if i < len(correct_answers) else ""
        #     answer_value = self.extract_answer_value(answer)
        #     hint_text = self.hint_template.format(answer=answer_value)
            
        #     # Encode hint
        #     hint_ids = self.processing_class.encode(
        #         hint_text,
        #         add_special_tokens=False,
        #         return_tensors="pt"
        #     ).squeeze(0).to(device)
            
        #     hint_lengths.append(hint_ids.size(0))
            
        #     # New prompt structure: [Original Prompt] + [Hint]
        #     # The truncated_completion will be used as the beginning of the completion
        #     new_prompt = torch.cat([orig_prompt, hint_ids], dim=0)
        #     new_mask = torch.ones(new_prompt.size(0), dtype=torch.long, device=device)
            
        #     new_prompt_ids_list.append(new_prompt)
        #     new_prompt_mask_list.append(new_mask)
        
        # # Pad to same length (left padding)
        # max_len = max(p.size(0) for p in new_prompt_ids_list)
        # padded_prompt_ids = torch.full(
        #     (num_regenerate, max_len),
        #     self.processing_class.pad_token_id,
        #     dtype=torch.long,
        #     device=device
        # )
        # padded_prompt_mask = torch.zeros(
        #     num_regenerate, max_len,
        #     dtype=torch.long,
        #     device=device
        # )
        
        # for i, (p_ids, p_mask) in enumerate(zip(new_prompt_ids_list, new_prompt_mask_list)):
        #     start_pos = max_len - p_ids.size(0)
        #     padded_prompt_ids[i, start_pos:] = p_ids
        #     padded_prompt_mask[i, start_pos:] = p_mask
        
        # return padded_prompt_ids, padded_prompt_mask, hint_lengths, truncated_completions
        for i, idx in enumerate(regenerate_indices):
            idx = idx.item()
            
            # Get original prompt tokens (non-padded, left-padded format)
            orig_prompt = prompt_ids[idx][prompt_mask[idx] == 1]
            
            # Calculate truncation position
            completion_length = completion_mask[idx].sum().item()
            truncate_pos = max(1, int(completion_length * self.truncate_ratio))
            
            # Get truncated completion
            truncated_completion = completion_ids[idx][:truncate_pos]
            truncated_completions.append(truncated_completion)
            
            hint_lengths.append(hint_length)
            
            # New prompt structure: [Original Prompt] + [Hint]
            new_prompt = torch.cat([orig_prompt, hint_ids], dim=0)
            new_mask = torch.ones(new_prompt.size(0), dtype=torch.long, device=device)
            
            new_prompt_ids_list.append(new_prompt)
            new_prompt_mask_list.append(new_mask)
        
        # Pad to same length (left padding)
        max_len = max(p.size(0) for p in new_prompt_ids_list)
        padded_prompt_ids = torch.full(
            (num_regenerate, max_len),
            self.processing_class.pad_token_id,
            dtype=torch.long,
            device=device
        )
        padded_prompt_mask = torch.zeros(
            num_regenerate, max_len,
            dtype=torch.long,
            device=device
        )
        
        for i, (p_ids, p_mask) in enumerate(zip(new_prompt_ids_list, new_prompt_mask_list)):
            start_pos = max_len - p_ids.size(0)
            padded_prompt_ids[i, start_pos:] = p_ids
            padded_prompt_mask[i, start_pos:] = p_mask
        
        return padded_prompt_ids, padded_prompt_mask, hint_lengths, truncated_completions
    
    # def regenerate_completions(
    #     self,
    #     model,
    #     accelerator,
    #     is_fsdp_enabled: bool,
    #     ds3_gather_for_generation: bool,
    #     padded_prompt_ids: torch.Tensor,
    #     padded_prompt_mask: torch.Tensor,
    #     truncated_completions: list[torch.Tensor],
    #     max_new_tokens: Optional[int] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Generate new completions and prepend truncated parts.
        
    #     Final completion structure: [Truncated Part] + [Newly Generated]
        
    #     Returns:
    #         - final_completion_ids: [Truncated Part] + [Newly Generated]
    #         - final_completion_mask: Mask for final completion
    #     """
    #     device = padded_prompt_ids.device
    #     prompt_length = padded_prompt_ids.size(1)
        
    #     # Create generation config with reduced length
    #     generation_config = self.generation_config
    #     if max_new_tokens is not None:
    #         from transformers import GenerationConfig
    #         generation_config = GenerationConfig(
    #             max_new_tokens=max_new_tokens,
    #             do_sample=self.generation_config.do_sample,
    #             pad_token_id=self.generation_config.pad_token_id,
    #             bos_token_id=self.generation_config.bos_token_id,
    #             eos_token_id=self.generation_config.eos_token_id,
    #             temperature=self.generation_config.temperature,
    #             top_p=self.generation_config.top_p,
    #             top_k=self.generation_config.top_k,
    #         )
        
    #     torch.cuda.empty_cache()
        
    #     with unwrap_model_for_generation(
    #         model, accelerator, gather_deepspeed3_params=ds3_gather_for_generation
    #     ) as unwrapped_model:
    #         with (
    #             FSDP.summon_full_params(model, recurse=False)
    #             if is_fsdp_enabled
    #             else nullcontext()
    #         ):
    #             with torch.no_grad():
    #                 new_prompt_completion_ids = unwrapped_model.generate(
    #                     padded_prompt_ids,
    #                     attention_mask=padded_prompt_mask,
    #                     generation_config=generation_config
    #                 )
        
    #     # Extract newly generated part (after the prompt with hint)
    #     newly_generated_ids = new_prompt_completion_ids[:, prompt_length:]
        
    #     # Build final completion: [Truncated Part] + [Newly Generated]
    #     num_samples = len(truncated_completions)
    #     final_completion_list = []
        
    #     for i in range(num_samples):
    #         truncated = truncated_completions[i]
    #         newly_gen = newly_generated_ids[i]
            
    #         # Find where EOS is in newly generated (if any)
    #         is_eos = newly_gen == self.processing_class.eos_token_id
    #         if is_eos.any():
    #             eos_pos = is_eos.int().argmax().item()
    #             newly_gen = newly_gen[:eos_pos + 1]  # Include EOS
            
    #         # Concatenate: [Truncated] + [Newly Generated]
    #         final_completion = torch.cat([truncated, newly_gen], dim=0)
    #         final_completion_list.append(final_completion)
        
    #     # Pad all final completions to same length
    #     max_completion_len = max(c.size(0) for c in final_completion_list)
    #     final_completion_ids = torch.full(
    #         (num_samples, max_completion_len),
    #         self.processing_class.pad_token_id,
    #         dtype=torch.long,
    #         device=device
    #     )
        
    #     for i, comp in enumerate(final_completion_list):
    #         final_completion_ids[i, :comp.size(0)] = comp
        
    #     # Create completion mask
    #     is_eos = final_completion_ids == self.processing_class.eos_token_id
    #     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #     eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    #     sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #     final_completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
    #     torch.cuda.empty_cache()
        
    #     return final_completion_ids, final_completion_mask
    
    # @staticmethod
    # def align_tensor_lengths(
    #     original: torch.Tensor,
    #     new: torch.Tensor,
    #     pad_value: int,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Align two tensors to the same sequence length by padding."""
    #     max_len = max(original.size(1), new.size(1))
    #     device = original.device
        
    #     if original.size(1) < max_len:
    #         padding = torch.full(
    #             (original.size(0), max_len - original.size(1)),
    #             pad_value,
    #             dtype=original.dtype,
    #             device=device
    #         )
    #         original = torch.cat([original, padding], dim=1)
        
    #     if new.size(1) < max_len:
    #         padding = torch.full(
    #             (new.size(0), max_len - new.size(1)),
    #             pad_value,
    #             dtype=new.dtype,
    #             device=device
    #         )
    #         new = torch.cat([new, padding], dim=1)
        
    #     return original, new
    def regenerate_completions(
        self,
        model,
        accelerator,
        is_fsdp_enabled: bool,
        ds3_gather_for_generation: bool,
        padded_prompt_ids: torch.Tensor,
        padded_prompt_mask: torch.Tensor,
        truncated_completions: list[torch.Tensor],
        max_new_tokens: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate new completions and prepend truncated parts.
        
        Final completion structure: [Truncated Part] + [Newly Generated]
        """
        device = padded_prompt_ids.device
        prompt_length = padded_prompt_ids.size(1)
        
        generation_config = self.generation_config
        if max_new_tokens is not None:
            from transformers import GenerationConfig
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=self.generation_config.do_sample,
                pad_token_id=self.generation_config.pad_token_id,
                bos_token_id=self.generation_config.bos_token_id,
                eos_token_id=self.generation_config.eos_token_id,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
            )
        
        torch.cuda.empty_cache()
        
        with unwrap_model_for_generation(
            model, accelerator, gather_deepspeed3_params=ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(model, recurse=False)
                if is_fsdp_enabled
                else nullcontext()
            ):
                with torch.no_grad():
                    new_prompt_completion_ids = unwrapped_model.generate(
                        padded_prompt_ids,
                        attention_mask=padded_prompt_mask,
                        generation_config=generation_config
                    )
        
        # Extract newly generated part
        newly_generated_ids = new_prompt_completion_ids[:, prompt_length:]
        
        # Build final completion: [Truncated Part] + [Newly Generated]
        num_samples = len(truncated_completions)
        final_completion_list = []
        
        for i in range(num_samples):
            truncated = truncated_completions[i]
            newly_gen = newly_generated_ids[i]
            
            is_eos = newly_gen == self.processing_class.eos_token_id
            if is_eos.any():
                eos_pos = is_eos.int().argmax().item()
                newly_gen = newly_gen[:eos_pos + 1]
            
            final_completion = torch.cat([truncated, newly_gen], dim=0)
            final_completion_list.append(final_completion)
        
        # Pad all final completions
        max_completion_len = max(c.size(0) for c in final_completion_list)
        final_completion_ids = torch.full(
            (num_samples, max_completion_len),
            self.processing_class.pad_token_id,
            dtype=torch.long,
            device=device
        )
        
        for i, comp in enumerate(final_completion_list):
            final_completion_ids[i, :comp.size(0)] = comp
        
        # Create completion mask
        is_eos = final_completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        final_completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        torch.cuda.empty_cache()
        
        return final_completion_ids, final_completion_mask
    
    @staticmethod
    def align_tensor_lengths(
        original: torch.Tensor,
        new: torch.Tensor,
        pad_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align two tensors to the same sequence length by padding."""
        max_len = max(original.size(1), new.size(1))
        device = original.device
        
        if original.size(1) < max_len:
            padding = torch.full(
                (original.size(0), max_len - original.size(1)),
                pad_value,
                dtype=original.dtype,
                device=device
            )
            original = torch.cat([original, padding], dim=1)
        
        if new.size(1) < max_len:
            padding = torch.full(
                (new.size(0), max_len - new.size(1)),
                pad_value,
                dtype=new.dtype,
                device=device
            )
            new = torch.cat([new, padding], dim=1)
        
        return original, new