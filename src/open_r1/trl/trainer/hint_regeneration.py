# """
# Hint-based regeneration module for GRPO training.

# Structure after regeneration:
#     Prompt: [Original Prompt] (unchanged)
#     Completion: [Truncated Part] + [Hint] + [Newly Generated Part]
    
# The Hint is part of the completion and DOES participate in advantage computation.
# This encourages the model to learn to generate self-check statements autonomously.
# """

# import re
# from contextlib import nullcontext
# from typing import Optional

# import torch
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# from ..models import unwrap_model_for_generation


# class HintRegenerator:
#     """
#     Handles hint-based regeneration for samples where all generations have zero accuracy.
#     The hint becomes part of the completion to encourage self-check behavior.
#     """
    
#     def __init__(
#         self,
#         processing_class,
#         generation_config,
#         num_generations: int,
#         truncate_ratio: float = 0.15,
#         hint_template: str = "\n\nWait, I think I made a mistake. Let me reconsider the problem step by step.\n\n",
#         regeneration_count: int = 1,
#     ):
#         self.processing_class = processing_class
#         self.generation_config = generation_config
#         self.num_generations = num_generations
#         self.truncate_ratio = truncate_ratio
#         self.hint_template = hint_template
#         self.regeneration_count = regeneration_count
        
#         # 预先编码 hint
#         self._hint_ids = None
    
#     @property
#     def hint_ids(self) -> torch.Tensor:
#         """Lazily encode hint template."""
#         if self._hint_ids is None:
#             self._hint_ids = self.processing_class.encode(
#                 self.hint_template,
#                 add_special_tokens=False,
#                 return_tensors="pt"
#             ).squeeze(0)
#         return self._hint_ids
    
#     def detect_all_zero_groups(
#         self,
#         rewards_per_func: torch.Tensor,
#         reward_func_names: list[str],
#     ) -> tuple[torch.Tensor, Optional[int], torch.Tensor]:
#         """Detect groups where all samples have zero accuracy."""
#         device = rewards_per_func.device
#         batch_size = rewards_per_func.size(0)
        
#         acc_reward_idx = None
#         for i, name in enumerate(reward_func_names):
#             name_lower = name.lower()
#             if "accuracy" in name_lower or "acc" in name_lower:
#                 acc_reward_idx = i
#                 break
        
#         if acc_reward_idx is None:
#             return (
#                 torch.zeros(batch_size, dtype=torch.bool, device=device),
#                 None,
#                 torch.arange(batch_size, device=device) // self.num_generations
#             )
        
#         acc_rewards = rewards_per_func[:, acc_reward_idx]
#         num_groups = batch_size // self.num_generations
#         acc_rewards_grouped = acc_rewards.view(num_groups, self.num_generations)
#         all_zero_groups = (acc_rewards_grouped == 0).all(dim=1)
#         all_zero_samples = all_zero_groups.repeat_interleave(self.num_generations)
#         group_indices = torch.arange(batch_size, device=device) // self.num_generations
        
#         return all_zero_samples, acc_reward_idx, group_indices
    
#     # def select_samples_to_regenerate(
#     #     self,
#     #     all_zero_samples: torch.Tensor,
#     #     group_indices: torch.Tensor,
#     # ) -> torch.Tensor:
#     #     """Select which samples to regenerate based on regeneration_count."""
#     #     device = all_zero_samples.device
        
#     #     if not all_zero_samples.any():
#     #         return torch.tensor([], dtype=torch.long, device=device)
        
#     #     all_zero_indices = all_zero_samples.nonzero(as_tuple=True)[0]
        
#     #     if self.regeneration_count == -1:
#     #         return all_zero_indices
        
#     #     unique_groups = group_indices[all_zero_indices].unique()
#     #     selected_indices = []
        
#     #     for group_id in unique_groups:
#     #         group_mask = group_indices[all_zero_indices] == group_id
#     #         group_sample_indices = all_zero_indices[group_mask]
#     #         num_to_select = min(self.regeneration_count, len(group_sample_indices))
#     #         selected_indices.append(group_sample_indices[:num_to_select])
        
#     #     if selected_indices:
#     #         return torch.cat(selected_indices)
#     #     return torch.tensor([], dtype=torch.long, device=device)

#     def select_samples_to_regenerate(
#     self,
#     all_zero_samples: torch.Tensor,
#     group_indices: torch.Tensor,
# ) -> torch.Tensor:
#         """
#         Select which samples to regenerate based on regeneration_count.
        
#         对于每个全零组，选择前 regeneration_count 个样本进行重采样。
#         regeneration_count = -1 表示选择该组的所有样本。
#         """
#         device = all_zero_samples.device
        
#         if not all_zero_samples.any():
#             return torch.tensor([], dtype=torch.long, device=device)
        
#         # 获取所有全零样本的索引
#         all_zero_indices = all_zero_samples.nonzero(as_tuple=True)[0]
        
#         # 如果 regeneration_count == -1，返回所有全零样本
#         if self.regeneration_count == -1:
#             return all_zero_indices
        
#         # 否则，按组选择前 regeneration_count 个样本
#         unique_groups = group_indices[all_zero_indices].unique()
#         selected_indices = []
        
#         for group_id in unique_groups:
#             # 找出属于这个组的全零样本索引
#             group_mask = group_indices[all_zero_indices] == group_id
#             group_sample_indices = all_zero_indices[group_mask]
            
#             # 选择前 regeneration_count 个（不超过该组的样本数和 num_generations）
#             num_to_select = min(self.regeneration_count, len(group_sample_indices), self.num_generations)
#             selected_indices.append(group_sample_indices[:num_to_select])
        
#         if selected_indices:
#             return torch.cat(selected_indices)
#         return torch.tensor([], dtype=torch.long, device=device)
    
#     def build_generation_inputs(
#         self,
#         prompt_ids: torch.Tensor,
#         prompt_mask: torch.Tensor,
#         completion_ids: torch.Tensor,
#         completion_mask: torch.Tensor,
#         regenerate_indices: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
#         """
#         Build inputs for generation.
        
#         Generation input: [Original Prompt] + [Truncated Part] + [Hint]
        
#         Returns:
#             - generation_input_ids: Input for model.generate()
#             - generation_attention_mask: Attention mask
#             - truncated_completions: List of truncated completion tensors
#             - hint_ids_list: List of hint tensors (for building final completion)
#         """
#         device = prompt_ids.device
#         num_regenerate = len(regenerate_indices)
        
#         generation_input_list = []
#         generation_mask_list = []
#         truncated_completions = []
#         hint_ids_list = []
        
#         hint_ids = self.hint_ids.to(device)
        
#         for idx in regenerate_indices:
#             idx = idx.item()
            
#             # Get original prompt (non-padded)
#             orig_prompt = prompt_ids[idx][prompt_mask[idx] == 1]
            
#             # Calculate truncation position
#             completion_length = completion_mask[idx].sum().item()
#             truncate_pos = max(1, int(completion_length * self.truncate_ratio))
            
#             # Get truncated completion
#             truncated_completion = completion_ids[idx][:truncate_pos]
#             truncated_completions.append(truncated_completion)
#             hint_ids_list.append(hint_ids.clone())
            
#             # Generation input: [Prompt] + [Truncated] + [Hint]
#             gen_input = torch.cat([orig_prompt, truncated_completion, hint_ids], dim=0)
#             gen_mask = torch.ones(gen_input.size(0), dtype=torch.long, device=device)
            
#             generation_input_list.append(gen_input)
#             generation_mask_list.append(gen_mask)
        
#         # Pad to same length (left padding for generation)
#         max_len = max(g.size(0) for g in generation_input_list)
#         padded_input_ids = torch.full(
#             (num_regenerate, max_len),
#             self.processing_class.pad_token_id,
#             dtype=torch.long,
#             device=device
#         )
#         padded_attention_mask = torch.zeros(
#             num_regenerate, max_len,
#             dtype=torch.long,
#             device=device
#         )
        
#         for i, (g_ids, g_mask) in enumerate(zip(generation_input_list, generation_mask_list)):
#             start_pos = max_len - g_ids.size(0)
#             padded_input_ids[i, start_pos:] = g_ids
#             padded_attention_mask[i, start_pos:] = g_mask
        
#         return padded_input_ids, padded_attention_mask, truncated_completions, hint_ids_list
    
#     def regenerate_and_build_completion(
#         self,
#         model,
#         accelerator,
#         is_fsdp_enabled: bool,
#         ds3_gather_for_generation: bool,
#         generation_input_ids: torch.Tensor,
#         generation_attention_mask: torch.Tensor,
#         truncated_completions: list[torch.Tensor],
#         hint_ids_list: list[torch.Tensor],
#         max_new_tokens: Optional[int] = None,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """
#         Generate new completions and build final completion.
        
#         Final completion structure: [Truncated Part] + [Hint] + [Newly Generated]
        
#         This structure encourages the model to learn:
#         1. The truncated part (what it originally generated)
#         2. The hint (self-check behavior)
#         3. The corrected generation
#         """
#         device = generation_input_ids.device
#         prompt_length = generation_input_ids.size(1)
        
#         generation_config = self.generation_config
#         if max_new_tokens is not None:
#             from transformers import GenerationConfig
#             generation_config = GenerationConfig(
#                 max_new_tokens=max_new_tokens,
#                 do_sample=self.generation_config.do_sample,
#                 pad_token_id=self.generation_config.pad_token_id,
#                 bos_token_id=self.generation_config.bos_token_id,
#                 eos_token_id=self.generation_config.eos_token_id,
#                 temperature=self.generation_config.temperature,
#                 top_p=self.generation_config.top_p,
#                 top_k=self.generation_config.top_k,
#             )
        
#         torch.cuda.empty_cache()
        
#         # Generate
#         with unwrap_model_for_generation(
#             model, accelerator, gather_deepspeed3_params=ds3_gather_for_generation
#         ) as unwrapped_model:
#             with (
#                 FSDP.summon_full_params(model, recurse=False)
#                 if is_fsdp_enabled
#                 else nullcontext()
#             ):
#                 with torch.no_grad():
#                     output_ids = unwrapped_model.generate(
#                         generation_input_ids,
#                         attention_mask=generation_attention_mask,
#                         generation_config=generation_config
#                     )
        
#         # Extract newly generated part (after the generation input)
#         newly_generated_ids = output_ids[:, prompt_length:]
        
#         # Build final completion: [Truncated] + [Hint] + [Newly Generated]
#         num_samples = len(truncated_completions)
#         final_completion_list = []
        
#         for i in range(num_samples):
#             truncated = truncated_completions[i]
#             hint = hint_ids_list[i]
#             newly_gen = newly_generated_ids[i]
            
#             # Trim newly generated at EOS
#             is_eos = newly_gen == self.processing_class.eos_token_id
#             if is_eos.any():
#                 eos_pos = is_eos.int().argmax().item()
#                 newly_gen = newly_gen[:eos_pos + 1]
            
#             # Final completion: [Truncated] + [Hint] + [Newly Generated]
#             final_completion = torch.cat([truncated, hint, newly_gen], dim=0)
#             final_completion_list.append(final_completion)
        
#         # Pad all final completions
#         max_completion_len = max(c.size(0) for c in final_completion_list)
#         final_completion_ids = torch.full(
#             (num_samples, max_completion_len),
#             self.processing_class.pad_token_id,
#             dtype=torch.long,
#             device=device
#         )
        
#         for i, comp in enumerate(final_completion_list):
#             final_completion_ids[i, :comp.size(0)] = comp
        
#         # Create completion mask
#         is_eos = final_completion_ids == self.processing_class.eos_token_id
#         eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
#         eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
#         sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
#         final_completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
#         torch.cuda.empty_cache()
        
#         return final_completion_ids, final_completion_mask
    
#     @staticmethod
#     def align_tensor_lengths(
#         original: torch.Tensor,
#         new: torch.Tensor,
#         pad_value: int,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Align two tensors to the same sequence length by padding."""
#         max_len = max(original.size(1), new.size(1))
#         device = original.device
        
#         if original.size(1) < max_len:
#             padding = torch.full(
#                 (original.size(0), max_len - original.size(1)),
#                 pad_value,
#                 dtype=original.dtype,
#                 device=device
#             )
#             original = torch.cat([original, padding], dim=1)
        
#         if new.size(1) < max_len:
#             padding = torch.full(
#                 (new.size(0), max_len - new.size(1)),
#                 pad_value,
#                 dtype=new.dtype,
#                 device=device
#             )
#             new = torch.cat([new, padding], dim=1)
        
#         return original, new
"""
Hint-based regeneration module for GRPO training.

Structure after regeneration:
    Prompt: [Original Prompt] (unchanged)
    Completion: [Truncated Part] + [Hint] + [Newly Generated Part]
    
The Hint is inserted after a high-entropy token's sentence within the 15%-35% range,
encouraging the model to learn self-check behavior at critical reasoning points.
"""

import re
from contextlib import nullcontext
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ..models import unwrap_model_for_generation


class HintRegenerator:
    """
    Handles hint-based regeneration for samples where all generations have zero accuracy.
    The hint is inserted after high-entropy tokens to encourage self-check behavior.
    """
    
    # 句子结束标点（用于找到 token 所在句子的末尾）
    SENTENCE_END_PATTERN = re.compile(r'[.!?。！？\n]')
    
    def __init__(
        self,
        processing_class,
        generation_config,
        num_generations: int,
        truncate_ratio: float = 0.15,  # fallback ratio
        hint_template: str = "\n\nWait, I think I made a mistake. Let me reconsider the problem step by step.\n\n",
        regeneration_count: int = 1,
        # 新增参数：高熵 token 检测
        entropy_search_start_ratio: float = 0.15,  # 搜索区间起点
        entropy_search_end_ratio: float = 0.35,    # 搜索区间终点
        entropy_threshold: float = 2.0,            # 熵阈值（高于此值视为高熵）
        use_entropy_detection: bool = True,        # 是否启用熵检测
    ):
        self.processing_class = processing_class
        self.generation_config = generation_config
        self.num_generations = num_generations
        self.truncate_ratio = truncate_ratio
        self.hint_template = hint_template
        self.regeneration_count = regeneration_count
        
        # 高熵检测参数
        self.entropy_search_start_ratio = entropy_search_start_ratio
        self.entropy_search_end_ratio = entropy_search_end_ratio
        self.entropy_threshold = entropy_threshold
        self.use_entropy_detection = use_entropy_detection
        
        # 预先编码 hint
        self._hint_ids = None
    
    @property
    def hint_ids(self) -> torch.Tensor:
        """Lazily encode hint template."""
        if self._hint_ids is None:
            self._hint_ids = self.processing_class.encode(
                self.hint_template,
                add_special_tokens=False,
                return_tensors="pt"
            ).squeeze(0)
        return self._hint_ids
    
    def compute_token_entropy(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token entropy from logits.
        
        Args:
            logits: Shape (batch_size, seq_len, vocab_size)
            
        Returns:
            entropy: Shape (batch_size, seq_len)
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        # Compute entropy: -sum(p * log(p))
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy
    
    def find_high_entropy_position(
        self,
        token_entropy: torch.Tensor,
        completion_mask: torch.Tensor,
        completion_text: str,
        completion_ids: torch.Tensor,
    ) -> int:
        """
        Find the position to insert hint based on high-entropy token detection.
        
        Logic:
        1. Search within [15%, 35%] of the completion length
        2. Find tokens with entropy > threshold
        3. Select the one closest to 35%
        4. Find the end of the sentence containing this token
        
        Args:
            token_entropy: Per-token entropy values, shape (seq_len,)
            completion_mask: Mask for valid tokens, shape (seq_len,)
            completion_text: Decoded completion text
            completion_ids: Token IDs, shape (seq_len,)
            
        Returns:
            Token position to truncate at (insert hint after this position)
        """
        completion_length = completion_mask.sum().item()
        
        if completion_length < 10:
            # Too short, use fallback
            return max(1, int(completion_length * self.truncate_ratio))
        
        # Define search range
        search_start = int(completion_length * self.entropy_search_start_ratio)
        search_end = int(completion_length * self.entropy_search_end_ratio)
        search_start = max(1, search_start)
        search_end = min(completion_length - 1, search_end)
        
        if search_start >= search_end:
            return max(1, int(completion_length * self.truncate_ratio))
        
        # Get entropy values in the search range
        search_entropy = token_entropy[search_start:search_end]
        
        # Find high-entropy tokens (above threshold)
        high_entropy_mask = search_entropy > self.entropy_threshold
        
        if not high_entropy_mask.any():
            # No high-entropy token found, use fallback (middle of search range)
            fallback_pos = (search_start + search_end) // 2
            return self._find_sentence_end_position(
                fallback_pos, completion_text, completion_ids, completion_mask
            )
        
        # Find the high-entropy token closest to the end of search range (35%)
        # Get indices of high-entropy tokens (relative to search_start)
        high_entropy_indices = high_entropy_mask.nonzero(as_tuple=True)[0]
        
        # Select the last one (closest to 35%)
        selected_relative_idx = high_entropy_indices[-1].item()
        selected_absolute_idx = search_start + selected_relative_idx
        
        # Find the end of the sentence containing this token
        truncate_pos = self._find_sentence_end_position(
            selected_absolute_idx, completion_text, completion_ids, completion_mask
        )
        
        return truncate_pos
    
    def _find_sentence_end_position(
        self,
        token_pos: int,
        completion_text: str,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> int:
        """
        Find the end of the sentence containing the given token position.
        
        Args:
            token_pos: Token position to start from
            completion_text: Full completion text
            completion_ids: Token IDs
            completion_mask: Completion mask
            
        Returns:
            Token position at the end of the sentence
        """
        completion_length = completion_mask.sum().item()
        valid_ids = completion_ids[:completion_length]
        
        # Decode tokens up to token_pos to find character position
        text_up_to_token = self.processing_class.decode(
            valid_ids[:token_pos + 1], 
            skip_special_tokens=True
        )
        char_pos = len(text_up_to_token)
        
        # Search for sentence end after this position
        remaining_text = completion_text[char_pos:]
        match = self.SENTENCE_END_PATTERN.search(remaining_text)
        
        if match:
            # Found sentence end
            sentence_end_char = char_pos + match.end()
            # Convert back to token position
            sentence_end_token = self._char_to_token_position(
                valid_ids, sentence_end_char, completion_text
            )
            # Make sure we don't exceed completion length
            return min(sentence_end_token, completion_length - 1)
        else:
            # No sentence end found, use current position + small offset
            return min(token_pos + 5, completion_length - 1)
    
    def _char_to_token_position(
        self,
        token_ids: torch.Tensor,
        char_pos: int,
        full_text: str,
    ) -> int:
        """
        Convert character position to token position using binary search.
        """
        seq_len = token_ids.size(0)
        
        low, high = 0, seq_len
        while low < high:
            mid = (low + high) // 2
            decoded = self.processing_class.decode(token_ids[:mid], skip_special_tokens=True)
            if len(decoded) < char_pos:
                low = mid + 1
            else:
                high = mid
        
        return min(low, seq_len - 1)
    
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
            num_to_select = min(self.regeneration_count, len(group_sample_indices), self.num_generations)
            selected_indices.append(group_sample_indices[:num_to_select])
        
        if selected_indices:
            return torch.cat(selected_indices)
        return torch.tensor([], dtype=torch.long, device=device)
    
    def build_generation_inputs_with_entropy(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        completion_logits: torch.Tensor,  # 新增：用于计算熵
        completions_text: List[str],       # 新增：解码后的文本
        regenerate_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[int]]:
        """
        Build inputs for generation using entropy-based truncation.
        
        Args:
            prompt_ids: Prompt token IDs
            prompt_mask: Prompt attention mask
            completion_ids: Completion token IDs
            completion_mask: Completion attention mask
            completion_logits: Logits for completion tokens, shape (batch, seq_len, vocab_size)
            completions_text: List of decoded completion texts
            regenerate_indices: Indices of samples to regenerate
            
        Returns:
            - generation_input_ids: Padded input IDs for generation
            - generation_attention_mask: Attention mask
            - truncated_completions: List of truncated completion tensors
            - hint_ids_list: List of hint tensors
            - truncate_positions: List of truncate positions (for logging)
        """
        device = prompt_ids.device
        num_regenerate = len(regenerate_indices)
        
        generation_input_list = []
        generation_mask_list = []
        truncated_completions = []
        hint_ids_list = []
        truncate_positions = []
        
        hint_ids = self.hint_ids.to(device)
        
        # Compute entropy for all completion tokens
        if self.use_entropy_detection and completion_logits is not None:
            all_entropy = self.compute_token_entropy(completion_logits)  # (batch, seq_len)
        else:
            all_entropy = None
        
        for i, idx in enumerate(regenerate_indices):
            idx_val = idx.item()
            
            # Get original prompt (non-padded)
            orig_prompt = prompt_ids[idx_val][prompt_mask[idx_val] == 1]
            
            # Get completion info
            comp_ids = completion_ids[idx_val]
            comp_mask = completion_mask[idx_val]
            comp_text = completions_text[idx_val]
            if isinstance(comp_text, list):
                comp_text = comp_text[0]["content"] if comp_text else ""
            
            # Find truncation position
            if self.use_entropy_detection and all_entropy is not None:
                token_entropy = all_entropy[idx_val]
                truncate_pos = self.find_high_entropy_position(
                    token_entropy=token_entropy,
                    completion_mask=comp_mask,
                    completion_text=comp_text,
                    completion_ids=comp_ids,
                )
            else:
                # Fallback to fixed ratio
                completion_length = comp_mask.sum().item()
                truncate_pos = max(1, int(completion_length * self.truncate_ratio))
            
            truncate_positions.append(truncate_pos)
            
            # Get truncated completion
            truncated_completion = comp_ids[:truncate_pos]
            truncated_completions.append(truncated_completion)
            hint_ids_list.append(hint_ids.clone())
            
            # Generation input: [Prompt] + [Truncated] + [Hint]
            gen_input = torch.cat([orig_prompt, truncated_completion, hint_ids], dim=0)
            gen_mask = torch.ones(gen_input.size(0), dtype=torch.long, device=device)
            
            generation_input_list.append(gen_input)
            generation_mask_list.append(gen_mask)
        
        # Pad to same length (left padding for generation)
        max_len = max(g.size(0) for g in generation_input_list)
        padded_input_ids = torch.full(
            (num_regenerate, max_len),
            self.processing_class.pad_token_id,
            dtype=torch.long,
            device=device
        )
        padded_attention_mask = torch.zeros(
            num_regenerate, max_len,
            dtype=torch.long,
            device=device
        )
        
        for i, (g_ids, g_mask) in enumerate(zip(generation_input_list, generation_mask_list)):
            start_pos = max_len - g_ids.size(0)
            padded_input_ids[i, start_pos:] = g_ids
            padded_attention_mask[i, start_pos:] = g_mask
        
        return padded_input_ids, padded_attention_mask, truncated_completions, hint_ids_list, truncate_positions
    
    def build_generation_inputs(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        regenerate_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Build inputs for generation (fallback without entropy).
        """
        device = prompt_ids.device
        num_regenerate = len(regenerate_indices)
        
        generation_input_list = []
        generation_mask_list = []
        truncated_completions = []
        hint_ids_list = []
        
        hint_ids = self.hint_ids.to(device)
        
        for idx in regenerate_indices:
            idx = idx.item()
            
            orig_prompt = prompt_ids[idx][prompt_mask[idx] == 1]
            
            completion_length = completion_mask[idx].sum().item()
            truncate_pos = max(1, int(completion_length * self.truncate_ratio))
            
            truncated_completion = completion_ids[idx][:truncate_pos]
            truncated_completions.append(truncated_completion)
            hint_ids_list.append(hint_ids.clone())
            
            gen_input = torch.cat([orig_prompt, truncated_completion, hint_ids], dim=0)
            gen_mask = torch.ones(gen_input.size(0), dtype=torch.long, device=device)
            
            generation_input_list.append(gen_input)
            generation_mask_list.append(gen_mask)
        
        max_len = max(g.size(0) for g in generation_input_list)
        padded_input_ids = torch.full(
            (num_regenerate, max_len),
            self.processing_class.pad_token_id,
            dtype=torch.long,
            device=device
        )
        padded_attention_mask = torch.zeros(
            num_regenerate, max_len,
            dtype=torch.long,
            device=device
        )
        
        for i, (g_ids, g_mask) in enumerate(zip(generation_input_list, generation_mask_list)):
            start_pos = max_len - g_ids.size(0)
            padded_input_ids[i, start_pos:] = g_ids
            padded_attention_mask[i, start_pos:] = g_mask
        
        return padded_input_ids, padded_attention_mask, truncated_completions, hint_ids_list
    
    def regenerate_and_build_completion(
        self,
        model,
        accelerator,
        is_fsdp_enabled: bool,
        ds3_gather_for_generation: bool,
        generation_input_ids: torch.Tensor,
        generation_attention_mask: torch.Tensor,
        truncated_completions: list[torch.Tensor],
        hint_ids_list: list[torch.Tensor],
        max_new_tokens: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate new completions and build final completion.
        
        Final completion structure: [Truncated Part] + [Hint] + [Newly Generated]
        """
        device = generation_input_ids.device
        prompt_length = generation_input_ids.size(1)
        
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
                    output_ids = unwrapped_model.generate(
                        generation_input_ids,
                        attention_mask=generation_attention_mask,
                        generation_config=generation_config
                    )
        
        newly_generated_ids = output_ids[:, prompt_length:]
        
        num_samples = len(truncated_completions)
        final_completion_list = []
        
        for i in range(num_samples):
            truncated = truncated_completions[i]
            hint = hint_ids_list[i]
            newly_gen = newly_generated_ids[i]
            
            is_eos = newly_gen == self.processing_class.eos_token_id
            if is_eos.any():
                eos_pos = is_eos.int().argmax().item()
                newly_gen = newly_gen[:eos_pos + 1]
            
            final_completion = torch.cat([truncated, hint, newly_gen], dim=0)
            final_completion_list.append(final_completion)
        
        max_completion_len = max(c.size(0) for c in final_completion_list)
        final_completion_ids = torch.full(
            (num_samples, max_completion_len),
            self.processing_class.pad_token_id,
            dtype=torch.long,
            device=device
        )
        
        for i, comp in enumerate(final_completion_list):
            final_completion_ids[i, :comp.size(0)] = comp
        
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