import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist


@dataclass
class ZeroAccCollectorConfig:
    enabled: bool = False
    output_path: str = "zero_acc_epoch_prepost.jsonl"
    target_epoch: int = 0
    num_generations: int = 4
    train_only: bool = True
    debug_every_steps: int = 50


class ZeroAccEpochCollector:
    def __init__(self, cfg: ZeroAccCollectorConfig):
        self.cfg = cfg
        self._last_debug_step = -1

        if self.cfg.enabled and self._is_main_process():
            Path(self.cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
            # 每次启动清空，避免混入旧实验
            with open(self.cfg.output_path, "w", encoding="utf-8"):
                pass

    @staticmethod
    def _is_dist() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

    @staticmethod
    def _world_size() -> int:
        return dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

    def _is_main_process(self) -> bool:
        return self._rank() == 0

    @staticmethod
    def _find_acc_idx(reward_func_names: list[str]) -> Optional[int]:
        for i, name in enumerate(reward_func_names):
            low = name.lower()
            if "accuracy" in low or "acc" in low:
                return i
        return None

    def _gather_payload(self, local_payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._is_dist():
            return [local_payload]
        gathered = [None for _ in range(self._world_size())]
        dist.all_gather_object(gathered, local_payload)
        return gathered

    def capture_pre_post(
        self,
        *,
        mode: str,
        epoch_float: float,
        global_step: int,
        reward_func_names: list[str],
        pre_rewards_per_func: torch.Tensor,   # [B_local, R]
        post_rewards_per_func: torch.Tensor,  # [B_local, R]
        prompts: list[Any],                   # len B_local
        pre_completions: list[Any],           # len B_local
        post_completions: list[Any],          # len B_local
        is_regenerated: Optional[list[bool]] = None,
    ) -> None:
        if not self.cfg.enabled:
            return
        if self.cfg.train_only and mode != "train":
            return
        if int(epoch_float if epoch_float is not None else 0) != self.cfg.target_epoch:
            return

        acc_idx = self._find_acc_idx(reward_func_names)
        if acc_idx is None:
            return

        local_payload = {
            "pre_acc": pre_rewards_per_func[:, acc_idx].detach().float().cpu().tolist(),
            "post_acc": post_rewards_per_func[:, acc_idx].detach().float().cpu().tolist(),
            "prompts": prompts,
            "pre_completions": pre_completions,
            "post_completions": post_completions,
            "is_regenerated": is_regenerated if is_regenerated is not None else [False] * len(prompts),
        }

        gathered = self._gather_payload(local_payload)
        if not self._is_main_process():
            return

        pre_acc_all, post_acc_all = [], []
        prompts_all, pre_comp_all, post_comp_all, regen_all = [], [], [], []

        for part in gathered:
            pre_acc_all.extend(part["pre_acc"])
            post_acc_all.extend(part["post_acc"])
            prompts_all.extend(part["prompts"])
            pre_comp_all.extend(part["pre_completions"])
            post_comp_all.extend(part["post_completions"])
            regen_all.extend(part["is_regenerated"])

        G = self.cfg.num_generations
        n = (len(prompts_all) // G) * G
        if n == 0:
            return

        rows = []
        for s in range(0, n, G):
            pre_group = pre_acc_all[s:s + G]
            # 仅按“重生成前全零”筛选
            if all(abs(float(x)) <= 1e-12 for x in pre_group):
                rows.append(
                    {
                        "epoch": float(epoch_float if epoch_float is not None else 0.0),
                        "epoch_int": int(epoch_float if epoch_float is not None else 0),
                        "global_step": int(global_step),
                        "prompt": prompts_all[s],
                        "pre_completions": pre_comp_all[s:s + G],
                        "post_completions": post_comp_all[s:s + G],
                        "pre_acc_rewards": pre_group,
                        "post_acc_rewards": post_acc_all[s:s + G],
                        "is_regenerated_group": regen_all[s:s + G],
                    }
                )

        if rows:
            with open(self.cfg.output_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if (
            self.cfg.debug_every_steps > 0
            and global_step % self.cfg.debug_every_steps == 0
            and global_step != self._last_debug_step
        ):
            self._last_debug_step = global_step
            print(
                f"[ZeroAccCollector] step={global_step} epoch={epoch_float:.4f} "
                f"rows_written={len(rows)} out={self.cfg.output_path}"
            )