from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

import torch
from tensordict import TensorDictBase
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torchrl.envs.transforms import Transform

logger = logging.getLogger(__name__)


class SimpleMCAdvantage(Transform):
    """Monte Carlo advantage transform that operates on dense reward tensors."""

    def __init__(
        self,
        grpo_size: int,
        prompt_key: Any = ("text", "prompt"),
        rewards_key: Any = "reward",
        advantage_key: Any = "advantage",
        done_key: Any = "done",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.in_keys = [prompt_key, rewards_key, done_key]
        self.out_keys = [advantage_key]
        self.prompt_key = prompt_key
        self.rewards_key = rewards_key
        self.advantage_key = advantage_key
        self.done_key = done_key
        self.grpo_size = grpo_size
        self.verbose = verbose
        self.queues: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=grpo_size))

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase | None:
        if self.verbose:
            logger.debug(
                "MCAdvantage _inv_call received td shape=%s ndim=%s keys=%s",
                tensordict.shape,
                tensordict.ndim,
                list(tensordict.keys(True)),
            )
        if tensordict.ndim == 1:
            done_tensor = tensordict.get(self.done_key)
            if done_tensor is None:
                return None
            num_done = done_tensor.sum()
            if num_done > 1:
                done_idx = done_tensor.nonzero(as_tuple=True)[0] + 1
                splits = torch.cat([done_idx.new_zeros((1,)), done_idx], dim=0).diff()
                tensordicts = tensordict.split(splits)
                if self.verbose:
                    logger.debug(
                        "Splitting trajectory into %d parts for prompt processing",
                        len(tensordicts),
                    )
                tensordicts = [self._inv_call(td) for td in tensordicts]
                tensordicts = [td for td in tensordicts if td is not None]
                return torch.cat(tensordicts) if tensordicts else None
            if not tensordict[-1].get(self.done_key).all():
                raise RuntimeError("Expected the trajectory to be done.")
            prompt_val = tensordict[0].get(self.prompt_key)
            prompt_key = self._normalize_prompt(prompt_val)
            queue = self.queues[prompt_key]
            queue.append(tensordict)
            if self.verbose:
                logger.debug(
                    "Queued trajectory for prompt key=%s (queue_len=%d/%d)",
                    prompt_key,
                    len(queue),
                    self.grpo_size,
                )
            if len(queue) == self.grpo_size:
                stacked = torch.cat(list(queue), -1)
                del self.queues[prompt_key]
                reward = stacked.get(self.rewards_key)
                if reward is None:
                    if self.verbose:
                        logger.debug(
                            "Reward missing for prompt key=%s; queue retained", prompt_key
                        )
                    return None
                reward = reward.to(torch.float32)
                reward_mean = reward.mean()
                reward_scale = reward.std(unbiased=False)
                if torch.isnan(reward_scale) or torch.isinf(reward_scale) or reward_scale.item() < 1e-6:
                    if self.verbose:
                        logger.debug(
                            "Degenerate reward distribution for prompt key=%s (scale=%s); using unit scale",
                            prompt_key,
                            reward_scale.item() if torch.isfinite(reward_scale) else float('nan'),
                        )
                    reward_scale = reward.new_tensor(1.0)
                else:
                    reward_scale = reward_scale.clamp_min(1e-6)
                advantage = (reward - reward_mean) / reward_scale
                stacked.set(self.advantage_key, advantage)
                if self.verbose:
                    logger.debug(
                        "Computed advantage for prompt key=%s (mean=%s, std=%s)",
                        prompt_key,
                        reward_mean,
                        reward_scale,
                    )
                return stacked
            if self.verbose:
                logger.debug(
                    "Prompt key=%s still waiting for %d/%d trajectories",
                    prompt_key,
                    len(queue),
                    self.grpo_size,
                )
            return None
        if tensordict.ndim > 2:
            tensordict = tensordict.flatten(0, -2)
        trajs = tensordict.unbind(0)
        processed = []
        for traj in trajs:
            td_out = self._inv_call(traj)
            if td_out is None:
                continue
            processed.append(td_out)
        if processed:
            return torch.cat(processed, 0)
        return None

    def _normalize_prompt(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, NonTensorData):
            return self._normalize_prompt(prompt.data)
        if isinstance(prompt, NonTensorStack):
            return "|".join(self._normalize_prompt(item) for item in prompt)
        if isinstance(prompt, TensorDictBase):
            items = [
                f"{key}:{self._normalize_prompt(prompt.get(key))}"
                for key in sorted(prompt.keys())
            ]
            return "|".join(items)
        if isinstance(prompt, (list, tuple)):
            return "|".join(self._normalize_prompt(item) for item in prompt)
        if torch.is_tensor(prompt):
            return "|".join(str(x) for x in prompt.detach().cpu().flatten().tolist())
        return str(prompt)

    def pending_summary(self, limit: int = 5) -> Dict[str, Any]:
        """Summarise how many trajectories are queued per prompt key."""
        active: List[Tuple[str, int]] = [
            (prompt_key, len(queue))
            for prompt_key, queue in self.queues.items()
            if len(queue) > 0
        ]
        active.sort(key=lambda item: item[1], reverse=True)
        total_prompts = len(active)
        total_pending = sum(length for _, length in active)

        def _abbreviate(key: str) -> str:
            key = key.strip()
            if len(key) <= 48:
                return key
            return f"{key[:45]}…"

        top = [
            {"prompt": _abbreviate(prompt_key), "count": length}
            for prompt_key, length in active[:limit]
        ]
        truncated = max(0, total_prompts - limit)

        return {
            "total_prompts": total_prompts,
            "total_pending": total_pending,
            "top": top,
            "truncated": truncated,
        }
