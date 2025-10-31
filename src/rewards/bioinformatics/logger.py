from typing import Dict, List, Any
import numpy as np
from transformers import TrainerCallback
import wandb


class RewardComponentLogger(TrainerCallback):
    """Log component-level rewards to W&B with buffering."""

    def __init__(self, log_frequency: int = 10):
        self.log_frequency = int(log_frequency)
        self.component_buffer: Dict[str, List[float]] = {
            "ori": [],
            "promoter": [],
            "terminator": [],
            "marker": [],
            "cds": [],
            "length_factor": [],
            "total_reward": [],
        }

    def add_components(self, components: Dict[str, float], total_reward: float) -> None:
        self.component_buffer["ori"].append(float(components["ori"]))
        self.component_buffer["promoter"].append(float(components["promoter"]))
        self.component_buffer["terminator"].append(float(components["terminator"]))
        self.component_buffer["marker"].append(float(components["marker"]))
        self.component_buffer["cds"].append(float(components["cds"]))
        self.component_buffer["length_factor"].append(float(components["length_factor"]))
        self.component_buffer["total_reward"].append(float(total_reward))

    def on_step_end(self, args, state, control, **kwargs):
        if self.log_frequency <= 0 or state.global_step == 0:
            return
        if (state.global_step % self.log_frequency) != 0:
            return
        if not self.component_buffer["total_reward"]:
            return

        log_dict: Dict[str, Any] = {}
        try:
            for name, values in self.component_buffer.items():
                if not values:
                    continue
                arr = np.asarray(values, dtype=np.float32)
                base = f"reward_components/{name}"
                log_dict[f"{base}/mean"] = float(np.mean(arr))
                log_dict[f"{base}/std"] = float(np.std(arr))
                log_dict[f"{base}/min"] = float(np.min(arr))
                log_dict[f"{base}/max"] = float(np.max(arr))

            # Simple violation counters
            ori_vals = self.component_buffer.get("ori", [])
            marker_vals = self.component_buffer.get("marker", [])
            cds_vals = self.component_buffer.get("cds", [])
            if ori_vals:
                log_dict["reward_violations/ori_perfect"] = float(sum(x >= 1.0 for x in ori_vals) / len(ori_vals))
            if marker_vals:
                log_dict["reward_violations/marker_perfect"] = float(sum(x >= 1.0 for x in marker_vals) / len(marker_vals))
            if cds_vals:
                log_dict["reward_violations/cds_zero"] = float(sum(x == 0.0 for x in cds_vals) / len(cds_vals))

            # Periodic histograms
            if (state.global_step % (self.log_frequency * 10)) == 0:
                log_dict["reward_histograms/total_reward"] = wandb.Histogram(self.component_buffer["total_reward"])  # type: ignore[arg-type]
                log_dict["reward_histograms/ori"] = wandb.Histogram(self.component_buffer["ori"])  # type: ignore[arg-type]
                log_dict["reward_histograms/cds"] = wandb.Histogram(self.component_buffer["cds"])  # type: ignore[arg-type]

            wandb.log(log_dict, step=int(state.global_step))
        finally:
            for key in self.component_buffer:
                self.component_buffer[key] = []


