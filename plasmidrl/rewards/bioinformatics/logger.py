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
            # counts
            "ori_count": [],
            "promoter_count": [],
            "terminator_count": [],
            "marker_count": [],
            "cds_count": [],
            "repeat_regions": [],
        }

    def add_components(self, components: Dict[str, float], total_reward: float) -> None:
        self.component_buffer["ori"].append(float(components["ori"]))
        self.component_buffer["promoter"].append(float(components["promoter"]))
        self.component_buffer["terminator"].append(float(components["terminator"]))
        self.component_buffer["marker"].append(float(components["marker"]))
        self.component_buffer["cds"].append(float(components["cds"]))
        self.component_buffer["length_factor"].append(float(components["length_factor"]))
        self.component_buffer["total_reward"].append(float(total_reward))
        # counts (optional if present)
        if "ori_count" in components:
            self.component_buffer["ori_count"].append(float(components["ori_count"]))
        if "promoter_count" in components:
            self.component_buffer["promoter_count"].append(float(components["promoter_count"]))
        if "terminator_count" in components:
            self.component_buffer["terminator_count"].append(float(components["terminator_count"]))
        if "marker_count" in components:
            self.component_buffer["marker_count"].append(float(components["marker_count"]))
        if "cds_count" in components:
            self.component_buffer["cds_count"].append(float(components["cds_count"]))
        if "repeat_regions" in components:
            self.component_buffer["repeat_regions"].append(float(components["repeat_regions"]))

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step to log reward components."""
        # Always try to log if there's data, regardless of step
        if not self.component_buffer["total_reward"]:
            return
        
        # Only log at specified frequency
        if self.log_frequency > 0 and (state.global_step % self.log_frequency) != 0:
            return

        log_dict: Dict[str, Any] = {}
        
        # Log component statistics (normalized rewards and factors)
        for name, values in self.component_buffer.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float32)
            if name in ("ori", "promoter", "terminator", "marker", "cds", "length_factor", "total_reward"):
                base = f"reward_components/{name}"
                log_dict[f"{base}/mean"] = float(np.mean(arr))
                log_dict[f"{base}/std"] = float(np.std(arr))
                log_dict[f"{base}/min"] = float(np.min(arr))
                log_dict[f"{base}/max"] = float(np.max(arr))

        # Descriptive stats for raw counts: mean, p25, p75
        def _quantiles(arr: np.ndarray) -> Dict[str, float]:
            return {
                "mean": float(np.mean(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }
        for cname in ("ori_count", "promoter_count", "terminator_count", "marker_count", "cds_count", "repeat_regions"):
            values = self.component_buffer.get(cname, [])
            if values:
                arr = np.asarray(values, dtype=np.float32)
                qs = _quantiles(arr)
                base = f"reward_counts/{cname.replace('_count','') if cname.endswith('_count') else cname}"
                log_dict[f"{base}/mean"] = qs["mean"]
                log_dict[f"{base}/p25"] = qs["p25"]
                log_dict[f"{base}/p75"] = qs["p75"]

        # Log sample count
        log_dict["reward_components/sample_count"] = len(self.component_buffer["total_reward"])

        # Periodic histograms (every 10x log frequency)
        if (state.global_step % (self.log_frequency * 10)) == 0 and len(self.component_buffer["total_reward"]) > 0:
            log_dict["reward_histograms/total_reward"] = wandb.Histogram(self.component_buffer["total_reward"])  # type: ignore[arg-type]
            if self.component_buffer["ori"]:
                log_dict["reward_histograms/ori"] = wandb.Histogram(self.component_buffer["ori"])  # type: ignore[arg-type]
            if self.component_buffer["cds"]:
                log_dict["reward_histograms/cds"] = wandb.Histogram(self.component_buffer["cds"])  # type: ignore[arg-type]
            if self.component_buffer["marker"]:
                log_dict["reward_histograms/marker"] = wandb.Histogram(self.component_buffer["marker"])  # type: ignore[arg-type]

        # Log to wandb (don't specify step, let W&B use its own counter)
        if log_dict:
            try:
                wandb.log(log_dict)
                print(f"[RewardLogger] Logged {len(self.component_buffer['total_reward'])} samples at trainer step {state.global_step}")
            except Exception as e:
                print(f"[RewardLogger] Warning: Failed to log to wandb: {e}")
        
        # Clear buffers after logging
        for key in self.component_buffer:
            self.component_buffer[key] = []


