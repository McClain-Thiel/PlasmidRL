import logging
import os

os.environ.setdefault("LIST_TO_STACK", "1")

import math
import time
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from datetime import datetime

from tensordict import set_list_to_stack
from torchrl.modules.llm import TransformersWrapper
from torchrl.collectors.llm import LLMCollector
from torchrl.objectives.llm import GRPOLoss
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import ListStorage
from src.objectives import SimpleMCAdvantage
from src.rewards.plasmid_informatics import RewardTransform
from src.utils.misc_transforms import DefaultQueryOnReset
from src.config import get_config
from src.envs.stateless_chat import StatelessChatEnv


# ----------------------
# Config & device
# ----------------------
set_list_to_stack(True).set()

config = get_config()
log_level = getattr(logging, str(config.log_level).upper(), logging.INFO)
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(level=log_level, format=log_format)
else:
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)

root_logger.setLevel(log_level)
logger = logging.getLogger("plasmidrl.runner")
logger.setLevel(log_level)

# Suppress noisy third-party loggers (e.g., HTTP clients)
for noisy_logger in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Initialize Weights & Biases
# ----------------------
if config.wandb_api_key:
    wandb.login(key=config.wandb_api_key.get_secret_value())
run_name = config.wandb_run_name or f"plasmidrl-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

wandb.init(
    project=config.wandb_project,
    entity=config.wandb_entity,
    name=run_name,
    tags=config.wandb_tags,
    notes=config.wandb_notes,
    config={
        # Model configuration
        "model": config.model,
        "device": str(device),
        "informatics_server_url": config.informatics_server_url,
        
        # Training hyperparameters
        "K": config.K,
        "num_iters": config.num_iters,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "max_new_tokens": config.max_new_tokens,
        "dialog_turns_per_batch": config.dialog_turns_per_batch,
        "reward_max_workers": config.reward_max_workers,
        "reward_log_timings": config.reward_log_timings,
        "replay_buffer_size": config.replay_buffer_size,
        
        # Logging configuration
        "log_interval": config.log_interval,
        "checkpoint_interval": config.checkpoint_interval,
        
        # Sequence information
        "default_query_length": len(config.default_query),
    }
)

# ----------------------
# HF tokenizer & model
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(
    config.model,
    token=config.huggingface_token.get_secret_value() if config.huggingface_token else None,
)

# causal decoders often need a pad token and left padding for chat RL
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if getattr(tokenizer, "padding_side", None) != "left":
    tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    config.model,
    token=config.huggingface_token.get_secret_value() if config.huggingface_token else None,
)
model.to(device)  # put underlying HF weights on device

max_model_length = getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
tokenizer.model_max_length = max_model_length
if hasattr(model, "generation_config") and getattr(model.generation_config, "max_length", None) is not None:
    model.generation_config.max_length = None
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# ----------------------
# TorchRL policy wrapper
# ----------------------
policy = TransformersWrapper(
    model=model,
    tokenizer=tokenizer,
    input_mode="text",
    return_log_probs=True,
    device=device,
    generate_kwargs={
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    },
).eval()  # keep eval() for GRPO rollouts/updates

# ----------------------
# Env + reward/initial query transforms
# ----------------------
batch_size = (config.dialog_turns_per_batch,)
env = StatelessChatEnv(
    input_mode="text",
    batch_size=batch_size,
    tokenizer=tokenizer,
    device=device,
)
env = env.append_transform(
    RewardTransform(
        config.informatics_server_url,
        max_workers=config.reward_max_workers,
        log_timings=config.reward_log_timings,
    )
)
env = env.append_transform(
    DefaultQueryOnReset([config.default_query] * config.dialog_turns_per_batch)
)

# ----------------------
# Collector
# ----------------------
collector = LLMCollector(
    policy=policy,
    env=env,
    dialog_turns_per_batch=config.dialog_turns_per_batch,
)

# ----------------------
# Replay buffer + MCAdvantage
# ----------------------
K = config.K  # number of samples per prompt to compute MC advantage

rb = ReplayBuffer(storage=ListStorage(max_size=config.replay_buffer_size))
advantage_transform = SimpleMCAdvantage(
    grpo_size=K,
    prompt_key=("text", "prompt"),
    rewards_key="reward",
    done_key="done",
    advantage_key="advantage",
    verbose=config.advantage_verbose,
)
rb = rb.append_transform(advantage_transform)

# ----------------------
# Loss (KL disabled unless you wire a ref policy/logits)
# ----------------------
loss_module = GRPOLoss(
    actor_network=policy,
    kl_to_ref_coeff=None,   # or a float if you explicitly provide reference KL
    masking_strategy="sft", # single-turn: SFT masks response tokens only
)

# ----------------------
# Optimizer
# ----------------------
optim = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

# ----------------------
# Training loop
# ----------------------
num_iters = config.num_iters

for outer in range(num_iters):
    # Track per-iteration metrics
    iter_rewards = []
    iter_losses = []
    iter_steps = 0
    iter_step_times: list[float] = []
    last_step_end: float | None = None
    
    # The collector yields one tensordict per dialog turn with dialog_turns_per_batch=1
    for step_idx, td in enumerate(collector):
        loop_start = time.perf_counter()
        wait_time = None if last_step_end is None else max(0.0, loop_start - last_step_end)
        # --- Make reward shape broadcastable & ensure both "reward" and ("next","reward") exist ---
        reward_tensor = None
        for key in (("next", "reward"), "reward"):
            if key in td.keys(True):
                val = td.get(key)
                if val is not None:
                    reward_tensor = val
                    break

        if reward_tensor is None:
            logger.warning(
                "[iter %s step %s] missing reward for batch; inserting zeros",
                outer,
                step_idx,
            )
            zeros = torch.zeros(td.batch_size + (1, 1), dtype=torch.float32)
            if td.device is not None:
                zeros = zeros.to(td.device)
            td.set(("next", "reward"), zeros)
            td.set("reward", zeros.clone())
        else:
            r = reward_tensor
            if isinstance(r, torch.Tensor):
                if r.ndim == len(td.batch_size):          # e.g., (B,)
                    r = r.unsqueeze(-1).unsqueeze(-1)     # -> (B,1,1)
                elif r.ndim == len(td.batch_size) + 1:    # e.g., (B,1)
                    r = r.unsqueeze(-1)                   # -> (B,1,1)
                if logger.isEnabledFor(logging.DEBUG):
                    r_cpu = r.detach().cpu()
                    logger.debug(
                        "[iter %s step %s] reward stats mean=%.4f min=%.4f max=%.4f (shape=%s)",
                        outer,
                        step_idx,
                        float(r_cpu.mean().item()),
                        float(r_cpu.min().item()),
                        float(r_cpu.max().item()),
                        tuple(r.shape),
                    )
            td.set(("next", "reward"), r)
            td.set("reward", r.clone() if isinstance(r, torch.Tensor) else r)

        # --- Force single-turn episodes to terminate ---
        for key in ["done", "terminated", ("next", "done"), ("next", "terminated")]:
            if key in td.keys(True):
                tensor = td.get(key)
                if tensor is not None:
                    ones = torch.ones_like(tensor, dtype=torch.bool, device=tensor.device)
                    td.set(key, ones)

        # --- Push to RB: MCAdvantage will fill "advantage" once it has K trajs for the same prompt ---
        rb.add(td)

        # Not every iter will have advantage ready; skip until MCAdvantage emits it
        if "advantage" not in td.keys():
            if step_idx % config.log_interval == 0:
                wait_str = f" wait={wait_time:.2f}s" if wait_time is not None else ""
                pending = advantage_transform.pending_summary(limit=3)
                if pending["total_prompts"]:
                    formatted_groups = ", ".join(
                        f"{item['prompt']} ({item['count']}/{K})" for item in pending["top"]
                    )
                    if pending["truncated"]:
                        formatted_groups += f", +{pending['truncated']} more"
                    pending_str = f" pending_prompts={pending['total_prompts']} [{formatted_groups}]"
                else:
                    pending_str = " pending_prompts=0"
                logger.info(
                    "[iter %s step %s] waiting for groups to fill (K=%s)…%s%s",
                    outer,
                    step_idx,
                    K,
                    wait_str,
                    pending_str,
                )
            last_step_end = loop_start
            continue

        # --- Compute GRPO loss and update (keep policy/model in eval mode) ---
        optim.zero_grad(set_to_none=True)
        loss_td = loss_module(td)          # produces "loss/*" scalars
        loss = sum(v for k, v in loss_td.items() if k.startswith("loss"))
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)

        optim.step()

        # --- Collect metrics for logging ---
        avg_r = float(td.get("reward").mean().item()) if "reward" in td.keys() else math.nan
        loss_val = loss.item()
        step_end = time.perf_counter()
        step_elapsed = step_end - loop_start
        last_step_end = step_end
        batch_elements = math.prod(td.batch_size) if td.batch_size else 1
        samples_per_sec = (batch_elements / step_elapsed) if step_elapsed > 0 else math.nan
        wait_logged = wait_time if wait_time is not None else math.nan
        
        iter_rewards.append(avg_r)
        iter_losses.append(loss_val)
        iter_step_times.append(step_elapsed)
        iter_steps += 1
        
        # Calculate global step for wandb
        global_step = outer * 1000 + step_idx  # Approximate global step
        
        # --- Enhanced logging ---
        if step_idx % config.log_interval == 0:
            wait_str = f" wait={wait_time:.2f}s" if wait_time is not None else ""
            logger.info(
                "[iter %s step %s] loss=%.4f avg_reward=%.3f step=%.2fs%s throughput=%.2f/s",
                outer,
                step_idx,
                loss_val,
                avg_r,
                step_elapsed,
                wait_str,
                samples_per_sec,
            )
            
            # Log individual step metrics to wandb
            log_dict = {
                "step/loss": loss_val,
                "step/reward": avg_r,
                "step/global_step": global_step,
                "step/iteration": outer,
                "step/step_idx": step_idx,
                "step/time": step_elapsed,
                "step/wait_time": wait_logged,
                "step/samples_per_sec": samples_per_sec,
            }
            
            # Log individual loss components if available
            for k, v in loss_td.items():
                if k.startswith("loss"):
                    log_dict[f"step/{k}"] = v.item()
            
            # Log advantage statistics if available
            if "advantage" in td.keys():
                advantage = td.get("advantage")
                log_dict.update({
                    "step/advantage_mean": float(advantage.mean().item()),
                    "step/advantage_std": float(advantage.std().item()),
                    "step/advantage_min": float(advantage.min().item()),
                    "step/advantage_max": float(advantage.max().item()),
                })
            
            wandb.log(log_dict, step=global_step)
    
    # --- Log iteration summary ---
    if iter_rewards:
        iter_avg_reward = sum(iter_rewards) / len(iter_rewards)
        iter_avg_loss = sum(iter_losses) / len(iter_losses)
        iter_max_reward = max(iter_rewards)
        iter_min_reward = min(iter_rewards)
        total_step_time = sum(iter_step_times) if iter_step_times else 0.0
        avg_step_time = (total_step_time / len(iter_step_times)) if iter_step_times else math.nan
        total_samples = len(iter_step_times) * config.dialog_turns_per_batch
        avg_throughput = (total_samples / total_step_time) if total_step_time > 0 else math.nan

        logger.info(
            "[ITER %s SUMMARY] steps=%s, avg_loss=%.4f, avg_reward=%.3f, reward_range=[%.3f, %.3f], "
            "avg_step_time=%.2fs, throughput=%.2f/s",
            outer,
            iter_steps,
            iter_avg_loss,
            iter_avg_reward,
            iter_min_reward,
            iter_max_reward,
            avg_step_time,
            avg_throughput,
        )
        
        # Log iteration summary to wandb
        wandb.log({
            "iteration/avg_loss": iter_avg_loss,
            "iteration/avg_reward": iter_avg_reward,
            "iteration/max_reward": iter_max_reward,
            "iteration/min_reward": iter_min_reward,
            "iteration/num_steps": iter_steps,
            "iteration/iteration": outer,
            "iteration/avg_step_time": avg_step_time,
            "iteration/samples_per_sec": avg_throughput,
        }, step=global_step)
    
    # --- Checkpoint saving ---
    if (outer + 1) % config.checkpoint_interval == 0 or outer == num_iters - 1:
        checkpoint_path = f"checkpoint_iter_{outer + 1}.pt"
        
        # Save model state
        checkpoint = {
            "iteration": outer + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "config": {
                "model": config.model,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "K": config.K,
                "max_grad_norm": config.max_grad_norm,
            },
            "global_step": global_step,
        }
        
        # Add recent performance metrics if available
        if iter_rewards:
            checkpoint["metrics"] = {
                "avg_reward": iter_avg_reward,
                "avg_loss": iter_avg_loss,
                "max_reward": iter_max_reward,
                "min_reward": iter_min_reward,
            }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved checkpoint: %s", checkpoint_path)
        
        # Upload checkpoint as wandb artifact
        artifact = wandb.Artifact(
            name=f"model-checkpoint-iter-{outer + 1}",
            type="model",
            description=f"Model checkpoint at iteration {outer + 1}",
            metadata={
                "iteration": outer + 1,
                "global_step": global_step,
                "avg_reward": iter_avg_reward if iter_rewards else None,
                "avg_loss": iter_avg_loss if iter_rewards else None,
            }
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        
        # Clean up local checkpoint file to save space
        os.remove(checkpoint_path)
        logger.info("Uploaded checkpoint to wandb and cleaned up local file")

# Training completed
logger.info("Training completed!")
wandb.finish()
