import os
import math
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from datetime import datetime

from torchrl.modules.llm import TransformersWrapper
from torchrl.collectors.llm import LLMCollector
from torchrl.objectives.llm import GRPOLoss, MCAdvantage
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import ListStorage
from torchrl.envs.llm import ChatEnv

from src.rewards.plasmid_informatics import RewardTransform
from src.utils.misc_transforms import DefaultQueryOnReset
from src.config import get_config


# ----------------------
# Config & device
# ----------------------
config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Initialize Weights & Biases
# ----------------------
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
    token=getattr(config, "huggingface_token", None),
)

# causal decoders often need a pad token and left padding for chat RL
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if getattr(tokenizer, "padding_side", None) != "left":
    tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    config.model,
    token=getattr(config, "huggingface_token", None),
)
model.to(device)  # put underlying HF weights on device

# ----------------------
# TorchRL policy wrapper
# ----------------------
policy = TransformersWrapper(
    model=model,
    tokenizer=tokenizer,
    input_mode="text",
    return_log_probs=True,
).eval()  # keep eval() for GRPO rollouts/updates

# If you want to be extra sure buffers live on device:
policy.to(device)

# ----------------------
# Env + reward/initial query transforms
# ----------------------
env = ChatEnv(
    input_mode="text",
    batch_size=(1,),
    tokenizer=tokenizer,
)
env = env.append_transform(RewardTransform("http://server:8080"))
env = env.append_transform(DefaultQueryOnReset([config.default_query]))

# ----------------------
# Collector
# ----------------------
collector = LLMCollector(
    policy=policy,
    env=env,
    dialog_turns_per_batch=1,
)

# ----------------------
# Replay buffer + MCAdvantage
# ----------------------
K = config.K  # number of samples per prompt to compute MC advantage

rb = ReplayBuffer(storage=ListStorage(max_size=config.replay_buffer_size))
rb = rb.append_transform(
    MCAdvantage(
        grpo_size=K,
        # Keys here must match what your env/collector/policy emit.
        # LLMCollector + TransformersWrapper commonly expose prompt under ("text","prompt").
        prompt_key=("text", "prompt"),
        rewards_key="reward",
        done_key="done",
        advantage_key="advantage",
    )
)

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
    
    # The collector yields one tensordict per dialog turn with dialog_turns_per_batch=1
    for step_idx, td in enumerate(collector):
        # --- Make reward shape broadcastable: (*B, 1, 1) ---
        if "reward" in td.keys():
            r = td.get("reward")
            if r.ndim == len(td.batch_size):          # e.g., (B,)
                r = r.unsqueeze(-1).unsqueeze(-1)     # -> (B,1,1)
            elif r.ndim == len(td.batch_size) + 1:    # e.g., (B,1)
                r = r.unsqueeze(-1)                   # -> (B,1,1)
            td.set("reward", r)

        # --- Push to RB: MCAdvantage will fill "advantage" once it has K trajs for the same prompt ---
        rb.add(td)

        # Not every iter will have advantage ready; skip until MCAdvantage emits it
        if "advantage" not in td.keys():
            if step_idx % config.log_interval == 0:
                print(f"[iter {outer} step {step_idx}] waiting for groups to fill (K={K})…")
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
        
        iter_rewards.append(avg_r)
        iter_losses.append(loss_val)
        iter_steps += 1
        
        # Calculate global step for wandb
        global_step = outer * 1000 + step_idx  # Approximate global step
        
        # --- Enhanced logging ---
        if step_idx % config.log_interval == 0:
            print(f"[iter {outer} step {step_idx}] loss={loss_val:.4f}  avg_reward={avg_r:.3f}")
            
            # Log individual step metrics to wandb
            log_dict = {
                "step/loss": loss_val,
                "step/reward": avg_r,
                "step/global_step": global_step,
                "step/iteration": outer,
                "step/step_idx": step_idx,
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
        
        print(f"[ITER {outer} SUMMARY] steps={iter_steps}, avg_loss={iter_avg_loss:.4f}, "
              f"avg_reward={iter_avg_reward:.3f}, reward_range=[{iter_min_reward:.3f}, {iter_max_reward:.3f}]")
        
        # Log iteration summary to wandb
        wandb.log({
            "iteration/avg_loss": iter_avg_loss,
            "iteration/avg_reward": iter_avg_reward,
            "iteration/max_reward": iter_max_reward,
            "iteration/min_reward": iter_min_reward,
            "iteration/num_steps": iter_steps,
            "iteration/iteration": outer,
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
        print(f"Saved checkpoint: {checkpoint_path}")
        
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
        print(f"Uploaded checkpoint to wandb and cleaned up local file")

# Training completed
print("Training completed!")
wandb.finish()
