from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import torch
from src.config import Config
from src.rewards.bioinformatics.scorer import Scorer
from src.rewards.bioinformatics.reward_config import RewardConfig
from src.rewards.bioinformatics.logger import RewardComponentLogger
from src.eval.eval import Evaluator
from src.eval.eval_config import EvalConfig
from src.utils.training_utils import EvalCallback, test_checkpoint_directory_write
from vllm import SamplingParams
import datetime
from typing import List
import wandb
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import re
import os

# Configuration
cfg = Config()
run_name = f"grpo-production-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Dataset loading
def load_train_val_datasets():
    """Load and preprocess training and validation datasets."""
    def select_prompt_column(ds):
        cols = set(ds.column_names)
        keep_cols = ["prompt"] + [c for c in ["data_source", "ability", "reward_model", "extra_info"] if c in cols]
        return ds.select_columns(keep_cols)
    
    train_ds = load_dataset("parquet", data_files=cfg.train_dataset, split="train")
    val_ds = load_dataset("parquet", data_files=cfg.val_dataset, split="train")
    
    return select_prompt_column(train_ds), select_prompt_column(val_ds)

train_ds, eval_ds = load_train_val_datasets()

# Tokenizer setup
tok = AutoTokenizer.from_pretrained(cfg.model, use_fast=True, trust_remote_code=True)
tok.padding_side = "left"
tok.eos_token = "</s>"
tok.bos_token = "<s>"
tok.pad_token = "[PAD]"

# Validate token IDs
assert tok.eos_token_id == 30001, f"Expected eos_token_id=30001, got {tok.eos_token_id}"
assert tok.bos_token_id == 30000, f"Expected bos_token_id=30000, got {tok.bos_token_id}"
assert tok.pad_token_id == 3, f"Expected pad_token_id=3, got {tok.pad_token_id}"

# Model initialization kwargs
model_init_kwargs = {
    "trust_remote_code": True,
    "eos_token_id": tok.eos_token_id,
    "bos_token_id": tok.bos_token_id,
    "pad_token_id": tok.pad_token_id,
}

# Training configuration - use /s3 mount point with prefix path
checkpoint_dir = f"/s3/{cfg.checkpoints_path.rstrip('/')}/grpo-production/{run_name}"

# Test checkpoint directory write access before proceeding
test_checkpoint_directory_write(checkpoint_dir)

args = GRPOConfig(
    model_init_kwargs=model_init_kwargs,
    output_dir=checkpoint_dir,
    
    # Training parameters
    num_train_epochs=20,
    learning_rate=cfg.grpo_learning_rate,
    lr_scheduler_type="constant",
    warmup_ratio=0.0,
    per_device_train_batch_size=cfg.grpo_per_device_train_batch_size,
    gradient_accumulation_steps=1,
    max_steps=-1,
    max_grad_norm=0.5,
    seed=42,
    
    # Logging and checkpointing
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,  # Keep last 5 checkpoints
    logging_strategy="steps",
    logging_steps=1,
    report_to=["wandb"],
    
    # Evaluation
    do_eval=True,
    eval_strategy="steps",
    eval_steps=50,  # Evaluate every 50 steps
    
    # Optimization
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=False,
    
    # GRPO-specific
    beta=cfg.grpo_beta,
    epsilon=cfg.grpo_epsilon,
    loss_type="bnpo",
    scale_rewards=True,
    mask_truncated_completions=False,
    disable_dropout=True,
    
    # Generation parameters
    remove_unused_columns=False,
    max_prompt_length=1024,
    num_generations=cfg.grpo_num_generations,
    max_completion_length=256,
    temperature=cfg.grpo_temperature,
    top_p=cfg.grpo_top_p,
    
    # vLLM configuration
    use_vllm=True,
    vllm_gpu_memory_utilization=0.15,
    vllm_mode="colocate",
)

# Reward configuration - production parameters from sweep
reward_config = RewardConfig(
    punish_mode=True,  # Use punish mode for better constraint learning
    length_reward_mode=True,
    min_length=2000,
    max_length=30000,
    ideal_min_length=7000,
    ideal_max_length=20000,
    length_reward_bonus=0.7085046275614012,
    ori_min=1,
    ori_max=1,
    ori_weight=1.0,
    promoter_min=1,
    promoter_max=5,
    promoter_weight=1.0,
    terminator_min=0,
    terminator_max=2,
    terminator_weight=0.5,
    marker_min=1,
    marker_max=2,
    marker_weight=1.0,
    cds_min=1,
    cds_max=5,
    cds_weight=1.0,
    location_aware=True,
)

# Initialize scorer and logger
scorer = Scorer(reward_config)
reward_logger = RewardComponentLogger(log_frequency=1)
component_lock = Lock()

# Initialize evaluation callback
eval_config = EvalConfig(
    model_name=cfg.model,
    model_path=cfg.model,  # Will be overridden by checkpoint path in callback
    prompts_path=cfg.val_dataset,  # Use test.parquet for evaluation prompts
    prompts_column="prompt",
    num_samples_per_prompt=5,  # Fewer samples for quick testing
    overlap_merge_threshold=0.8,
    sampling_params=SamplingParams(
        max_tokens=256,
        temperature=0.95,
        top_p=0.90,
        top_k=0,
    ),
    write_to_wandb=True,
    wandb_project=cfg.wandb_project,
    wandb_run_name=run_name,
)
evaluator = Evaluator(eval_config)
eval_callback = EvalCallback(evaluator)

# Reward function
def score_single(idx_and_seq):
    """Score a single sequence and log components thread-safely."""
    idx, seq = idx_and_seq
    try:
        score, components = scorer.score(seq)
        with component_lock:
            reward_logger.add_components(components, float(score))
        return float(score), components
    except Exception as e:
        print(f"Warning: Failed to score completion {idx} (len={len(seq)}): {str(e)[:100]}")
        return 0.0, None

def batch_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Compute rewards for a batch of completions.
    
    Args:
        prompts: List of prompt strings
        completions: List of completion strings (without prompts)
        
    Returns:
        List of reward scores
    """
    # Clean sequences: remove non-DNA characters
    cleaned = [re.sub(r'[^ATCG]', '', c.upper().replace(" ", "")) for c in completions]
    
    # Parallelize scoring
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(score_single, enumerate(cleaned)))
    
    return [r[0] for r in results]

# Initialize W&B
wandb_run = wandb.init(
    project=cfg.wandb_project,
    entity=cfg.wandb_entity,
    name=run_name,
    tags=["production", "grpo", "optimized-hyperparams"],
    config={
        "model": cfg.model,
        "reward_config": reward_config.model_dump(),
        "training": {
            "learning_rate": cfg.grpo_learning_rate,
            "batch_size": cfg.grpo_per_device_train_batch_size,
            "num_epochs": args.num_train_epochs,
            "num_generations": cfg.grpo_num_generations,
        },
        "grpo": {
            "beta": cfg.grpo_beta,
            "epsilon": cfg.grpo_epsilon,
            "temperature": cfg.grpo_temperature,
            "top_p": cfg.grpo_top_p,
            "loss_type": args.loss_type,
        },
        "checkpoint_dir": checkpoint_dir,
    },
)

# Print wandb URL and checkpoint info
if wandb_run:
    print(f"\n{'='*80}")
    print(f"🚀 W&B Run URL: {wandb_run.url}")
    print(f"📁 Checkpoint Directory: {checkpoint_dir}")
    print(f"{'='*80}\n")

# Initialize trainer
trainer = GRPOTrainer(
    model=cfg.model,
    reward_funcs=[batch_reward_fn],
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tok,
    callbacks=[reward_logger, eval_callback],
)

# Set trainer reference in callback (for accessing trainer.llm)
eval_callback.set_trainer(trainer)

# Train and save
print(f"Starting training with {args.num_train_epochs} epochs...")
trainer.train()

# Save final model and tokenizer
print(f"Saving final model to {checkpoint_dir}...")
trainer.save_model(checkpoint_dir)
tok.save_pretrained(checkpoint_dir)

# Log final checkpoint as W&B artifact
artifact = wandb.Artifact(
    name=f"model-{run_name}",
    type="model",
    description=f"Final GRPO model checkpoint from production run with optimized hyperparameters",
)
artifact.add_dir(checkpoint_dir)
wandb_run.log_artifact(artifact)

print(f"✓ Training complete! Model saved to {checkpoint_dir}")
print(f"✓ Model artifact logged to W&B: {wandb_run.url}")

# Finish run
wandb.finish()
