from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import torch
from src.config import Config
from src.rewards.bioinformatics.scorer import Scorer
from src.rewards.bioinformatics.reward_config import RewardConfig
from src.rewards.bioinformatics.logger import RewardComponentLogger
import datetime
from typing import List
import wandb
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import re

# Configuration
cfg = Config()
run_name = f"grpo-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

# Training configuration
args = GRPOConfig(
    model_init_kwargs=model_init_kwargs,
    output_dir=f"/s3/checkpoints/verl-grpo/{run_name}",
    
    # Training parameters
    num_train_epochs=20,
    learning_rate=3e-6,
    lr_scheduler_type="constant",
    warmup_ratio=0.0,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    max_steps=-1,
    max_grad_norm=0.5,
    seed=42,
    
    # Logging and checkpointing
    save_strategy="steps",
    save_steps=100,
    logging_strategy="steps",
    logging_steps=1,
    report_to=["wandb"],
    
    # Evaluation
    do_eval=True,
    eval_strategy="steps",
    eval_steps=100,
    
    # Optimization
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=False,
    
    # GRPO-specific
    beta=1e-3,
    epsilon=0.2,
    loss_type="bnpo",
    scale_rewards=True,
    mask_truncated_completions=False,
    disable_dropout=True,
    
    # Generation parameters
    remove_unused_columns=False,
    max_prompt_length=1024,
    num_generations=8,
    max_completion_length=256,
    temperature=0.95,
    top_p=0.90,
    
    # vLLM configuration
    use_vllm=True,
    vllm_gpu_memory_utilization=0.15,
    vllm_mode="colocate",
)

# Reward configuration
reward_config = RewardConfig(
    punish_mode=False,
    length_penalty=True,
    min_length=1000,
    max_length=30000,
    ori_min=1,
    ori_max=1,
    promoter_min=1,
    promoter_max=5,
    terminator_min=0,
    terminator_max=2,
    marker_min=1,
    marker_max=2,
)

# Initialize scorer and logger
scorer = Scorer(reward_config)
reward_logger = RewardComponentLogger(log_frequency=1)
component_lock = Lock()

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
wandb.init(
    project=cfg.wandb_project,
    entity=cfg.wandb_entity,
    name=run_name,
    config={
        "model": cfg.model,
        "reward_config": reward_config.model_dump(),
        "training": {
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "num_epochs": args.num_train_epochs,
        },
        "grpo": {
            "beta": args.beta,
            "epsilon": args.epsilon,
            "temperature": args.temperature,
            "num_generations": args.num_generations,
            "loss_type": args.loss_type,
        },
    },
)

# Initialize trainer
trainer = GRPOTrainer(
    model=cfg.model,
    reward_funcs=[batch_reward_fn],
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tok,
    callbacks=[reward_logger],
)

# Train and save
trainer.train()
trainer.save_model(args.output_dir)
tok.save_pretrained(args.output_dir)
