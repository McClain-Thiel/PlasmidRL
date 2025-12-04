"""
W&B Sweep runner for GRPO training.
This script is called by wandb agent with sweep parameters injected via wandb.config.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import torch
from src.config import Config, EvalConfig
from src.rewards.bioinformatics.scorer import Scorer
from src.rewards.bioinformatics.reward_config import RewardConfig
from src.rewards.bioinformatics.logger import RewardComponentLogger
from src.eval.eval import Evaluator
from src.utils.training_utils import EvalCallback, test_checkpoint_directory_write
from vllm import SamplingParams
import datetime
from typing import List
import wandb
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import re
import os


def load_train_val_datasets(cfg):
    """Load and preprocess training and validation datasets."""
    def select_prompt_column(ds):
        cols = set(ds.column_names)
        keep_cols = ["prompt"] + [c for c in ["data_source", "ability", "reward_model", "extra_info"] if c in cols]
        return ds.select_columns(keep_cols)
    
    train_ds = load_dataset("parquet", data_files=cfg.train_dataset, split="train")
    val_ds = load_dataset("parquet", data_files=cfg.val_dataset, split="train")
    
    return select_prompt_column(train_ds), select_prompt_column(val_ds)


def main():
    """Run GRPO training with W&B sweep parameters."""
    # Get base config first
    cfg = Config()
    
    # Initialize W&B run with a meaningful name
    run_name = f"grpo-sweep-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(
        name=run_name,
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
    )
    
    # Get sweep params
    sweep_config = wandb.config
    
    # Dataset loading
    train_ds, eval_ds = load_train_val_datasets(cfg)
    
    # Tokenizer setup
    tok = AutoTokenizer.from_pretrained(cfg.model, use_fast=True, trust_remote_code=True)
    tok.padding_side = "left"
    tok.eos_token = "</s>"
    tok.bos_token = "<s>"
    tok.pad_token = "[PAD]"
    
    assert tok.eos_token_id == 30001, f"Expected eos_token_id=30001, got {tok.eos_token_id}"
    assert tok.bos_token_id == 30000, f"Expected bos_token_id=30000, got {tok.bos_token_id}"
    assert tok.pad_token_id == 3, f"Expected pad_token_id=3, got {tok.pad_token_id}"
    
    model_init_kwargs = {
        "trust_remote_code": True,
        "eos_token_id": tok.eos_token_id,
        "bos_token_id": tok.bos_token_id,
        "pad_token_id": tok.pad_token_id,
    }
    
    # Training configuration - use /s3 mount point with prefix path
    checkpoint_dir = f"/s3/{cfg.checkpoints_path.rstrip('/')}/grpo-sweeps/{run_name}"
    
    # Test checkpoint directory write access before proceeding
    test_checkpoint_directory_write(checkpoint_dir)
    
    args = GRPOConfig(
        model_init_kwargs=model_init_kwargs,
        output_dir=checkpoint_dir,
        
        # Training parameters (use sweep config or defaults)
        num_train_epochs=1,  # Short for sweeps
        learning_rate=sweep_config.get("learning_rate", 3e-6),
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        per_device_train_batch_size=sweep_config.get("per_device_train_batch_size", 16),
        gradient_accumulation_steps=1,
        max_steps=sweep_config.get("max_steps", 100),
        max_grad_norm=0.5,
        seed=42,
        
        # Logging and checkpointing
        save_strategy="steps",
        save_steps=100,  # Save every 100 steps
        save_total_limit=2,  # Keep only the last 2 checkpoints
        logging_strategy="steps",
        logging_steps=5,
        report_to=["wandb"],
        
        # Evaluation
        do_eval=True,
        eval_strategy="steps",
        eval_steps=50,  # More frequent eval for sweeps to track progress
        
        # Optimization
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        
        # GRPO-specific (use sweep config or defaults)
        beta=sweep_config.get("beta", 1e-3),
        epsilon=sweep_config.get("epsilon", 0.2),
        loss_type="bnpo",
        scale_rewards=True,
        mask_truncated_completions=False,
        disable_dropout=True,
        
        # Generation parameters (use sweep config or defaults)
        remove_unused_columns=False,
        max_prompt_length=1024,
        num_generations=sweep_config.get("num_generations", 8),
        max_completion_length=256,
        temperature=sweep_config.get("temperature", 0.95),
        top_p=sweep_config.get("top_p", 0.90),
        
        # vLLM configuration
        use_vllm=True,
        vllm_gpu_memory_utilization=0.15,
        vllm_mode="colocate",
    )
    
    # Reward configuration with sweep parameters
    reward_config = RewardConfig(
        punish_mode=sweep_config.get("reward_punish_mode", False),
        length_reward_mode=sweep_config.get("reward_length_reward_mode", False),
        min_length=sweep_config.get("reward_min_length", 1000),
        max_length=sweep_config.get("reward_max_length", 30000),
        ideal_min_length=sweep_config.get("reward_ideal_min_length", None),
        ideal_max_length=sweep_config.get("reward_ideal_max_length", None),
        length_reward_bonus=sweep_config.get("reward_length_reward_bonus", 0.5),
        ori_min=1,
        ori_max=1,
        ori_weight=sweep_config.get("reward_ori_weight", 1.0),
        promoter_min=1,
        promoter_max=sweep_config.get("reward_promoter_max", 5),
        promoter_weight=sweep_config.get("reward_promoter_weight", 1.0),
        terminator_min=0,
        terminator_max=sweep_config.get("reward_terminator_max", 2),
        terminator_weight=sweep_config.get("reward_terminator_weight", 0.5),
        marker_min=1,
        marker_max=sweep_config.get("reward_marker_max", 2),
        marker_weight=sweep_config.get("reward_marker_weight", 1.0),
        cds_min=1,
        cds_max=sweep_config.get("reward_cds_max", 5),
        cds_weight=sweep_config.get("reward_cds_weight", 1.0),
        location_aware=sweep_config.get("reward_location_aware", True),
    )
    
    # Log reward_config to wandb
    wandb.config.update({"reward_config": reward_config.model_dump()})
    
    # Initialize scorer and logger
    scorer = Scorer(reward_config)
    reward_logger = RewardComponentLogger(log_frequency=5)  # Log frequently for short runs
    component_lock = Lock()
    
    # Initialize evaluation callback
    eval_config = EvalConfig(
        model_name=cfg.model,
        model_path=cfg.model,
        prompts_path=cfg.val_dataset,  # Use test.parquet for evaluation prompts
        prompts_column="prompt",
        num_samples_per_prompt=5,  # Fewer samples for sweeps
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
        cleaned = [re.sub(r'[^ATCG]', '', c.upper().replace(" ", "")) for c in completions]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(score_single, enumerate(cleaned)))
        return [r[0] for r in results]
    
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
    
    # Train
    trainer.train()
    
    # Save best checkpoint info to W&B
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    
    # Finish run
    wandb.finish()


if __name__ == "__main__":
    main()

