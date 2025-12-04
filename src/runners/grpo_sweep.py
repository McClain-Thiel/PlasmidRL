"""Optuna-powered GRPO sweep runner (no external YAML/CLI sweep required)."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Lock
from typing import List
import re

import optuna
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from src.config import Config, EvalConfig
from src.eval.eval import Evaluator
from src.rewards.bioinformatics.logger import RewardComponentLogger
from src.rewards.bioinformatics.reward_config import RewardConfig
from src.rewards.bioinformatics.scorer import Scorer
from src.utils.training_utils import EvalCallback, test_checkpoint_directory_write


def load_train_val_datasets(cfg: Config):
    """Load prompt datasets for training and evaluation."""
    def select_prompt_column(ds):
        cols = set(ds.column_names)
        keep_cols = ["prompt"] + [
            c for c in ["data_source", "ability", "reward_model", "extra_info"] if c in cols
        ]
        return ds.select_columns(keep_cols)

    train_ds = load_dataset("parquet", data_files=cfg.train_dataset, split="train")
    val_ds = load_dataset("parquet", data_files=cfg.val_dataset, split="train")
    return select_prompt_column(train_ds), select_prompt_column(val_ds)


def build_checkpoint_path(cfg: Config, run_name: str, trial_number: int) -> str:
    checkpoint_dir = f"/s3/{cfg.checkpoints_path.rstrip('/')}/grpo-sweeps/{run_name}/trial-{trial_number}"
    test_checkpoint_directory_write(checkpoint_dir)
    return checkpoint_dir


def build_grpo_args(cfg: Config, trial_params: dict, checkpoint_dir: str, tok: AutoTokenizer) -> GRPOConfig:
    return GRPOConfig(
        model_init_kwargs={
            "trust_remote_code": True,
            "eos_token_id": tok.eos_token_id,
            "bos_token_id": tok.bos_token_id,
            "pad_token_id": tok.pad_token_id,
        },
        output_dir=checkpoint_dir,
        num_train_epochs=1,
        learning_rate=trial_params["learning_rate"],
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        per_device_train_batch_size=trial_params["per_device_train_batch_size"],
        gradient_accumulation_steps=1,
        max_steps=trial_params["max_steps"],
        max_grad_norm=0.5,
        seed=42,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=cfg.sweep.log_frequency,
        report_to=["wandb"],
        do_eval=True,
        eval_strategy=trial_params["eval_strategy"],
        eval_steps=trial_params["eval_steps"],
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        beta=trial_params["beta"],
        epsilon=trial_params["epsilon"],
        loss_type="bnpo",
        scale_rewards=True,
        mask_truncated_completions=False,
        disable_dropout=True,
        remove_unused_columns=False,
        max_prompt_length=1024,
        num_generations=trial_params["num_generations"],
        max_completion_length=256,
        temperature=trial_params["temperature"],
        top_p=trial_params["top_p"],
        use_vllm=True,
        vllm_gpu_memory_utilization=0.15,
        vllm_mode="colocate",
    )


def build_reward_config(trial_params: dict) -> RewardConfig:
    return RewardConfig(
        punish_mode=True,
        length_reward_mode=trial_params["reward_length_reward_mode"],
        min_length=int(trial_params["reward_min_length"]),
        max_length=int(trial_params["reward_max_length"]),
        ideal_min_length=int(trial_params["reward_ideal_min_length"]),
        ideal_max_length=int(trial_params["reward_ideal_max_length"]),
        length_reward_bonus=float(trial_params["reward_length_reward_bonus"]),
        location_aware=True,
        violation_penalty_factor=1.0,
        ori_min=1,
        ori_max=1,
        ori_weight=float(trial_params["reward_ori_weight"]),
        promoter_min=1,
        promoter_max=5,
        promoter_weight=float(trial_params["reward_promoter_weight"]),
        terminator_min=0,
        terminator_max=2,
        terminator_weight=float(trial_params["reward_terminator_weight"]),
        marker_min=1,
        marker_max=2,
        marker_weight=float(trial_params["reward_marker_weight"]),
        cds_min=1,
        cds_max=5,
        cds_weight=float(trial_params["reward_cds_weight"]),
    )


def run_trial(
    trial: optuna.Trial,
    cfg: Config,
    train_ds,
    eval_ds,
    tok: AutoTokenizer,
    run_name: str,
) -> float:
    trial_params = cfg.sweep.sample_trial(trial)
    checkpoint_dir = build_checkpoint_path(cfg, run_name, trial.number)
    args = build_grpo_args(cfg, trial_params, checkpoint_dir, tok)
    reward_config = build_reward_config(trial_params)

    trial_run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=f"{run_name}-trial-{trial.number}",
        tags=["grpo-sweep"],
        config=trial_params,
        reinit=True,
    )
    trial_run.config.update({"reward_config": reward_config.model_dump()})

    scorer = Scorer(reward_config)
    reward_logger = RewardComponentLogger(log_frequency=cfg.sweep.log_frequency)
    component_lock = Lock()

    eval_config = EvalConfig(
        model_name=cfg.model,
        model_path=cfg.model,
        prompts_path=cfg.val_dataset,
        prompts_column="prompt",
        num_samples_per_prompt=max(1, cfg.sweep.num_generations_choices[0]),
        overlap_merge_threshold=0.8,
        sampling_params=cfg.sweep.sampling_params,
        write_to_wandb=True,
        wandb_project=cfg.wandb_project,
        wandb_run_name=trial_run.name,
    )
    evaluator = Evaluator(eval_config)
    eval_callback = EvalCallback(evaluator)

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

    trainer = GRPOTrainer(
        model=cfg.model,
        reward_funcs=[batch_reward_fn],
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=[reward_logger, eval_callback],
    )
    eval_callback.set_trainer(trainer)

    try:
        trainer.train()
    finally:
        reward_mean = reward_logger.last_total_reward_mean or 0.0
        trial_run.summary[cfg.sweep.objective_metric] = reward_mean
        wandb.finish()
        torch.cuda.empty_cache()

    return reward_mean


def main():
    cfg = Config()
    sweep_cfg = cfg.sweep
    run_name = f"grpo-sweep-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_ds, eval_ds = load_train_val_datasets(cfg)

    tok = AutoTokenizer.from_pretrained(cfg.model, use_fast=True, trust_remote_code=True)
    tok.padding_side = "left"
    tok.eos_token = "</s>"
    tok.bos_token = "<s>"
    tok.pad_token = "[PAD]"
    assert tok.eos_token_id == 30001, f"Expected eos_token_id=30001, got {tok.eos_token_id}"
    assert tok.bos_token_id == 30000, f"Expected bos_token_id=30000, got {tok.bos_token_id}"
    assert tok.pad_token_id == 3, f"Expected pad_token_id=3, got {tok.pad_token_id}"

    study = optuna.create_study(direction=sweep_cfg.direction)
    study.optimize(
        lambda trial: run_trial(trial, cfg, train_ds, eval_ds, tok, run_name),
        n_trials=sweep_cfg.n_trials,
        timeout=sweep_cfg.timeout_minutes * 60,
        show_progress_bar=True,
        n_jobs=cfg.opt_jobs
    )

    best_trial = study.best_trial
    print(f"🧪 Best trial {best_trial.number} → {sweep_cfg.objective_metric}={best_trial.value:.4f}")


if __name__ == "__main__":
    main()

