from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
import torch
from src.config import Config
from src.rewards.bioinformatics.scorer import Scorer
from src.rewards.bioinformatics.reward_config import RewardConfig
from src.rewards.bioinformatics.logger import RewardComponentLogger
import datetime
from typing import List
import wandb

cfg = Config()
MODEL_ID = cfg.model
TRAIN_PARQUET = cfg.train_dataset
VAL_PARQUET = cfg.val_dataset

run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"grpo-{run_name}"
save_path = f"/s3/checkpoints/verl-grpo/{run_name}"
SEED = 42

# ---- dataset (keep your extra cols for reward) ----
PROMPT_KEY = "prompt"
KEEP_EXTRA_COLS = ["data_source", "ability", "reward_model", "extra_info"]


def select_prompt_and_extras(ds):
    cols = set(ds.column_names)
    keep = ["prompt"] + [c for c in KEEP_EXTRA_COLS if c in cols]
    return ds.select_columns(keep)

train_ds = load_dataset("parquet", data_files=TRAIN_PARQUET, split="train")
train_ds = select_prompt_and_extras(train_ds)

try:
    eval_ds = load_dataset("parquet", data_files=VAL_PARQUET, split="train")
    eval_ds = select_prompt_and_extras(eval_ds)
    use_eval = True
except Exception:
    eval_ds = None
    use_eval = False


tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
tok.padding_side = "left"

# Remap specials to the correct strings (these already exist in your vocab)
tok.eos_token = "</s>"
tok.bos_token = "<s>"
tok.pad_token = "[PAD]"

# Re-assert
assert tok.eos_token_id == 30001, tok.eos_token_id
assert tok.bos_token_id == 30000, tok.bos_token_id
assert tok.pad_token_id == 3, tok.pad_token_id

# Pass IDs explicitly to the model so nothing “helpfully” changes at runtime
model_init_kwargs = {
    "trust_remote_code": True,
    "eos_token_id": tok.eos_token_id,
    "bos_token_id": tok.bos_token_id,
    "pad_token_id": tok.pad_token_id,
}


# ---- GRPO config (keys verified against TRL docs) ----
args = GRPOConfig(
    model_init_kwargs=model_init_kwargs,
    # transformers-style
    output_dir=save_path,
    num_train_epochs=20,
    learning_rate=3e-6,
    lr_scheduler_type="constant",
    warmup_ratio=0.0,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    max_steps=-1,
    save_strategy="steps",
    save_steps=100,
    logging_strategy="steps",
    logging_steps=1,
    report_to=["wandb"],
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=False,
    max_grad_norm=0.5,
    seed=SEED,
    do_eval=use_eval,
    eval_strategy="steps" if use_eval else "no",
    eval_steps=100 if use_eval else None,

    # model/ref-model handling
    disable_dropout=True,          # stabilizes ref-policy logprobs

    # data & generation
    remove_unused_columns=False,   # keep your extras for reward_fn
    max_prompt_length=1024,
    num_generations=8,            
    max_completion_length=256,
    #min_completion_length=16, #not supported currently
    temperature=0.95,
    top_p=0.90,

    # GRPO specifics
    beta=1e-3,                     # KL in loss → auto-loads ref model
    epsilon=0.2,                   # PPO-style clip (replaces cliprange)
    loss_type="bnpo",              # token-level normalization; avoids length bias
    scale_rewards=True,
    mask_truncated_completions=False,

    # vLLM (colocated serverless flag is not a key here)
    use_vllm=True,
    vllm_gpu_memory_utilization=0.15,
    vllm_mode="colocate"
)

# ---- reward config and scorer ----
reward_config = RewardConfig(
    ori_min=1,
    ori_max=1,
    promoter_min=1,
    promoter_max=5,
    terminator_min=0,
    terminator_max=2,
    marker_min=1,
    marker_max=2,
)

scorer = Scorer(reward_config)

reward_logger = RewardComponentLogger(log_frequency=10)


def batch_reward_fn(samples: List[str], **kwargs) -> List[float]:
    rewards: List[float] = []
    for seq in samples:
        score, components = scorer.score(seq)
        rewards.append(float(score))
        reward_logger.add_components(components, float(score))
    return rewards


# ---- W&B init ----
wandb.init(
    project=cfg.wandb_project,
    entity=cfg.wandb_entity,
    name=run_name,
    config={
        "model": MODEL_ID,
        "reward_config": reward_config.model_dump(),
        "grpo_config": {
            "beta": args.beta,
            "epsilon": args.epsilon,
            "temperature": args.temperature,
            "num_generations": args.num_generations,
            "loss_type": args.loss_type,
        },
    },
)

# ---- trainer (NO ref_model kwarg) ----
trainer = GRPOTrainer(
    model=MODEL_ID,               
    reward_funcs=[batch_reward_fn],
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds if use_eval else None,
    processing_class=tok,
    callbacks=[reward_logger],
)

trainer.train()
trainer.save_model(args.output_dir)
tok.save_pretrained(args.output_dir)
