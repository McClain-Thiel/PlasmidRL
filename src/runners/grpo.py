from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from trl import GRPOTrainer, GRPOConfig
import torch
from src.config import Config
from src.rewards import Score
import datetime

MODEL_ID = "McClain/plasmidgpt-addgene-gpt2"
TRAIN_PARQUET = "data/train.parquet"
VAL_PARQUET = "data/test.parquet"

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

# ---- tokenizer ----
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"



# ---- GRPO config (keys verified against TRL docs) ----
args = GRPOConfig(
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
    save_steps=10,
    logging_strategy="steps",
    logging_steps=1,
    bf16=torch.cuda.is_available(),
    gradient_checkpointing=False,
    max_grad_norm=0.5,
    seed=SEED,
    do_eval=use_eval,
    eval_strategy="steps" if use_eval else "no",
    eval_steps=10 if use_eval else None,

    # model/ref-model handling
    model_init_kwargs={"trust_remote_code": True},  # used if model is passed as string
    disable_dropout=True,          # stabilizes ref-policy logprobs

    # data & generation
    remove_unused_columns=False,   # keep your extras for reward_fn
    max_prompt_length=1024,
    num_generations=8,            
    max_completion_length=256,     
    temperature=0.80,
    top_p=0.90,

    # GRPO specifics
    beta=1e-3,                     # KL in loss → auto-loads ref model
    epsilon=0.2,                   # PPO-style clip (replaces cliprange)
    loss_type="bnpo",              # token-level normalization; avoids length bias
    scale_rewards=True,
    mask_truncated_completions=True,

    # vLLM (colocated serverless flag is not a key here)
    use_vllm=True,
    vllm_gpu_memory_utilization=0.15,
    vllm_mode="colocate"
)

# ---- trainer (NO ref_model kwarg) ----
trainer = GRPOTrainer(
    model=MODEL_ID,               
    reward_funcs=Score,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds if use_eval else None,
    processing_class=tok,
)

trainer.train()
trainer.save_model(args.output_dir)
tok.save_pretrained(args.output_dir)
