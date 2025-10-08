from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from src.rewards import Score
from src.config import Config

config = Config()

dataset_dict = load_dataset(
    "parquet",
    data_files={"train": config.train_dataset, "validation": config.test_dataset},
)
train_dataset = dataset_dict["train"]
val_dataset = dataset_dict["validation"]

policy_id = config.model
tokenizer = AutoTokenizer.from_pretrained(policy_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

args = GRPOConfig(
    output_dir=config.output_dir,
    num_generations=32,                
    generation_batch_size=64,
    per_device_train_batch_size=2,    
    learning_rate=5e-6,
    max_steps=1000,
    logging_steps=10,
    save_steps=20,
    warmup_ratio=0.03,
    bf16=True,
    gradient_accumulation_steps=4,
    max_prompt_length=256,
    max_completion_length=512,
    gradient_checkpointing=False,
    # Reward shaping & loss style:
    scale_rewards="batch",            # robust scaling; try False for Dr.GRPO-style
    loss_type="dapo",                 # token-normalized variant helps long CoT
    beta=0.0,                         # KL off by default; turn on (e.g., 0.02) if you see drift
    # Generation params:
    temperature=1.2,
    top_p=0.9,
)

trainer = GRPOTrainer(
    model=policy_id,                  # can also pass a loaded model object
    reward_funcs=Score,         # can be a list: [grpo_reward, "rm-model-id", ...]
    args=args,
    train_dataset=train_dataset,            # expects a column "prompt" by default; see docs to customize
    processing_class=tokenizer,       # padding side must be left; pad_token must be set
)

if __name__ == "__main__":
    trainer.train()