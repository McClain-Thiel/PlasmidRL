from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState
from trl import GRPOConfig, GRPOTrainer
from src.rewards import Score
from src.config import Config
import os
import wandb


class KLAdaptiveCallback(TrainerCallback):
    """Adjust the KL coefficient on the fly to keep divergence near a target."""

    def __init__(
        self,
        target: float = 20.0,
        increase_factor: float = 2.0,
        decrease_factor: float = 0.5,
        beta_min: float = 1e-4,
        beta_max: float = 10.0,
    ) -> None:
        self.target = target
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.beta_min = beta_min
        self.beta_max = beta_max

    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ) -> TrainerControl:
        if not logs or not hasattr(self, "trainer"):
            return control
        kl = None
        for key in ("kl", "train/kl", "eval/kl"):
            if key in logs:
                kl = logs[key]
                break
        if kl is None:
            return control

        beta = getattr(self.trainer, "beta", None)
        if beta is None:
            return control

        if kl > self.target * 1.5:
            beta = min(beta * self.increase_factor, self.beta_max)
        elif kl < self.target / 1.5:
            beta = max(beta * self.decrease_factor, self.beta_min)
        else:
            return control

        setattr(self.trainer, "beta", beta)
        logs.setdefault("adaptive_beta", beta)
        return control

config = Config()

# Initialize Weights & Biases logging
wandb_enabled = bool(config.wandb_project)
if wandb_enabled:
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        config={
            "model": config.model,
            "train_dataset": config.train_dataset,
            "test_dataset": config.test_dataset,
            "output_dir": config.output_dir,
        },
        reinit=True,
    )

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
    learning_rate=1e-6,
    max_steps=1000,
    logging_steps=10,
    save_steps=20,
    warmup_ratio=0.03,
    bf16=True,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    max_prompt_length=256,
    max_completion_length=192,
    max_grad_norm=1.0,
    # Reward shaping & loss style:
    scale_rewards="none",             # use raw rewards (already normalized upstream)
    loss_type="dapo",                 # token-normalized variant helps long CoT
    beta=0.2,                         # stronger KL penalty to keep policy near the reference
    # Generation params:
    temperature=1.2,
    top_p=0.9,
    # Logging
    logging_dir=os.path.join(config.output_dir, "logs"),
    report_to=("wandb" if wandb_enabled else "none"),
    run_name=config.wandb_run_name,
)

trainer = GRPOTrainer(
    model=policy_id,                  # can also pass a loaded model object
    reward_funcs=Score,         # can be a list: [grpo_reward, "rm-model-id", ...]
    args=args,
    train_dataset=train_dataset,            # expects a column "prompt" by default; see docs to customize
    processing_class=tokenizer,       # padding side must be left; pad_token must be set
)

# Stop early if KL spikes unreasonably high (indicative of divergence)
trainer.add_callback(KLAdaptiveCallback(target=50.0))

if __name__ == "__main__":
    try:
        # Log dataset sizes before training
        if wandb_enabled:
            wandb.log({
                "dataset/train_size": len(train_dataset),
                "dataset/val_size": len(val_dataset),
            }, step=0)
        trainer.train()
    finally:
        if wandb_enabled:
            wandb.finish()
