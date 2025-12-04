from typing import Optional, Tuple
import optuna
from pydantic import BaseModel, ConfigDict, SecretStr, Field, AliasChoices
from pydantic_settings import BaseSettings
from vllm import SamplingParams


class SweepConfig(BaseModel):
    """Defines hyperparameter ranges for programmatic GRPO sweeps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    direction: str = "maximize"
    objective_metric: str = "reward_components/total_reward/mean"
    n_trials: int = 60
    timeout_minutes: int = 60

    training_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 50
    log_frequency: int = 5

    learning_rate_range: Tuple[float, float] = (1e-6, 1e-4)
    per_device_train_batch_size_choices: Tuple[int, ...] = (8, 16, 32)
    num_generations_choices: Tuple[int, ...] = (4, 8, 16)
    temperature_range: Tuple[float, float] = (0.7, 1.3)
    top_p_range: Tuple[float, float] = (0.85, 0.95)
    beta_range: Tuple[float, float] = (1e-4, 1e-2)
    epsilon_range: Tuple[float, float] = (0.1, 0.3)

    reward_ori_weight_choices: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    reward_promoter_weight_choices: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    reward_terminator_weight_choices: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    reward_marker_weight_choices: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    reward_cds_weight_choices: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    reward_length_reward_mode_options: Tuple[bool, ...] = (False, True)
    reward_length_configs: Tuple[Tuple[int, int, int, int, float], ...] = (
        (2000, 30000, 3000, 12000, 0.7),
        (5000, 30000, 7000, 20000, 0.5),
    )

    sampling_params: SamplingParams = SamplingParams(
        max_tokens=256,
        temperature=0.95,
        top_p=0.9,
        top_k=0,
    )

    def sample_trial(self, trial: "optuna.Trial") -> dict[str, float | bool | int]:
        """Sample hyperparameters for a single trial."""
        params: dict[str, float | int | bool] = {
            "learning_rate": trial.suggest_float(
                "learning_rate", self.learning_rate_range[0], self.learning_rate_range[1], log=True
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", self.per_device_train_batch_size_choices
            ),
            "num_generations": trial.suggest_categorical("num_generations", self.num_generations_choices),
            "temperature": trial.suggest_float(
                "temperature", self.temperature_range[0], self.temperature_range[1]
            ),
            "top_p": trial.suggest_float("top_p", self.top_p_range[0], self.top_p_range[1]),
            "beta": trial.suggest_float("beta", self.beta_range[0], self.beta_range[1], log=True),
            "epsilon": trial.suggest_float("epsilon", self.epsilon_range[0], self.epsilon_range[1]),
            "reward_ori_weight": trial.suggest_categorical(
                "reward_ori_weight", self.reward_ori_weight_choices
            ),
            "reward_promoter_weight": trial.suggest_categorical(
                "reward_promoter_weight", self.reward_promoter_weight_choices
            ),
            "reward_terminator_weight": trial.suggest_categorical(
                "reward_terminator_weight", self.reward_terminator_weight_choices
            ),
            "reward_marker_weight": trial.suggest_categorical(
                "reward_marker_weight", self.reward_marker_weight_choices
            ),
            "reward_cds_weight": trial.suggest_categorical(
                "reward_cds_weight", self.reward_cds_weight_choices
            ),
            "reward_length_reward_mode": trial.suggest_categorical(
                "reward_length_reward_mode", self.reward_length_reward_mode_options
            ),
        }
        length_config = trial.suggest_categorical("reward_length_config", self.reward_length_configs)
        params.update(
            {
                "reward_min_length": length_config[0],
                "reward_max_length": length_config[1],
                "reward_ideal_min_length": length_config[2],
                "reward_ideal_max_length": length_config[3],
                "reward_length_reward_bonus": length_config[4],
            }
        )
        params["max_steps"] = self.training_steps
        params["eval_strategy"] = self.eval_strategy
        params["eval_steps"] = self.eval_steps
        return params


class Config(BaseSettings):
    # Model and environment configuration
    informatics_server_url: str = "http://server:8080"
    huggingface_token: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("hf_token", "HF_TOKEN", "HUGGINGFACE_TOKEN"),
    )
    model: str = "UCL-CSSB/PlasmidGPT-SFT"#"McClain/plasmidgpt-addgene-gpt2"
    
    # Additional environment variables
    cuda_visible_devices: str = "all"

    #huggingface configuration
    huggingface_token: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("hf_token", "HF_TOKEN", "HUGGINGFACE_TOKEN"),
    )

    train_dataset: str = "data/train.parquet"
    val_dataset: str = "data/test.parquet"


    #this is the GFP cassette
    default_query: str = "tttacggctagctcagtcctaggtatagtgctagcTACTagagaaagaggagaaatactaAATGatgcgtaaaggagaagaacttttcactggagttgtcccaattcttgttgaattagatggtgatgttaatgggcacaaattttctgtcagtggagagggtgaaggtgatgcaacatacggaaaacttacccttaaatttatttgcactactggaaaactacctgttccatggccaacacttgtcactactttcggttatggtgttcaatgctttgcgagatacccagatcatatgaaacagcatgactttttcaagagtgccatgcccgaaggttatgtacaggaaagaactatatttttcaaagatgacgggaactacaagacacgtgctgaagtcaagtttgaaggtgatacccttgttaatagaatcgagttaaaaggtattgattttaaagaagatggaaacattcttggacacaaattggaatacaactataactcacacaatgtatacatcatggcagacaaacaaaagaatggaatcaaagttaacttcaaaattagacacaacattgaagatggaagcgttcaactagcagaccattatcaacaaaatactccaattggcgatggccctgtccttttaccagacaaccattacctgtccacacaatctgccctttcgaaagatcccaacgaaaagagagatcacatggtccttcttgagtttgtaacagctgttgtttgtcggtgaacgctctctactagagtcacactggctcaccttcgggtgggcctttctgcgtttata".upper()
    
    # Weights & Biases configuration
    wandb_api_key: Optional[SecretStr] = None
    wandb_entity: str = "ucl-cssb" 
    wandb_project: str = "PlasmidRL"


    # Training logging configuration
    log_interval: int = 2  # How often to print progress
    checkpoint_interval: int = 5  # How often to save checkpoints

    #sample generation configuration
    sample_model: str = "UCL-CSSB/PlasmidGPT-SFT"
    
    # Replay buffer configuration
    replay_buffer_size: int = 10_000

    s3_bucket: str = "s3://phd-research-storage-1758274488/"
    region_name: str = "us-east-1"
    runs_path: str = "runs/"
    infered_path: str = "infered/"
    checkpoints_path: str = "checkpoints/"  # S3 prefix for checkpoint storage

    # Production GRPO hyperparameters (from sweep optimization)
    grpo_learning_rate: float = 0.00001906419115928539
    grpo_per_device_train_batch_size: int = 16
    grpo_num_generations: int = 4
    grpo_temperature: float = 1.2292317925218237
    grpo_top_p: float = 0.9086524230707756
    grpo_beta: float = 0.00088482365318492
    grpo_epsilon: float = 0.2649093053949679

    sweep: SweepConfig = SweepConfig()

    opt_jobs: int = 1

    model_config = {
        "env_file": ".env",
        "extra": "ignore"  # Ignore extra environment variables
    }


class EvalConfig(BaseModel):
    """
    Evaluation configuration that used to live in eval/eval_config.py.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    model_path: str

    prompts_path: Optional[str] = None
    prompts_column: str = "prompt"
    num_samples_per_prompt: int = 10

    overlap_merge_threshold: float = 0.8

    sampling_params: Optional[SamplingParams] = None

    write_to_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    self_bleu_prompt: Optional[str] = "ATG"
    self_bleu_sample_count: int = 10
    self_bleu_max_n: int = 4


def get_config() -> Config:
    """Get the configuration instance."""
    return Config()


# For backward compatibility
config = get_config()
