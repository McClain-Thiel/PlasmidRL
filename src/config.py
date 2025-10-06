from typing import Optional
from pydantic import SecretStr, Field, AliasChoices
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    # Model and environment configuration
    huggingface_token: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("hf_token", "HF_TOKEN", "HUGGINGFACE_TOKEN"),
    )
    model: str = "McClain/plasmidgpt-addgene-gpt2"
    train_dataset: str = "data/train.parquet"
    test_dataset: str = "data/test.parquet"

    output_dir: str = "checkpoints"
    
    # Additional environment variables
    cuda_visible_devices: str = "all"

    #this is the GFP cassette
    default_query: str = "tttacggctagctcagtcctaggtatagtgctagcTACTagagaaagaggagaaatactaAATGatgcgtaaaggagaagaacttttcactggagttgtcccaattcttgttgaattagatggtgatgttaatgggcacaaattttctgtcagtggagagggtgaaggtgatgcaacatacggaaaacttacccttaaatttatttgcactactggaaaactacctgttccatggccaacacttgtcactactttcggttatggtgttcaatgctttgcgagatacccagatcatatgaaacagcatgactttttcaagagtgccatgcccgaaggttatgtacaggaaagaactatatttttcaaagatgacgggaactacaagacacgtgctgaagtcaagtttgaaggtgatacccttgttaatagaatcgagttaaaaggtattgattttaaagaagatggaaacattcttggacacaaattggaatacaactataactcacacaatgtatacatcatggcagacaaacaaaagaatggaatcaaagttaacttcaaaattagacacaacattgaagatggaagcgttcaactagcagaccattatcaacaaaatactccaattggcgatggccctgtccttttaccagacaaccattacctgtccacacaatctgccctttcgaaagatcccaacgaaaagagagatcacatggtccttcttgagtttgtaacagctgttgtttgtcggtgaacgctctctactagagtcacactggctcaccttcgggtgggcctttctgcgtttata"
    
    # Weights & Biases configuration
    wandb_api_key: Optional[SecretStr] = None
    wandb_entity: str = "mcclain"  # Use your personal/team entity instead of organization
    wandb_project: str = "plasmidrl"
    wandb_run_name: Optional[str] = None  # If None, wandb will auto-generate
    wandb_tags: list[str] = ["plasmid", "rl", "grpo"]
    wandb_notes: str = "GRPO training for plasmid design optimization"

    # RL Training configuration
    K: int = 1  # number of samples per prompt to compute MC advantage
    num_iters: int = 200  
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_new_tokens: int = 512
    dialog_turns_per_batch: int = 16
    reward_max_workers: int = 16
    reward_log_timings: bool = False
    log_level: str = "INFO"
    advantage_verbose: bool = False

    # Training logging configuration
    log_interval: int = 2  # How often to print progress
    checkpoint_interval: int = 5  # How often to save checkpoints
    
    # Replay buffer configuration
    replay_buffer_size: int = 10_000

    model_config = {
        "env_file": ".env",
        "extra": "ignore"  # Ignore extra environment variables
    }

def get_config() -> Config:
    """Get the configuration instance."""
    return Config()

# For backward compatibility
config = get_config()
