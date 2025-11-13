from typing import Optional
from pydantic import SecretStr, Field, AliasChoices
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    # Model and environment configuration
    informatics_server_url: str = "http://server:8080"
    huggingface_token: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("hf_token", "HF_TOKEN", "HUGGINGFACE_TOKEN"),
    )
    model: str = "McClain/plasmidgpt-addgene-gpt2"
    
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
    sample_model: str = "/mnt/s3/phd-research-storage-1758274488/checkpoints/grpo-production/grpo-production-20251110_132247"
    
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

    model_config = {
        "env_file": ".env",
        "extra": "ignore"  # Ignore extra environment variables
    }

def get_config() -> Config:
    """Get the configuration instance."""
    return Config()

# For backward compatibility
config = get_config()
