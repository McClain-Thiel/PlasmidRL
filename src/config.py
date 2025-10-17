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
    sample_model: str = "McClain/plasmid-rl-grpo"
    
    # Replay buffer configuration
    replay_buffer_size: int = 10_000

    s3_bucket: str = "s3://phd-research-storage-1758274488/"
    runs_path: str = "runs/"
    infered_path: str = "infered/"

    model_config = {
        "env_file": ".env",
        "extra": "ignore"  # Ignore extra environment variables
    }

def get_config() -> Config:
    """Get the configuration instance."""
    return Config()

# For backward compatibility
config = get_config()
