from pydantic import BaseSettings, SecretStr

class Config(BaseSettings):
    # Model and environment configuration
    informatics_server_url: str = "http://server:8080"
    huggingface_token: SecretStr 
    model: str = "McClain/plasmidgpt-addgene-gpt2"

    #this is the GFP cassette
    default_query: str = "tttacggctagctcagtcctaggtatagtgctagcTACTagagaaagaggagaaatactaAATGatgcgtaaaggagaagaacttttcactggagttgtcccaattcttgttgaattagatggtgatgttaatgggcacaaattttctgtcagtggagagggtgaaggtgatgcaacatacggaaaacttacccttaaatttatttgcactactggaaaactacctgttccatggccaacacttgtcactactttcggttatggtgttcaatgctttgcgagatacccagatcatatgaaacagcatgactttttcaagagtgccatgcccgaaggttatgtacaggaaagaactatatttttcaaagatgacgggaactacaagacacgtgctgaagtcaagtttgaaggtgatacccttgttaatagaatcgagttaaaaggtattgattttaaagaagatggaaacattcttggacacaaattggaatacaactataactcacacaatgtatacatcatggcagacaaacaaaagaatggaatcaaagttaacttcaaaattagacacaacattgaagatggaagcgttcaactagcagaccattatcaacaaaatactccaattggcgatggccctgtccttttaccagacaaccattacctgtccacacaatctgccctttcgaaagatcccaacgaaaagagagatcacatggtccttcttgagtttgtaacagctgttgtttgtcggtgaacgctctctactagagtcacactggctcaccttcgggtgggcctttctgcgtttata"
    
    # Weights & Biases configuration
    wandb_api_key: SecretStr
    wandb_entity: str = "ucl-cssb-org"
    wandb_project: str = "plasmidrl"
    wandb_run_name: str = None  # If None, wandb will auto-generate
    wandb_tags: list[str] = ["plasmid", "rl", "grpo"]
    wandb_notes: str = "GRPO training for plasmid design optimization"

    # RL Training configuration
    K: int = 4  # number of samples per prompt to compute MC advantage
    num_iters: int = 200  # Fixed typo from "itterations"
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    
    # Training logging configuration
    log_interval: int = 20  # How often to print progress
    checkpoint_interval: int = 50  # How often to save checkpoints
    
    # Replay buffer configuration
    replay_buffer_size: int = 10_000

    class Config:
        env_file = ".env"

def get_config() -> Config:
    """Get the configuration instance."""
    return Config()

# For backward compatibility
config = get_config()