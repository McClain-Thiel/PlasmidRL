from pydantic import BaseModel, ConfigDict
from vllm import SamplingParams
from typing import Optional

class EvalConfig(BaseModel):
    """
    Configuration for evaluation analysis.
    
    Similar to RewardConfig but focused on detailed annotation extraction
    rather than scoring.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Model configuration
    model_name: str
    model_path: str

    # Prompts configuration
    prompts_path: Optional[str] = None  # Path to CSV/parquet file with prompts
    prompts_column: str = "prompt"  # Column name containing prompts
    num_samples_per_prompt: int = 10  # Number of samples to generate per prompt

    # Annotation configuration (similar to RewardConfig)
    overlap_merge_threshold: float = 0.8  # Overlap merge threshold for annotations

    # Generation configuration
    sampling_params: Optional[SamplingParams] = None

    # Logging configuration
    write_to_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

