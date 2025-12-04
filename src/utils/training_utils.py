from dataclasses import dataclass
from transformers import TrainerCallback
from typing import Any, Dict, Protocol, Optional
import wandb
import pandas as pd
import os
import sys
import datetime
from abc import ABC, abstractmethod


def test_checkpoint_directory_write(checkpoint_dir: str) -> None:
    """
    Test that checkpoint directory exists and is writable.
    Raises an error and exits if write test fails.
    
    Args:
        checkpoint_dir: Path to checkpoint directory to test
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Test write access
        test_file = os.path.join(checkpoint_dir, ".write_test")
        test_content = f"Write test at {datetime.datetime.now().isoformat()}\n"
        
        # Write test file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Read back to verify
        with open(test_file, 'r') as f:
            read_content = f.read()
        
        if read_content != test_content:
            raise IOError(f"Write test failed: content mismatch")
        
        # Clean up test file
        os.remove(test_file)
        
        print(f"✓ Checkpoint directory write test passed: {checkpoint_dir}")
        
    except Exception as e:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ CHECKPOINT DIRECTORY WRITE TEST FAILED\n"
            f"{'='*80}\n"
            f"Directory: {checkpoint_dir}\n"
            f"Error: {str(e)}\n"
            f"\nThis usually means:\n"
            f"  - S3 mount is not available at /s3\n"
            f"  - Insufficient permissions to write to S3\n"
            f"  - Disk space is full\n"
            f"\nPlease check:\n"
            f"  1. Docker volume mount: /mnt/s3/phd-research-storage-1758274488:/s3:rw\n"
            f"  2. S3 mount is accessible: ls -la /s3\n"
            f"  3. Write permissions on S3 mount\n"
            f"{'='*80}\n"
        )
        print(error_msg, file=sys.stderr)
        sys.exit(1)


@dataclass
class EvaluationResult:
    """Container for evaluation table and additional summary metrics."""

    dataframe: pd.DataFrame
    metrics: Dict[str, float]


class EvalRunner(ABC):
    """
    Protocol for evaluation runner classes.
    
    Classes implementing this protocol should have a `run_with_trainer()` method that:
    - Takes trainer and wandb_run as parameters
    - Uses the trainer's model directly (already loaded on GPU)
    - Returns a pandas DataFrame containing evaluation results
    - Each row should contain evaluation metrics (e.g., score, length, etc.)
    """
    
    @abstractmethod
    def run_with_trainer(self, trainer: Any, wandb_run: Optional[Any] = None) -> EvaluationResult:
        """
        Run evaluation using the trainer's model directly.
        
        Args:
            trainer: Trainer instance with model already loaded
            wandb_run: Optional wandb run object for logging
            
        Returns:
            EvaluationResult containing a DataFrame with evaluation metrics and
            optional summary metrics that should also be logged (e.g., self-BLEU).
        """
        ...


class EvalCallback(TrainerCallback):
    """
    Minimal training callback that runs evaluation during training and logs results to wandb.
    
    Uses the trainer's in-memory model directly (no need to reload from disk).
    Delegates all evaluation logic to the provided evaluator class.
    """
    
    def __init__(self, evaluator: EvalRunner):
        """
        Initialize the evaluation callback.
        
        Args:
            evaluator: An object with a `run_with_trainer(trainer, wandb_run)` method that
                      performs evaluation using the trainer's model and returns a DataFrame
        """
        self.evaluator = evaluator
        self.last_eval_step = -1
        self._trainer_ref = None  # Will be set when trainer is available
    
    def set_trainer(self, trainer: Any):
        """Set trainer reference for callbacks that need it."""
        self._trainer_ref = trainer
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Run evaluation when trainer evaluation is triggered."""
        print(f"[EvalCallback] on_evaluate called at step {state.global_step}")
        
        # Avoid duplicate evals at the same step
        if state.global_step == self.last_eval_step:
            print(f"[EvalCallback] Already evaluated at step {state.global_step}, skipping")
            return
        
        self.last_eval_step = state.global_step
        
        try:
            # Get trainer from kwargs - transformers TrainerCallback passes 'model' and sometimes 'trainer'
            # For GRPOTrainer, we need to access it via the callback's parent reference or kwargs
            trainer = kwargs.get('trainer')
            if trainer is None:
                # Try to get model directly - GRPOTrainer might pass model in kwargs
                model = kwargs.get('model')
                if model is not None:
                    # If we have model but not trainer, we need to find trainer another way
                    # Actually, let's check if we can store a reference to trainer in __init__
                    print("[EvalCallback] Warning: Trainer not found in kwargs, checking callback context")
                    # Fallback: use self if callback was passed trainer reference
                    if hasattr(self, '_trainer_ref'):
                        trainer = self._trainer_ref
                    else:
                        print("[EvalCallback] Warning: Cannot access trainer, skipping evaluation")
                        return
            
            print(f"[EvalCallback] Running evaluation at step {state.global_step} (using model from trainer)")
            
            # Get wandb run object and URL
            wandb_run = wandb.run
            if wandb_run:
                wandb_url = wandb_run.url
                print(f"[EvalCallback] W&B Run URL: {wandb_url}")
            
            # Run evaluation using the trainer's model directly
            evaluation_result = self.evaluator.run_with_trainer(trainer, wandb_run)
            results_df = evaluation_result.dataframe if evaluation_result else pd.DataFrame()
            extra_metrics = evaluation_result.metrics if evaluation_result else {}
            
            if (results_df is None or len(results_df) == 0) and not extra_metrics:
                print("[EvalCallback] Warning: Evaluation returned no results")
                return
            
            # Log results to wandb
            self._log_results(results_df, state.global_step, extra_metrics)
            
            print(f"[EvalCallback] Logged evaluation results for step {state.global_step}")
            
        except Exception as e:
            print(f"[EvalCallback] Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_checkpoint_path(self, args, state) -> Optional[str]:
        """
        Get the path to the current checkpoint.
        
        Args:
            args: Training arguments
            state: Trainer state
            
        Returns:
            Path to checkpoint directory, or base model path if checkpoint doesn't exist yet
        """
        # Check if there's a checkpoint directory for this step
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        if os.path.exists(checkpoint_dir) and os.path.exists(f"{checkpoint_dir}/config.json"):
            return checkpoint_dir
        
        # Fallback: use output_dir if checkpoint doesn't exist yet
        # Check if output_dir has config.json (meaning it's a valid checkpoint)
        if os.path.exists(args.output_dir) and os.path.exists(f"{args.output_dir}/config.json"):
            return args.output_dir
        
        # If no checkpoint exists yet, return None - evaluator will use base model
        return None
    
    def _log_results(
        self,
        results_df: Optional[pd.DataFrame],
        step: int,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log evaluation results to wandb as both table and artifact.
        
        Args:
            results_df: DataFrame containing evaluation results (optional when logging only extra metrics)
            step: Current training step
            extra_metrics: Additional scalar metrics to log alongside the table
        """
        has_df = results_df is not None and len(results_df) > 0
        if not has_df and not extra_metrics:
            return

        payload = {"eval/step": step}
        if has_df:
            payload["eval/results_table"] = wandb.Table(dataframe=results_df)

        if extra_metrics:
            payload.update(extra_metrics)

        wandb.log(payload)
        
        if not has_df:
            return

        df = results_df

        # Log summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        stats = {}
        for col in numeric_cols:
            if col != "step":
                stats[f"eval/stats/{col}/mean"] = float(df[col].mean())
                stats[f"eval/stats/{col}/std"] = float(df[col].std())
                stats[f"eval/stats/{col}/min"] = float(df[col].min())
                stats[f"eval/stats/{col}/max"] = float(df[col].max())
                stats[f"eval/stats/{col}/median"] = float(df[col].median())
        
        if stats:
            wandb.log(stats)
        
        # Create and log artifact (for full data export)
        artifact_name = f"eval_results_step_{step}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="evaluation_results",
            description=f"Evaluation results at training step {step}",
            metadata={
                "step": step,
                "total_samples": len(df),
            }
        )
        
        # Add CSV to artifact
        with artifact.new_file("results.csv", mode="w") as f:
            df.to_csv(f, index=False)
        
        # Log artifact
        wandb.log_artifact(artifact)

