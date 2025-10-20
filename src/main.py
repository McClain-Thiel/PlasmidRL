import click
import os
import torch.multiprocessing as mp


@click.group()
def cli():
    """PlasmidRL: Reinforcement Learning for Plasmid Design"""
    pass


@cli.command("train-es")
def train_es():
    """Train model using Evolution Strategies (ES)"""
    from src.runners.es import main
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()


@cli.command("train-grpo")
def train_grpo():
    """Train model using Group Relative Policy Optimization (GRPO)"""
    from src.runners.grpo import trainer
    trainer.train()


@cli.command("generate-samples")
def generate_samples():
    """Generate samples using vLLM"""
    from src.runners.generate_samples import main
    df = main()
    print(df.head())


@cli.command("convert-checkpoint")
@click.option("--checkpoint-path", required=True, help="S3 path to checkpoint (e.g., s3://bucket/path/to/checkpoint)")
@click.option("--hf-repo", required=True, help="HuggingFace repository path (e.g., username/repo-name)")
def convert_checkpoint(checkpoint_path: str, hf_repo: str):
    """Convert VERL/GRPO checkpoint to HuggingFace format and upload"""
    from src.utils.model_utils import checkpoint_to_huggingface, s3_client
    
    click.echo(f"Converting checkpoint from {checkpoint_path}")
    click.echo(f"Target HuggingFace repo: {hf_repo}")
    
    try:
        result_url = checkpoint_to_huggingface(s3_client, checkpoint_path, hf_repo)
        click.echo(f"✓ Successfully converted and uploaded checkpoint")
        click.echo(f"Model available at: {result_url}")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise


if __name__ == "__main__":
    cli()

