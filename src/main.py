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


@cli.command("qc")
@click.option("--input-path", required=True, help="Local dir/file or s3://bucket/prefix containing .fasta files")
@click.option("--oridb-prefix", required=False, default=None, help="Prefix/path for ori BLAST DB (nucl)")
@click.option("--oridb-ref", required=False, default=None, help="FASTA of ori references to build DB if missing")
@click.option("--threads", required=False, default=1, type=int, help="Threads for BLAST/AMRFinder")
@click.option("--skip-prodigal", is_flag=True, default=False, help="Skip running Prodigal")
def qc(input_path: str, oridb_prefix: str | None, oridb_ref: str | None, threads: int, skip_prodigal: bool):
    """Run plasmid QC on FASTA files; writes report back to same location and prints summary paths."""
    from src.runners.qc_pipeline import main as qc_main

    result = qc_main(
        input_path=input_path,
        oridb_prefix=oridb_prefix,
        oridb_ref=oridb_ref,
        threads=threads,
        skip_prodigal=skip_prodigal,
    )
    click.echo(result)

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

