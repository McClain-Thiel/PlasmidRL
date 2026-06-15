"""Rejection sampling and best-of-N baselines.

Generates samples from the pretrained and SFT base models, scores them with
the full reward function, and reports pass rates.

Outputs are saved as CSV plus one FASTA per sequence (the format consumed by
the downstream QC/analysis pipeline).
"""

import datetime
import os
import re
import time
from pathlib import Path

import pandas as pd
import wandb
from vllm import LLM, SamplingParams

from plasmidrl.ablations import get_ablation_config
from plasmidrl.config import Config
from plasmidrl.rewards.bioinformatics.scorer import Scorer


DEFAULT_MODELS = {
    "Base": "UCL-CSSB/PlasmidGPT",
    "SFT": "UCL-CSSB/PlasmidGPT-SFT",
}


def _get_prompts(cfg: Config) -> list[str]:
    return ["ATG", cfg.default_query]


def _clean_sequence(text: str) -> str:
    return re.sub(r'[^ATCG]', '', text.upper().replace(" ", ""))


def _save_outputs(df: pd.DataFrame, output_dir: str, model_label: str):
    """Write `outputs.csv` plus one FASTA per sequence."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "outputs.csv", index=False)
    for i, row in df.iterrows():
        seq_id = row.get("id", f"seq_{i}")
        (out / f"seq_{i}.fasta").write_text(f">{seq_id}\n{row['full']}\n")
    print(f"[{model_label}] Saved {len(df)} sequences to {output_dir}")


_SAMPLING_PARAMS = SamplingParams(
    max_tokens=256,
    temperature=1.1,
    top_p=0.90,
    stop_token_ids=[2],
)


def run_rejection_sampling(
    model_path: str,
    model_label: str,
    prompts: list[str],
    n_samples: int,
    scorer: Scorer,
    output_dir: str,
    gpu_util: float = 0.8,
) -> pd.DataFrame:
    """Generate `n_samples` sequences (split evenly across prompts) and score them."""
    samples_per_prompt = n_samples // len(prompts)

    print(f"\n[{model_label}] Loading model: {model_path}")
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_util, trust_remote_code=True)

    expanded = [p for p in prompts for _ in range(samples_per_prompt)]

    print(f"[{model_label}] Generating {len(expanded)} samples ({samples_per_prompt} per prompt)...")
    t0 = time.time()
    outputs = llm.generate(expanded, _SAMPLING_PARAMS)
    gen_time = time.time() - t0

    records = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text.replace(" ", "")
        full = prompt + completion
        cleaned = _clean_sequence(full)
        reward, _ = scorer.score(cleaned)

        records.append({
            "id": f"seq_{i}",
            "prompt": prompt[:50],
            "prompt_full": prompt,
            "completion": completion,
            "full": full,
            "length": len(cleaned),
            "reward": float(reward),
        })

    df = pd.DataFrame(records)
    _save_outputs(df, output_dir, model_label)

    print(f"[{model_label}] Generation time: {gen_time:.1f}s ({len(df)/gen_time:.1f} samples/sec)")
    print(f"[{model_label}] Mean reward: {df['reward'].mean():.4f}")
    print(f"[{model_label}] Reward > 0.5: {(df['reward'] > 0.5).mean()*100:.1f}%")
    print(f"[{model_label}] Reward > 0.7: {(df['reward'] > 0.7).mean()*100:.1f}%")

    del llm
    return df


def run_best_of_n(
    model_path: str,
    model_label: str,
    prompts: list[str],
    group_size: int,
    target_per_prompt: int,
    scorer: Scorer,
    output_dir: str,
    gpu_util: float = 0.8,
) -> pd.DataFrame:
    """Generate `group_size` candidates per prompt and keep the highest-reward one."""
    print(f"\n[{model_label}] Best-of-{group_size}: Loading model: {model_path}")
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_util, trust_remote_code=True)

    expanded = []
    prompt_indices = []
    for pi, prompt in enumerate(prompts):
        expanded.extend([prompt] * (target_per_prompt * group_size))
        prompt_indices.extend([pi] * (target_per_prompt * group_size))

    print(f"[{model_label}] Generating {len(expanded)} samples for best-of-{group_size} selection...")
    t0 = time.time()
    outputs = llm.generate(expanded, _SAMPLING_PARAMS)
    gen_time = time.time() - t0

    all_records = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text.replace(" ", "")
        full = prompt + completion
        cleaned = _clean_sequence(full)
        reward, _ = scorer.score(cleaned)

        all_records.append({
            "prompt_idx": prompt_indices[i],
            "prompt": prompt[:50],
            "prompt_full": prompt,
            "completion": completion,
            "full": full,
            "length": len(cleaned),
            "reward": float(reward),
        })

    df_all = pd.DataFrame(all_records)

    selected = []
    for pi in range(len(prompts)):
        prompt_df = df_all[df_all["prompt_idx"] == pi].reset_index(drop=True)
        for g in range(target_per_prompt):
            group = prompt_df.iloc[g * group_size : (g + 1) * group_size]
            selected.append(group.loc[group["reward"].idxmax()])

    df_selected = pd.DataFrame(selected).reset_index(drop=True)
    df_selected["id"] = [f"seq_{i}" for i in range(len(df_selected))]
    _save_outputs(df_selected, output_dir, model_label)

    print(f"[{model_label}] Best-of-{group_size} time: {gen_time:.1f}s")
    print(f"[{model_label}] Selected {len(df_selected)} samples")
    print(f"[{model_label}] Mean reward (selected): {df_selected['reward'].mean():.4f}")
    print(f"[{model_label}] Reward > 0.5: {(df_selected['reward'] > 0.5).mean()*100:.1f}%")

    del llm
    return df_selected


def main(
    n_samples: int = 10000,
    best_of_n: int = 16,
    model_name: str | None = None,
    output_base: str = "results/baselines",
):
    """Run rejection sampling and best-of-N for baseline models.

    If `model_name` is given, only that HF model is evaluated (under the key
    `custom`); otherwise the two DEFAULT_MODELS are used.
    """
    cfg = Config()
    prompts = _get_prompts(cfg)
    scorer = Scorer(get_ablation_config("full_reward"))

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    models = {"custom": model_name} if model_name else DEFAULT_MODELS
    s3_base = "s3://phd-research-storage-1758274488/icml-revision/baselines"

    wandb_run = wandb.init(
        project="plasmid-rl-icml-revision",
        entity=cfg.wandb_entity,
        group="baselines",
        name=f"rejection-sampling-{timestamp}",
        tags=["baselines", "rejection-sampling"],
        notes=(
            f"Rejection sampling & best-of-{best_of_n} baselines.\n\n"
            f"Generates {n_samples} samples per model from Base (UCL-CSSB/PlasmidGPT) "
            f"and SFT (UCL-CSSB/PlasmidGPT-SFT), scores with the full reward function, "
            f"and selects best-of-{best_of_n} per group.\n\n"
            f"## Data locations\n"
            f"- S3 results: {s3_base}/\n"
            f"  - `rejection_sampling/{{model}}/outputs.csv` — all scored samples\n"
            f"  - `rejection_sampling/{{model}}/seq_*.fasta` — individual sequences\n"
            f"  - `best_of_{best_of_n}/{{model}}/outputs.csv` — selected samples\n"
            f"  - `best_of_{best_of_n}/{{model}}/seq_*.fasta` — selected sequences\n"
            f"- Prompts: ATG + GFP cassette ({len(prompts)} prompts)\n"
        ),
        config={
            "n_samples": n_samples,
            "best_of_n": best_of_n,
            "models": models,
            "prompts": [p[:50] for p in prompts],
            "s3_results": s3_base,
        },
    )

    target_per_prompt = 500
    all_results = {}

    for label, model_path in models.items():
        rs_dir = os.path.join(output_base, f"rejection_sampling/{label}_{timestamp}")
        df_rs = run_rejection_sampling(
            model_path=model_path, model_label=f"{label}/RS",
            prompts=prompts, n_samples=n_samples,
            scorer=scorer, output_dir=rs_dir,
        )

        bon_dir = os.path.join(output_base, f"best_of_{best_of_n}/{label}_{timestamp}")
        df_bon = run_best_of_n(
            model_path=model_path, model_label=f"{label}/BoN",
            prompts=prompts, group_size=best_of_n,
            target_per_prompt=target_per_prompt,
            scorer=scorer, output_dir=bon_dir,
        )

        wandb.log({
            f"rs/{label}/mean_reward": df_rs["reward"].mean(),
            f"rs/{label}/pass_rate_0.5": (df_rs["reward"] > 0.5).mean(),
            f"rs/{label}/pass_rate_0.7": (df_rs["reward"] > 0.7).mean(),
            f"rs/{label}/n_samples": len(df_rs),
            f"bon/{label}/mean_reward": df_bon["reward"].mean(),
            f"bon/{label}/pass_rate_0.5": (df_bon["reward"] > 0.5).mean(),
            f"bon/{label}/pass_rate_0.7": (df_bon["reward"] > 0.7).mean(),
            f"bon/{label}/n_selected": len(df_bon),
        })

        all_results[label] = {"rejection_sampling": df_rs, "best_of_n": df_bon}

    summary_rows = []
    for label, results in all_results.items():
        for method, df in results.items():
            summary_rows.append({
                "model": label,
                "method": method,
                "n_samples": len(df),
                "mean_reward": df["reward"].mean(),
                "pass_rate_0.5": (df["reward"] > 0.5).mean(),
                "pass_rate_0.7": (df["reward"] > 0.7).mean(),
                "mean_length": df["length"].mean(),
            })
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

    wandb.log({"summary": wandb.Table(dataframe=summary_df)})

    for label, results in all_results.items():
        for method, df in results.items():
            artifact = wandb.Artifact(
                name=f"baseline-{label}-{method}".replace("_", "-"),
                type="baseline-sequences",
                description=f"{method} results for {label} model ({len(df)} sequences)",
            )
            csv_path = f"/tmp/{label}_{method}.csv"
            df.to_csv(csv_path, index=False)
            artifact.add_file(csv_path)
            wandb_run.log_artifact(artifact)

    wandb.finish()
    return all_results
