import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import LlamaConfig, LlamaForCausalLM

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tokenizers.naive_dna_kmer import NaiveDNAKmerTokenizer


def write_model_card(dst: Path, repo_id: str) -> None:
    card = f"""
---
language: en
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
  - dna
  - kmer
  - naive
  - llama
  - rl
---

# {repo_id}

Randomly initialized LLaMA model for DNA sequence generation using a custom 6-mer tokenizer (A,C,G,T) with stride 2.

This repository contains:
- A custom tokenizer (`tokenization_naive_dna_kmer.py`) implementing 6-mer tokenization with stride 2
- Tokenizer files with `auto_map` so `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` works
- A randomly initialized `LlamaForCausalLM` config and weights sized per the provided hyperparameters

Intended usage:
- Research baseline for reinforcement learning from a completely untrained policy
- Load with trust_remote_code=True so the custom tokenizer can be imported

Example:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "{repo_id}"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
```

Notes:
- Tokenizer vocabulary size is 4096 6-mers + 4 specials (BOS/EOS/PAD/UNK)
- Decoding reconstructs the DNA string by overlapping the last 2 bases of successive k-mers
- Model is untrained; it should be optimized purely via RL or other post-training methods
"""
    (dst / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish naive DNA LLaMA to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Destination repo id, e.g., McClain/naive-dna-llama-6mer")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    parser.add_argument("--context", type=int, default=2048, help="Max position embeddings")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden size")
    parser.add_argument("--heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--layers", type=int, default=8, help="Transformer layers")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var required for uploading to Hugging Face")

    api = HfApi(token=hf_token)
    create_repo(args.repo_id, private=args.private, exist_ok=True, token=hf_token)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "naive_dna_llama"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Tokenizer
        tokenizer = NaiveDNAKmerTokenizer(k=6, stride=2)
        tokenizer.padding_side = "left"
        tokenizer.save_pretrained(str(out_dir))
        # Ensure tokenizer config exposes the custom class and automap
        tok_cfg_path = out_dir / "tokenizer_config.json"
        tok_cfg = {}
        if tok_cfg_path.exists():
            tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        tok_cfg["tokenizer_class"] = "NaiveDNAKmerTokenizer"
        tok_cfg["auto_map"] = {"AutoTokenizer": "tokenization_naive_dna_kmer.NaiveDNAKmerTokenizer"}
        tok_cfg_path.write_text(json.dumps(tok_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        # Include tokenizer implementation so trust_remote_code can import from repo
        src_impl = Path(__file__).resolve().parents[1] / "src" / "tokenizers" / "naive_dna_kmer.py"
        shutil.copyfile(src_impl, out_dir / "tokenization_naive_dna_kmer.py")

        # Model
        hidden_size = args.hidden
        num_attention_heads = args.heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden must be divisible by heads")
        config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=args.layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=args.context,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            rope_scaling=None,
        )
        model = LlamaForCausalLM(config)
        model.save_pretrained(str(out_dir))

        # Model card
        write_model_card(out_dir, args.repo_id)

        # Upload folder
        upload_folder(
            folder_path=str(out_dir),
            repo_id=args.repo_id,
            token=hf_token,
            commit_message="Initial naive DNA LLaMA and tokenizer",
        )

    print(f"Published {args.repo_id} to Hugging Face")


if __name__ == "__main__":
    main()


