"""Canonical model identifiers and loaders.

Three models are referenced throughout the paper:

- ``BASE`` — pretrained autoregressive plasmid LM with no fine-tuning.
- ``SFT``  — supervised-fine-tuned variant trained on the Addgene corpus.
- ``RL``   — GRPO-trained variant; the published headline model.

The optimal sampling temperature for each model is the one used in every
headline result; deviating from it shifts the QC pass rate by tens of
percentage points and breaks comparability with the paper figures.
"""
from __future__ import annotations

from typing import Tuple

BASE = "UCL-CSSB/PlasmidGPT"
SFT = "UCL-CSSB/PlasmidGPT-SFT"
RL = "UCL-CSSB/PlasmidGPT-GRPO"

OPTIMAL_T: dict[str, float] = {"Base": 1.00, "SFT": 1.00, "RL": 1.15}

_NAME_TO_PATH: dict[str, str] = {"Base": BASE, "SFT": SFT, "RL": RL}


def path(name: str) -> str:
    """Return the canonical Hugging Face path for a model label."""
    if name not in _NAME_TO_PATH:
        raise ValueError(
            f"unknown model {name!r}; expected one of {list(_NAME_TO_PATH)}")
    return _NAME_TO_PATH[name]


def load(name: str) -> Tuple["object", "object"]:
    """Load (model, tokenizer) for a canonical label.

    Imports ``transformers`` lazily so that callers who only use the model
    constants don't pay the heavy import cost.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = path(name)
    return (AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True),
            AutoTokenizer.from_pretrained(p, trust_remote_code=True))


__all__ = ["BASE", "SFT", "RL", "OPTIMAL_T", "path", "load"]
