import os
import pytest

from src.rewards.scorer import Scorer
from src.rewards.reward_config import RewardConfig
from src.rewards import rewards as rewards_mod


def _read_fasta_sequence(path: str) -> str:
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    seq_lines = [ln for ln in lines if not ln.startswith(">") and ln]
    return "".join(seq_lines).replace(" ", "").replace("\n", "").upper()


def test_scorer_components_and_bounds():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    fasta_paths = [
        os.path.join(data_dir, "pUC19.fasta"),
        os.path.join(data_dir, "pSC101.fasta"),
        os.path.join(data_dir, "RF0G-IodoY.fasta"),
    ]
    cfg = RewardConfig()
    scorer = Scorer(cfg)

    for p in fasta_paths:
        score, components = scorer.score_fasta(p)
        cfg_dump = cfg.model_dump(exclude_none=True)
        seq = _read_fasta_sequence(p)
        ann = scorer.annotate(seq)
        ann_view = [
            {
                "type": getattr(a, "type", None),
                "id": getattr(a, "id", None),
                "start": getattr(a, "start", None),
                "end": getattr(a, "end", None),
                "strand": getattr(a, "strand", None),
            }
            for a in ann if getattr(a, "type", None) != "restriction_site"
        ]
        print(
            f"testlog file={os.path.basename(p)} config={cfg_dump} score={score:.4f} components={components} annotations={ann_view}"
        )

        assert 0.0 <= score <= 1.0
        for key in ["ori", "promoter", "terminator", "marker", "cds", "length_factor"]:
            assert key in components
            assert 0.0 <= float(components[key]) <= 1.0 or key == "length_factor"


def test_location_bonus_non_decreasing():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, "pUC19.fasta")
    seq = _read_fasta_sequence(path)
    cfg_off = RewardConfig(location_aware=False)
    cfg_on = RewardConfig(location_aware=True)
    scorer_off = Scorer(cfg_off)
    scorer_on = Scorer(cfg_on)

    score_off, comp_off = scorer_off.score(seq, source=os.path.basename(path))
    score_on, comp_on = scorer_on.score(seq, source=os.path.basename(path))
    ann = scorer_on.annotate(seq)
    ann_view = [
        {
            "type": getattr(a, "type", None),
            "id": getattr(a, "id", None),
            "start": getattr(a, "start", None),
            "end": getattr(a, "end", None),
            "strand": getattr(a, "strand", None),
        }
        for a in ann if getattr(a, "type", None) != "restriction_site"
    ]
    print(f"testlog file={os.path.basename(path)} config_off={cfg_off.model_dump()} score_off={score_off:.4f} comp_off={comp_off}")
    print(f"testlog file={os.path.basename(path)} config_on={cfg_on.model_dump()} score_on={score_on:.4f} comp_on={comp_on} annotations={ann_view}")

    assert comp_on["cds"] >= comp_off["cds"]
    assert score_on >= score_off


def test_wrapper_score_sequence_bounds_and_consistency():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, "pSC101.fasta")
    seq = _read_fasta_sequence(path)
    # wrapper returns 0..100
    s100 = rewards_mod.score_sequence(seq)
    print(f"testlog file={os.path.basename(path)} wrapper_score_0_100={s100:.2f}")
    assert 0.0 <= s100 <= 100.0


def test_score_completions_handles_empty():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, "RF0G-IodoY.fasta")
    seq = _read_fasta_sequence(path)
    scores = rewards_mod.score_completions([seq, ""]) 
    print(f"testlog file={os.path.basename(path)} completions_scores={scores}")
    assert len(scores) == 2
    assert scores[1] == 0.0
    assert 0.0 <= scores[0] <= 100.0
