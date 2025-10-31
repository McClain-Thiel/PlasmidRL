import os
import pytest

from src.rewards.bioinformatics.scorer import Scorer
from src.rewards.bioinformatics.reward_config import RewardConfig


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
        seq = _read_fasta_sequence(p)
        score, components = scorer.score(seq, source=os.path.basename(p))
        cfg_dump = cfg.model_dump(exclude_none=True)
        ann = scorer.annotate(seq)
        ann_view = [
            {
                "type": a.type if hasattr(a, "type") else None,
                "id": a.id if hasattr(a, "id") else None,
                "start": a.start if hasattr(a, "start") else None,
                "end": a.end if hasattr(a, "end") else None,
                "strand": a.strand if hasattr(a, "strand") else None,
            }
            for a in ann if hasattr(a, "type") and a.type != "restriction_site"
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
            "type": a.type if hasattr(a, "type") else None,
            "id": a.id if hasattr(a, "id") else None,
            "start": a.start if hasattr(a, "start") else None,
            "end": a.end if hasattr(a, "end") else None,
            "strand": a.strand if hasattr(a, "strand") else None,
        }
        for a in ann if hasattr(a, "type") and a.type != "restriction_site"
    ]
    print(f"testlog file={os.path.basename(path)} config_off={cfg_off.model_dump()} score_off={score_off:.4f} comp_off={comp_off}")
    print(f"testlog file={os.path.basename(path)} config_on={cfg_on.model_dump()} score_on={score_on:.4f} comp_on={comp_on} annotations={ann_view}")

    assert comp_on["cds"] >= comp_off["cds"]
    assert score_on >= score_off


def test_scorer_score_bounds():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, "pSC101.fasta")
    seq = _read_fasta_sequence(path)
    cfg = RewardConfig()
    scorer = Scorer(cfg)
    s, _ = scorer.score(seq)
    print(f"testlog file={os.path.basename(path)} scorer_score_0_1={s:.4f}")
    assert 0.0 <= s <= 1.0


def test_simple_batch_handles_empty():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, "RF0G-IodoY.fasta")
    seq = _read_fasta_sequence(path)
    cfg = RewardConfig()
    scorer = Scorer(cfg)
    s_valid, _ = scorer.score(seq)
    print(f"testlog file={os.path.basename(path)} simple_score_valid={s_valid}")
    assert 0.0 <= s_valid <= 1.0
    with pytest.raises(Exception):
        scorer.score("")
