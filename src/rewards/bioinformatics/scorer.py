from src.rewards.bioinformatics.reward_config import RewardConfig
import plasmidkit as pk
from typing import Any, List, Tuple, Union, Dict
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
import os


class Scorer:
    def __init__(self, reward_config: RewardConfig):
        self.reward_config = reward_config
        # Fixed set for simplicity; location/length are folded into component scores/final modulation
        self.score_functions = [
            self.score_ori,
            self.score_promoter,
            self.score_terminator,
            self.score_marker,
            self.score_cds,
        ]
        self.weights = [
            self.reward_config.ori_weight,
            self.reward_config.promoter_weight,
            self.reward_config.terminator_weight,
            self.reward_config.marker_weight,
            self.reward_config.cds_weight,
        ]
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            uniform = 1.0 / len(self.score_functions)
            self.weights = [uniform for _ in self.score_functions]

    def annotate(self, sequence: str) -> Any:
        raw = pk.annotate(sequence, is_sequence=True)
        return self._preprocess_annotations(raw)

    # --- preprocessing helpers ---
    class _Feat:
        def __init__(self, type: str, id: str | None, start: int, end: int, strand: str | None, evidence: Any = None):
            self.type = type
            self.id = id
            self.start = int(start)
            self.end = int(end)
            self.strand = strand or "+"
            self.evidence = evidence or {}

    @staticmethod
    def _len(x: Any) -> int:
        return max(0, int(getattr(x, "end", 0)) - int(getattr(x, "start", 0)))

    @staticmethod
    def _overlap_len(a: Any, b: Any) -> int:
        s1, e1 = int(getattr(a, "start", 0)), int(getattr(a, "end", 0))
        s2, e2 = int(getattr(b, "start", 0)), int(getattr(b, "end", 0))
        lo = max(min(s1, e1), min(s2, e2))
        hi = min(max(s1, e1), max(s2, e2))
        return max(0, hi - lo)

    def _to_feat(self, x: Any) -> "Scorer._Feat":
        return Scorer._Feat(
            type=(getattr(x, "type", None) or "").lower(),
            id=getattr(x, "id", None),
            start=int(getattr(x, "start", 0)),
            end=int(getattr(x, "end", 0)),
            strand=getattr(x, "strand", "+"),
            evidence=getattr(x, "evidence", {}),
        )

    def _merge_group(self, feats: List[Any], threshold: float, *, respect_strand: bool) -> List["Scorer._Feat"]:
        if not feats:
            return []
        items = [self._to_feat(f) for f in feats]
        items.sort(key=lambda f: (f.strand, f.start, f.end))
        merged: List[Scorer._Feat] = []
        cur = items[0]
        for nxt in items[1:]:
            ovl = self._overlap_len(cur, nxt)
            min_len = max(1, min(self._len(cur), self._len(nxt)))
            strands_compatible = (cur.strand == nxt.strand) or (not respect_strand)
            if ovl / float(min_len) >= threshold and strands_compatible:
                cur.start = min(cur.start, nxt.start)
                cur.end = max(cur.end, nxt.end)
                cur.id = f"{cur.id}|{nxt.id}" if cur.id or nxt.id else None
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged

    def _preprocess_annotations(self, annotations: Any) -> List[Any]:
        feats = list(annotations)
        thr = float(self.reward_config.overlap_merge_threshold)
        type_key = lambda x: (getattr(x, "type", None) or "").lower()

        # collect groups
        groups: Dict[str, List[Any]] = {}
        for f in feats:
            groups.setdefault(type_key(f), []).append(f)

        # merge per group for relevant types
        merged_groups: Dict[str, List[Scorer._Feat]] = {}
        for t in ("rep_origin", "ori", "origin_of_replication", "promoter", "terminator", "marker", "cds"):
            if t in groups:
                # Ignore strand for ORI and marker; respect for others
                respect = t not in ("rep_origin", "ori", "origin_of_replication", "marker")
                merged_groups[t] = self._merge_group(groups[t], thr, respect_strand=respect)

        # suppress CDS if overlaps any non-CDS (ori/promoter/terminator/marker)
        non_cds: List[Scorer._Feat] = []
        for t in ("rep_origin", "ori", "origin_of_replication", "promoter", "terminator", "marker"):
            non_cds.extend(merged_groups.get(t, []))

        filtered_cds: List[Scorer._Feat] = []
        for c in merged_groups.get("cds", []):
            if any(self._overlap_len(c, o) > 0 for o in non_cds):
                continue
            filtered_cds.append(c)
        merged_groups["cds"] = filtered_cds

        # rebuild final list: prefer merged groups for those types; keep other features as-is
        final: List[Any] = []
        merged_types = set(merged_groups.keys())
        for t, items in merged_groups.items():
            final.extend(items)
        for f in feats:
            t = type_key(f)
            if t not in merged_types:
                final.append(f)
        return final

    @staticmethod
    def _read_fasta_file(path: str) -> str:
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]
        seq_lines = [ln for ln in lines if ln and not ln.startswith(">")]
        return "".join(seq_lines).replace(" ", "").upper()
    @staticmethod
    def _derive_source(sequence: str, provided: str | None) -> str:
        if provided:
            return provided
        # Try FASTA header if present
        if sequence and sequence.lstrip().startswith(">"):
            first = sequence.splitlines()[0].lstrip()[1:].strip()
            return first.split()[0] if first else "<sequence>"
        # Fallback: length + sha1 fingerprint
        sha = hashlib.sha1((sequence or "").encode("utf-8")).hexdigest()[:8]
        return f"seq:bp{len(sequence or '')}:sha{sha}"

    def runner(self, sequence: str, annotations: Any) -> List[float]:
        """
        Takes a list of runnable functions with the same signature and runs them in parallel.

        Args:
            sequence: The sequence to score
            annotations: The annotations to score

        Returns:
            A list of scores in the same order as `self.score_functions`.
        """
        with ThreadPoolExecutor(max_workers=len(self.score_functions)) as executor:
            futures = [executor.submit(func, sequence, annotations) for func in self.score_functions]
            return [future.result() for future in futures]

    # --- helpers ---
    @staticmethod
    def _feat_type(x: Any) -> str:
        return (getattr(x, "type", None) or "").lower()

    @staticmethod
    def _feat_id(x: Any) -> str:
        return (getattr(x, "id", None) or "").lower()

    @staticmethod
    def _strand(x: Any) -> str:
        return getattr(x, "strand", "+")

    @staticmethod
    def _start(x: Any) -> int:
        return int(getattr(x, "start", 0))

    @staticmethod
    def _end(x: Any) -> int:
        return int(getattr(x, "end", 0))

    @staticmethod
    def _distance(a: Any, b: Any) -> int:
        if Scorer._end(a) < Scorer._start(b):
            return Scorer._start(b) - Scorer._end(a)
        if Scorer._end(b) < Scorer._start(a):
            return Scorer._start(a) - Scorer._end(b)
        return 0

    @staticmethod
    def _in_order_same_strand(a: Any, b: Any) -> bool:
        if Scorer._strand(a) != Scorer._strand(b):
            return False
        if Scorer._strand(a) == "+":
            return Scorer._start(a) <= Scorer._start(b)
        return Scorer._end(a) >= Scorer._end(b)

    def _prox_points(self, d: int) -> float:
        return (
            float(self.reward_config.cassette_proximity_points)
            if d <= int(self.reward_config.proximity_threshold_bp)
            else 0.0
        )

    @staticmethod
    def _filter_allowed(xs: List[Any], allowed: List[str] | None) -> List[Any]:
        if not allowed:
            return xs
        allowed_l = [a.lower() for a in allowed]
        return [x for x in xs if any(a in Scorer._feat_id(x) for a in allowed_l)]

    def _count_score(self, count: int, min_req: int, max_allowed: int) -> float:
        if max_allowed <= 0:
            return 0.0
        # Below minimum: proportion of requirement; softer if punish_mode
        if count < min_req:
            proportion = (count / float(min_req)) if min_req > 0 else 0.0
            return max(0.0, min(1.0, proportion * (0.5 if self.reward_config.punish_mode else 1.0)))
        # Above maximum: penalize if punishing, otherwise cap at full credit
        if count > max_allowed:
            return float(self.reward_config.violation_penalty_factor) if self.reward_config.punish_mode else 1.0
        # Within [min, max]: full credit
        return 1.0

    def _compute_cassette_points(self, promoters: List[Any], cdss: List[Any], terminators: List[Any]) -> float:
        # Returns summed raw points across top-N cassettes
        triples: List[Tuple[Any, Any, Any, float]] = []
        for p in promoters:
            cds_same = [c for c in cdss if self._strand(c) == self._strand(p)]
            if not cds_same:
                continue
            cds_same.sort(key=lambda c: self._distance(p, c))
            c = cds_same[0]
            term_cands = [t for t in terminators if self._strand(t) == self._strand(c)]
            term_cands.sort(key=lambda t: self._distance(c, t))
            t = term_cands[0] if term_cands else None
            pts = 0.0
            # promoter -> CDS
            if self._in_order_same_strand(p, c):
                pts += self.reward_config.cassette_order_points
                pts += min(self.reward_config.cassette_proximity_points, self._prox_points(self._distance(p, c)))
            else:
                pts += min(3.0, self._prox_points(self._distance(p, c)))
            # CDS -> terminator
            if t is not None:
                if self._in_order_same_strand(c, t):
                    pts += self.reward_config.cassette_order_points
                    pts += min(self.reward_config.cassette_proximity_points, self._prox_points(self._distance(c, t)))
                else:
                    pts += min(3.0, self._prox_points(self._distance(c, t)))
            triples.append((p, c, t, pts))
        triples.sort(key=lambda x: x[3], reverse=True)
        top = triples[: int(self.reward_config.cassette_max_cassettes)]
        total = 0.0
        for (_, _, _, pts) in top:
            total += min(self.reward_config.cassette_max_points_per, float(pts))
        return total

    # --- component scores (0..1) ---
    def score_ori(self, seq: str, annotations: Any) -> float:
        feats = list(annotations)
        oris = [x for x in feats if self._feat_type(x) in ("rep_origin", "ori", "origin_of_replication")]
        oris = self._filter_allowed(oris, self.reward_config.allowed_oris)
        return self._count_score(len(oris), self.reward_config.ori_min, self.reward_config.ori_max)

    def score_promoter(self, seq: str, annotations: Any) -> float:
        feats = list(annotations)
        promoters = [x for x in feats if self._feat_type(x) == "promoter"]
        promoters = self._filter_allowed(promoters, self.reward_config.allowed_promoters)
        score = self._count_score(len(promoters), self.reward_config.promoter_min, self.reward_config.promoter_max)
        # small standalone credit (capped implicitly by count scoring)
        return score

    def score_terminator(self, seq: str, annotations: Any) -> float:
        feats = list(annotations)
        terms = [x for x in feats if self._feat_type(x) == "terminator"]
        terms = self._filter_allowed(terms, self.reward_config.allowed_terminators)
        return self._count_score(len(terms), self.reward_config.terminator_min, self.reward_config.terminator_max)

    def score_marker(self, seq: str, annotations: Any) -> float:
        feats = list(annotations)
        markers = [x for x in feats if self._feat_type(x) == "marker"]
        markers = self._filter_allowed(markers, self.reward_config.allowed_markers)
        if markers:
            return 1.0
        return 0.0 if self.reward_config.punish_mode else 0.5

    def score_cds(self, seq: str, annotations: Any) -> float:
        feats = list(annotations)
        cdss = [x for x in feats if self._feat_type(x) == "cds"]
        cdss = self._filter_allowed(cdss, self.reward_config.allowed_cds)
        base = self._count_score(len(cdss), self.reward_config.cds_min, self.reward_config.cds_max)

        if not self.reward_config.location_aware:
            return base

        promoters = [x for x in feats if self._feat_type(x) == "promoter"]
        terminators = [x for x in feats if self._feat_type(x) == "terminator"]
        cassette_pts = self._compute_cassette_points(promoters, cdss, terminators)
        max_total = self.reward_config.cassette_max_cassettes * self.reward_config.cassette_max_points_per
        bonus_scale = float(self.reward_config.location_bonus_scale)
        cassette_bonus = bonus_scale * (cassette_pts / max_total if max_total > 0 else 0.0)
        return max(0.0, min(1.0, base + cassette_bonus))

    def _length_factor(self, seq: str) -> float:
        if not self.reward_config.length_penalty:
            return 1.0
        L = len(seq or "")
        mn = self.reward_config.min_length
        mx = self.reward_config.max_length
        if mn is None and mx is None:
            return 1.0
        # inside range: full credit; outside: halve if punish_mode else no change
        in_range = (mn is None or L >= mn) and (mx is None or L <= mx)
        if in_range:
            return 1.0
        return float(self.reward_config.violation_penalty_factor) if self.reward_config.punish_mode else 1.0

    def score(self, seq: str, source: str | None = None) -> Tuple[float, Dict[str, float]]:
        t0 = time.perf_counter()
        src = Scorer._derive_source(seq, source)
        annotations = self.annotate(seq)
        results = self.runner(seq, annotations)
        ori, prom, term, mark, cds = results
        base = sum(w * r for w, r in zip(self.weights, results))
        length_factor = self._length_factor(seq)
        final = max(0.0, min(1.0, base * length_factor))
        components: Dict[str, float] = {
            "ori": float(ori),
            "promoter": float(prom),
            "terminator": float(term),
            "marker": float(mark),
            "cds": float(cds),
            "length_factor": float(length_factor),
        }
        dt_ms = (time.perf_counter() - t0) * 1000.0
        cfg = self.reward_config.model_dump(exclude_none=True)
        print(f"source={src} config={cfg} score={final:.4f} len={len(seq)} time_ms={dt_ms:.2f} components={components}")
        return float(final), components

    def score_fasta(self, fasta_path: str) -> Tuple[float, Dict[str, float]]:
        seq = Scorer._read_fasta_file(fasta_path)
        return self.score(seq, source=os.path.basename(fasta_path))