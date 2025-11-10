from src.rewards.bioinformatics.reward_config import RewardConfig
import plasmidkit as pk
from typing import Any, List, Tuple, Dict
import time


class _Feat:
    """Simple feature container for annotation merging."""
    def __init__(self, type: str, id: str | None, start: int, end: int, strand: str | None, evidence: Any = None):
        self.type = type
        self.id = id
        self.start = int(start)
        self.end = int(end)
        self.strand = strand or "+"
        self.evidence = evidence or {}


class Scorer:
    """
    Scores plasmid sequences based on biological features (ori, promoter, terminator, marker, CDS).
    
    Uses plasmidkit for annotation, then computes weighted scores for each component.
    Optionally applies length penalties and location-aware bonuses for gene cassettes.
    """
    def __init__(self, reward_config: RewardConfig):
        self.reward_config = reward_config
        if self.reward_config.length_penalty:
            assert (
                self.reward_config.min_length is not None or self.reward_config.max_length is not None
            ), "At least one of min_length or max_length must be set if length_penalty is True"
            if (
                self.reward_config.min_length is not None 
                and self.reward_config.max_length is not None
            ):
                assert self.reward_config.min_length < self.reward_config.max_length, "min_length must be less than max_length"

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
        assert total_weight > 0, "Total weight must be greater than 0"
        self.weights = [w / total_weight for w in self.weights]

    def annotate(self, sequence: str) -> Any:
        """Annotate sequence with plasmidkit and merge overlapping features."""
        assert sequence, "sequence cannot be empty"
        raw = pk.annotate(sequence, is_sequence=True)
        return self._preprocess_annotations(raw)

    @staticmethod
    def _overlap_len(a: Any, b: Any) -> int:
        """Calculate overlap length between two features."""
        s1, e1 = int(a.start), int(a.end)
        s2, e2 = int(b.start), int(b.end)
        lo = max(min(s1, e1), min(s2, e2))
        hi = min(max(s1, e1), max(s2, e2))
        return max(0, hi - lo)

    @staticmethod
    def _canonical_type(t: str | None) -> str:
        """
        Map annotation types from plasmidkit to canonical categories used by scoring and merging.
        Canonical types: 'ori', 'promoter', 'terminator', 'marker', 'cds'.
        """
        if not t:
            return ""
        tl = t.lower()
        # Origins of replication
        if tl in ("rep_origin", "origin_of_replication", "ori"):
            return "ori"
        # Selectable markers / antibiotic resistance genes
        if tl in ("marker", "selectable_marker", "antibiotic_resistance", "antibiotic_resistance_gene", "arg"):
            return "marker"
        # Standard categories
        if tl in ("promoter", "terminator", "cds"):
            return tl
        # Default: return lowercased original
        return tl
    
    @staticmethod
    def _revcomp(s: str) -> str:
        """Reverse complement of a DNA string (expects A/T/C/G)."""
        comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
        return "".join(comp.get(c, "N") for c in reversed(s))
    
    def _find_repeat_regions(self, seq: str) -> List[Tuple[int, int]]:
        """
        Find regions containing repeats of length >= repeat_min_length.
        Includes reverse complements (always enabled).
        Returns merged intervals [start, end) for each repeated occurrence region.
        """
        if not self.reward_config.repeat_penalty_enabled:
            return []
        k = max(1, int(self.reward_config.repeat_min_length))
        n = len(seq)
        if n < k:
            return []
        seq_u = seq.upper()
        valid = set("ATCG")
        index_map: Dict[str, List[int]] = {}
        for i in range(0, n - k + 1):
            kmer = seq_u[i:i+k]
            if any(c not in valid for c in kmer):
                continue
            rc = self._revcomp(kmer)
            canon = kmer if kmer <= rc else rc
            index_map.setdefault(canon, []).append(i)
        intervals: List[Tuple[int, int]] = []
        for _, idxs in index_map.items():
            if len(idxs) >= 2:
                for s in idxs:
                    intervals.append((s, s + k))
        if not intervals:
            return []
        intervals.sort(key=lambda x: (x[0], x[1]))
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def _to_feat(self, x: Any) -> "_Feat":
        """Convert annotation object to internal _Feat representation."""
        return _Feat(
            type=self._canonical_type(x.type),
            id=x.id if hasattr(x, "id") else None,
            start=int(x.start),
            end=int(x.end),
            strand=x.strand if hasattr(x, "strand") else "+",
            evidence=x.evidence if hasattr(x, "evidence") else {},
        )

    def _merge_group(self, feats: List[Any], threshold: float, *, respect_strand: bool) -> List["_Feat"]:
        """Merge overlapping features of the same type based on overlap threshold."""
        if not feats:
            return []
        items = [self._to_feat(f) for f in feats]
        items.sort(key=lambda f: (f.strand, f.start, f.end))
        merged: List[_Feat] = []
        cur = items[0]
        for nxt in items[1:]:
            ovl = self._overlap_len(cur, nxt)
            # Inline length calculation
            cur_len = max(0, cur.end - cur.start)
            nxt_len = max(0, nxt.end - nxt.start)
            min_len = max(1, min(cur_len, nxt_len))
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
        """
        Merge overlapping annotations and filter out CDS overlapping with other feature types.
        
        Groups features by type, merges overlapping ones based on threshold, and removes
        CDS annotations that overlap with ori/promoter/terminator/marker features.
        """
        feats = list(annotations)
        thr = float(self.reward_config.overlap_merge_threshold)
        type_key = lambda x: self._canonical_type(x.type)

        # Collect groups by type
        groups: Dict[str, List[Any]] = {}
        for f in feats:
            groups.setdefault(type_key(f), []).append(f)

        # Merge per group for relevant types
        merged_groups: Dict[str, List[_Feat]] = {}
        for t in ("ori", "promoter", "terminator", "marker", "cds"):
            if t in groups:
                # Ignore strand for ORI and marker; respect for others
                respect = t not in ("ori", "marker")
                merged_groups[t] = self._merge_group(groups[t], thr, respect_strand=respect)

        # Suppress CDS if overlaps any non-CDS (ori/promoter/terminator/marker)
        non_cds: List[_Feat] = []
        for t in ("ori", "promoter", "terminator", "marker"):
            non_cds.extend(merged_groups.get(t, []))

        filtered_cds: List[_Feat] = []
        for c in merged_groups.get("cds", []):
            if any(self._overlap_len(c, o) > 0 for o in non_cds):
                continue
            filtered_cds.append(c)
        merged_groups["cds"] = filtered_cds

        # Rebuild final list: prefer merged groups for those types; keep other features as-is
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
    def _distance(a: Any, b: Any) -> int:
        """Calculate distance between two non-overlapping features (0 if overlapping)."""
        a_start, a_end = int(a.start), int(a.end)
        b_start, b_end = int(b.start), int(b.end)
        if a_end < b_start:
            return b_start - a_end
        if b_end < a_start:
            return a_start - b_end
        return 0

    @staticmethod
    def _in_order_same_strand(a: Any, b: Any) -> bool:
        """Check if features a and b are on same strand and in correct order."""
        strand_a = a.strand if hasattr(a, "strand") else "+"
        strand_b = b.strand if hasattr(b, "strand") else "+"
        if strand_a != strand_b:
            return False
        start_a = int(a.start)
        start_b = int(b.start)
        if strand_a == "+":
            return start_a <= start_b
        end_a = int(a.end)
        end_b = int(b.end)
        return end_a >= end_b

    @staticmethod
    def _filter_allowed(xs: List[Any], allowed: List[str] | None) -> List[Any]:
        """Filter features to only those matching allowed IDs."""
        if not allowed:
            return xs
        allowed_l = [a.lower() for a in allowed]
        return [x for x in xs if hasattr(x, "id") and x.id and any(a in x.id.lower() for a in allowed_l)]

    def _count_score(self, count: int, min_req: int, max_allowed: int) -> float:
        """
        Score feature count based on min/max requirements.
        
        Returns 1.0 if within range, proportional score if below min, and penalty if above max.
        """
        if max_allowed <= 0:
            return 0.0
        # Below minimum: proportion of requirement
        if count < min_req:
            proportion = (count / float(min_req)) if min_req > 0 else 0.0
            return max(0.0, min(1.0, proportion * (0.5 if self.reward_config.punish_mode else 1.0)))
        # Above maximum: penalize if punishing, otherwise cap at full credit
        if count > max_allowed:
            return float(self.reward_config.violation_penalty_factor) if self.reward_config.punish_mode else 1.0
        # Within [min, max]: full credit
        return 1.0

    def _compute_cassette_points(self, promoters: List[Any], cdss: List[Any], terminators: List[Any]) -> float:
        """
        Compute bonus points for well-formed gene cassettes (promoter -> CDS -> terminator).
        
        Awards points for correct order and proximity on same strand.
        Returns summed raw points across top-N cassettes.
        """
        triples: List[Tuple[Any, Any, Any, float]] = []
        for p in promoters:
            # Find closest CDS on same strand
            p_strand = p.strand if hasattr(p, "strand") else "+"
            cds_same = [c for c in cdss if (c.strand if hasattr(c, "strand") else "+") == p_strand]
            if not cds_same:
                continue
            cds_same.sort(key=lambda c: self._distance(p, c))
            c = cds_same[0]
            
            # Find closest terminator on same strand as CDS
            c_strand = c.strand if hasattr(c, "strand") else "+"
            term_cands = [t for t in terminators if (t.strand if hasattr(t, "strand") else "+") == c_strand]
            term_cands.sort(key=lambda t: self._distance(c, t))
            t = term_cands[0] if term_cands else None
            
            pts = 0.0
            # Promoter -> CDS
            dist_pc = self._distance(p, c)
            if self._in_order_same_strand(p, c):
                pts += self.reward_config.cassette_order_points
                if dist_pc <= self.reward_config.proximity_threshold_bp:
                    pts += self.reward_config.cassette_proximity_points
            else:
                if dist_pc <= self.reward_config.proximity_threshold_bp:
                    pts += min(3.0, self.reward_config.cassette_proximity_points)
            
            # CDS -> Terminator
            if t is not None:
                dist_ct = self._distance(c, t)
                if self._in_order_same_strand(c, t):
                    pts += self.reward_config.cassette_order_points
                    if dist_ct <= self.reward_config.proximity_threshold_bp:
                        pts += self.reward_config.cassette_proximity_points
                else:
                    if dist_ct <= self.reward_config.proximity_threshold_bp:
                        pts += min(3.0, self.reward_config.cassette_proximity_points)
            
            triples.append((p, c, t, pts))
        
        # Take top N cassettes
        triples.sort(key=lambda x: x[3], reverse=True)
        top = triples[: int(self.reward_config.cassette_max_cassettes)]
        return sum(min(self.reward_config.cassette_max_points_per, pts) for _, _, _, pts in top)

    def score_ori(self, seq: str, annotations: Any) -> float:
        """Score origin of replication features."""
        feats = list(annotations)
        oris = [x for x in feats if x.type and x.type.lower() == "ori"]
        oris = self._filter_allowed(oris, self.reward_config.allowed_oris)
        return self._count_score(len(oris), self.reward_config.ori_min, self.reward_config.ori_max)

    def score_promoter(self, seq: str, annotations: Any) -> float:
        """Score promoter features."""
        feats = list(annotations)
        promoters = [x for x in feats if x.type and x.type.lower() == "promoter"]
        promoters = self._filter_allowed(promoters, self.reward_config.allowed_promoters)
        return self._count_score(len(promoters), self.reward_config.promoter_min, self.reward_config.promoter_max)

    def score_terminator(self, seq: str, annotations: Any) -> float:
        """Score terminator features."""
        feats = list(annotations)
        terms = [x for x in feats if x.type and x.type.lower() == "terminator"]
        terms = self._filter_allowed(terms, self.reward_config.allowed_terminators)
        return self._count_score(len(terms), self.reward_config.terminator_min, self.reward_config.terminator_max)

    def score_marker(self, seq: str, annotations: Any) -> float:
        """Score selectable marker features (binary: present or absent)."""
        feats = list(annotations)
        markers = [x for x in feats if x.type and x.type.lower() == "marker"]
        markers = self._filter_allowed(markers, self.reward_config.allowed_markers)
        if markers:
            return 1.0
        return 0.0 if self.reward_config.punish_mode else 0.5

    def score_cds(self, seq: str, annotations: Any) -> float:
        """Score CDS features with optional location-aware cassette bonus."""
        feats = list(annotations)
        cdss = [x for x in feats if x.type and x.type.lower() == "cds"]
        cdss = self._filter_allowed(cdss, self.reward_config.allowed_cds)
        base = self._count_score(len(cdss), self.reward_config.cds_min, self.reward_config.cds_max)

        if not self.reward_config.location_aware:
            return base

        # Add cassette bonus for properly arranged promoter->CDS->terminator
        promoters = [x for x in feats if x.type and x.type.lower() == "promoter"]
        terminators = [x for x in feats if x.type and x.type.lower() == "terminator"]
        cassette_pts = self._compute_cassette_points(promoters, cdss, terminators)
        max_total = self.reward_config.cassette_max_cassettes * self.reward_config.cassette_max_points_per
        cassette_bonus = self.reward_config.location_bonus_scale * (cassette_pts / max_total if max_total > 0 else 0.0)
        return max(0.0, min(1.0, base + cassette_bonus))

    def score_length(self, seq: str, annotations: Any) -> float:
        """
        Score sequence length with rewards for being within target range.
        
        If length_reward_mode=True:
            - Sequences within [min_length, max_length] get full credit (1.0)
            - Sequences within [ideal_min, ideal_max] get a bonus (up to 1.0 + length_reward_bonus)
            - Sequences outside range get penalty or 0 based on punish_mode
        
        If length_reward_mode=False:
            - Returns 1.0 (no length-based reward/penalty)
        """
        if not self.reward_config.length_reward_mode:
            return 1.0
        
        L = len(seq)
        mn = self.reward_config.min_length
        mx = self.reward_config.max_length
        assert mn is not None and mx is not None, "min_length and max_length must be set if length_reward_mode is True"
        
        # Outside acceptable range: penalty
        if L < mn or L > mx:
            return float(self.reward_config.violation_penalty_factor) if self.reward_config.punish_mode else 0.5
        
        # Within acceptable range: base credit of 1.0
        base_score = 1.0
        
        # Check for ideal range bonus
        ideal_min = self.reward_config.ideal_min_length
        ideal_max = self.reward_config.ideal_max_length
        
        if ideal_min is not None and ideal_max is not None:
            # If within ideal range, give full bonus
            if ideal_min <= L <= ideal_max:
                return base_score + self.reward_config.length_reward_bonus
            
            # If between min-ideal_min or ideal_max-max, give partial bonus
            # based on distance from ideal range
            if L < ideal_min:
                # Between min and ideal_min
                distance_ratio = (L - mn) / (ideal_min - mn) if ideal_min > mn else 1.0
                bonus = distance_ratio * self.reward_config.length_reward_bonus
                return base_score + bonus
            else:
                # Between ideal_max and max
                distance_ratio = (mx - L) / (mx - ideal_max) if mx > ideal_max else 1.0
                bonus = distance_ratio * self.reward_config.length_reward_bonus
                return base_score + bonus
        
        return base_score

    def score(self, seq: str, source: str | None = None) -> Tuple[float, Dict[str, float]]:
        """
        Score a DNA sequence based on biological features.
        
        Args:
            seq: DNA sequence to score
            source: Optional identifier for logging
            
        Returns:
            Tuple of (final_score, component_scores_dict)
        """
        assert seq, "sequence cannot be empty"
        t0 = time.perf_counter()
        src = source if source else f"seq_{len(seq)}bp"
        
        # Annotate and score each component
        annotations = self.annotate(seq)
        ori = self.score_ori(seq, annotations)
        prom = self.score_promoter(seq, annotations)
        term = self.score_terminator(seq, annotations)
        mark = self.score_marker(seq, annotations)
        cds = self.score_cds(seq, annotations)
        length_factor = self.score_length(seq, annotations)
        
        # Raw counts (post-merge, using preprocessed annotations)
        feats = list(annotations)
        ori_count = sum(1 for x in feats if x.type and x.type.lower() == "ori")
        promoter_count = sum(1 for x in feats if x.type and x.type.lower() == "promoter")
        terminator_count = sum(1 for x in feats if x.type and x.type.lower() == "terminator")
        marker_count = sum(1 for x in feats if x.type and x.type.lower() == "marker")
        cds_count = sum(1 for x in feats if x.type and x.type.lower() == "cds")
        
        # Weighted sum
        results = [ori, prom, term, mark, cds]
        base = sum(w * r for w, r in zip(self.weights, results))
        
        # Apply length penalty as multiplier
        final = max(0.0, min(1.0, base * length_factor))
        
        # Apply repeat penalty (subtract per repeated region)
        repeat_regions = self._find_repeat_regions(seq)
        repeat_penalty = 0.0
        if repeat_regions:
            repeat_penalty = float(len(repeat_regions)) * float(self.reward_config.repeat_penalty_per_region)
            final = max(0.0, min(1.0, final - repeat_penalty))
        
        components: Dict[str, float] = {
            "ori": float(ori),
            "promoter": float(prom),
            "terminator": float(term),
            "marker": float(mark),
            "cds": float(cds),
            "length_factor": float(length_factor),
            "repeat_regions": float(len(repeat_regions)),
            "repeat_penalty": float(-repeat_penalty),
            # Raw counts for descriptive logging
            "ori_count": float(ori_count),
            "promoter_count": float(promoter_count),
            "terminator_count": float(terminator_count),
            "marker_count": float(marker_count),
            "cds_count": float(cds_count),
        }
        
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return float(final), components