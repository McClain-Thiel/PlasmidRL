from collections import Counter
import math

from vllm import LLM, SamplingParams
from typing import Optional, List, Dict, Any
from src.utils.training_utils import EvalRunner, EvaluationResult
from src.config import Config, EvalConfig
import pandas as pd
import os
import plasmidkit as pk
import re


class _Feat:
    """Simple feature container for annotation merging (same as Scorer)."""
    def __init__(self, type: str, id: str | None, start: int, end: int, strand: str | None, evidence: Any = None):
        self.type = type
        self.id = id
        self.start = int(start)
        self.end = int(end)
        self.strand = strand or "+"
        self.evidence = evidence or {}


class SequenceAnalyzer:
    """
    Analyzes plasmid sequences and extracts detailed annotation information.
    
    Similar to Scorer but returns detailed information instead of scores.
    Uses plasmidkit for annotation and extracts counts and IDs for each feature type.
    """
    
    def __init__(self, eval_config: EvalConfig):
        self.eval_config = eval_config
    
    def annotate(self, sequence: str) -> List[Any]:
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
    
    def _to_feat(self, x: Any) -> _Feat:
        """Convert annotation object to internal _Feat representation."""
        return _Feat(
            type=x.type.lower() if x.type else "",
            id=x.id if hasattr(x, "id") else None,
            start=int(x.start),
            end=int(x.end),
            strand=x.strand if hasattr(x, "strand") else "+",
            evidence=x.evidence if hasattr(x, "evidence") else {},
        )
    
    def _merge_group(self, feats: List[Any], threshold: float, *, respect_strand: bool) -> List[_Feat]:
        """Merge overlapping features of the same type based on overlap threshold."""
        if not feats:
            return []
        items = [self._to_feat(f) for f in feats]
        items.sort(key=lambda f: (f.strand, f.start, f.end))
        merged: List[_Feat] = []
        cur = items[0]
        for nxt in items[1:]:
            ovl = self._overlap_len(cur, nxt)
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
        
        Same logic as Scorer._preprocess_annotations.
        """
        feats = list(annotations)
        thr = float(self.eval_config.overlap_merge_threshold)
        type_key = lambda x: x.type.lower() if x.type else ""

        # Collect groups by type
        groups: Dict[str, List[Any]] = {}
        for f in feats:
            groups.setdefault(type_key(f), []).append(f)

        # Merge per group for relevant types
        merged_groups: Dict[str, List[_Feat]] = {}
        for t in ("rep_origin", "ori", "origin_of_replication", "promoter", "terminator", "marker", "cds"):
            if t in groups:
                respect = t not in ("rep_origin", "ori", "origin_of_replication", "marker")
                merged_groups[t] = self._merge_group(groups[t], thr, respect_strand=respect)

        # Suppress CDS if overlaps any non-CDS
        non_cds: List[_Feat] = []
        for t in ("rep_origin", "ori", "origin_of_replication", "promoter", "terminator", "marker"):
            non_cds.extend(merged_groups.get(t, []))

        filtered_cds: List[_Feat] = []
        for c in merged_groups.get("cds", []):
            if any(self._overlap_len(c, o) > 0 for o in non_cds):
                continue
            filtered_cds.append(c)
        merged_groups["cds"] = filtered_cds

        # Rebuild final list
        final: List[Any] = []
        merged_types = set(merged_groups.keys())
        for t, items in merged_groups.items():
            final.extend(items)
        for f in feats:
            t = type_key(f)
            if t not in merged_types:
                final.append(f)
        return final
    
    def analyze(self, sequence: str) -> Dict[str, Any]:
        """
        Analyze a sequence and extract detailed annotation information.
        
        Args:
            sequence: DNA sequence to analyze
            
        Returns:
            Dictionary with counts and IDs for each feature type
        """
        annotations = self.annotate(sequence)
        feats = list(annotations)
        type_key = lambda x: x.type.lower() if x.type else ""
        
        # Extract features by type (no filtering - report all IDs found)
        oris = [x for x in feats if type_key(x) in ("rep_origin", "ori", "origin_of_replication")]
        promoters = [x for x in feats if type_key(x) == "promoter"]
        terminators = [x for x in feats if type_key(x) == "terminator"]
        markers = [x for x in feats if type_key(x) == "marker"]
        cdss = [x for x in feats if type_key(x) == "cds"]
        
        # Extract IDs (handle merged IDs separated by |)
        def extract_ids(features: List[Any]) -> List[str]:
            ids = []
            for f in features:
                if hasattr(f, "id") and f.id:
                    # Split merged IDs (separated by |)
                    ids.extend([id.strip() for id in str(f.id).split("|") if id.strip()])
            return ids
        
        return {
            "ori_count": len(oris),
            "ori_ids": ",".join(extract_ids(oris)) if oris else "",
            "promoter_count": len(promoters),
            "promoter_ids": ",".join(extract_ids(promoters)) if promoters else "",
            "terminator_count": len(terminators),
            "terminator_ids": ",".join(extract_ids(terminators)) if terminators else "",
            "marker_count": len(markers),
            "marker_ids": ",".join(extract_ids(markers)) if markers else "",
            "cds_count": len(cdss),
            "cds_ids": ",".join(extract_ids(cdss)) if cdss else "",
        }


class Evaluator(EvalRunner):
    """
    Evaluator that generates rollouts from a checkpoint and analyzes them.
    
    Loads prompts from CSV, generates samples using vLLM, analyzes each sequence
    with plasmidkit, and returns a DataFrame with detailed annotation information.
    """
    
    def __init__(self, config: EvalConfig):
        """
        Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.llm: Optional[LLM] = None
        self.base_config = Config()  # For default prompts
        self.analyzer = SequenceAnalyzer(config)
        
    def run_with_trainer(self, trainer: Any, wandb_run: Optional[Any] = None) -> pd.DataFrame:
        """
        Run evaluation using the trainer's model directly (already loaded on GPU).
        
        Args:
            trainer: Trainer instance with vLLM model already loaded
            wandb_run: Optional wandb run object for logging
            
        Returns:
            DataFrame with detailed annotation information for each sequence
        """
        # Load prompts
        prompts = self._load_prompts()
        
        if not prompts:
            print("[Evaluator] Warning: No prompts loaded, returning empty DataFrame")
            return pd.DataFrame()
        
        # Use trainer's vLLM instance directly
        llm = trainer.llm if hasattr(trainer, 'llm') else None
        if llm is None:
            print("[Evaluator] Warning: Trainer does not have 'llm' attribute, cannot use in-memory model")
            return pd.DataFrame()
        
        print(f"[Evaluator] Using trainer's vLLM instance directly (no model reload needed)")
        self.llm = llm
        
        # Get sampling parameters
        sampling_params = self.config.sampling_params
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.8,
                top_p=0.95,
                top_k=0,
            )
        
        # Expand prompts for multiple samples per prompt
        expanded_prompts = []
        prompt_indices = []
        for i, prompt in enumerate(prompts):
            for _ in range(self.config.num_samples_per_prompt):
                expanded_prompts.append(prompt)
                prompt_indices.append(i)
        
        print(f"[Evaluator] Generating {len(expanded_prompts)} samples from {len(prompts)} prompts")
        
        # Generate rollouts
        outputs = self.llm.generate(expanded_prompts, sampling_params)
        
        # Process results and analyze each sequence
        records = []
        for idx, output in enumerate(outputs):
            prompt = output.prompt
            completion = output.outputs[0].text.replace(" ", "")
            full = prompt + completion
            
            # Clean sequence for analysis (remove non-DNA characters)
            cleaned_full = re.sub(r'[^ATCG]', '', full.upper())
            
            # Analyze sequence
            try:
                analysis = self.analyzer.analyze(cleaned_full)
            except Exception as e:
                print(f"[Evaluator] Warning: Failed to analyze sequence {idx}: {e}")
                analysis = {
                    "ori_count": 0,
                    "ori_ids": "",
                    "promoter_count": 0,
                    "promoter_ids": "",
                    "terminator_count": 0,
                    "terminator_ids": "",
                    "marker_count": 0,
                    "marker_ids": "",
                    "cds_count": 0,
                    "cds_ids": "",
                }
            
            records.append({
                "prompt": prompt,
                "response": completion,
                "full": full,
                "length": len(cleaned_full),
                "completion_length": len(completion),
                "full_length": len(full),
                **analysis,
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        print(f"[Evaluator] Generated and analyzed {len(df)} rollouts")
        
        metrics = self._maybe_compute_self_bleu_metrics(sampling_params)
        return EvaluationResult(dataframe=df, metrics=metrics)
    
    def _load_prompts(self) -> List[str]:
        """
        Load prompts from CSV/parquet file.
        
        Returns:
            List of prompt strings
        """
        if not self.config.prompts_path:
            print("[Evaluator] Warning: No prompts_path specified, using default prompts")
            return [self._get_default_prompts()]
        
        prompts_path = self.config.prompts_path
        
        # Check if file exists
        if not os.path.exists(prompts_path):
            print(f"[Evaluator] Warning: Prompts file not found: {prompts_path}")
            return [self._get_default_prompts()]
        
        try:
            # Load based on file extension
            if prompts_path.endswith('.parquet'):
                df = pd.read_parquet(prompts_path)
            elif prompts_path.endswith('.csv'):
                df = pd.read_csv(prompts_path)
            else:
                print(f"[Evaluator] Warning: Unsupported file format: {prompts_path}")
                return [self._get_default_prompts()]
            
            # Extract prompts column
            if self.config.prompts_column not in df.columns:
                print(f"[Evaluator] Warning: Column '{self.config.prompts_column}' not found in prompts file")
                print(f"[Evaluator] Available columns: {list(df.columns)}")
                return [self._get_default_prompts()]
            
            prompts = df[self.config.prompts_column].dropna().tolist()
            prompts = [str(p).strip() for p in prompts if str(p).strip()]
            
            print(f"[Evaluator] Loaded {len(prompts)} prompts from {prompts_path}")
            return prompts
            
        except Exception as e:
            print(f"[Evaluator] Error loading prompts: {e}")
            return [self._get_default_prompts()]
    
    def _get_default_prompts(self) -> str:
        """Get default prompt if prompts file is not available."""
        # Use the default query from config (GFP cassette)
        return self.base_config.default_query
    
    def _initialize_model(self, checkpoint_path: str) -> None:
        """
        Initialize vLLM model from checkpoint path.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        # Check if model is already initialized for this checkpoint
        if self.llm is not None:
            # For now, always reinitialize - could optimize later
            pass
        
        print(f"[Evaluator] Loading model from checkpoint: {checkpoint_path}")
        
        try:
            # Initialize vLLM with checkpoint path
            # vLLM can load from local checkpoint directories
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            self.llm = LLM(model=checkpoint_path, **model_kwargs)
            print(f"[Evaluator] Model loaded successfully")
            
        except Exception as e:
            print(f"[Evaluator] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _maybe_compute_self_bleu_metrics(self, sampling_params: SamplingParams) -> Dict[str, float]:
        """Compute self-BLEU on repeated prompt rollouts if configured."""
        metrics: Dict[str, float] = {}

        prompt = (self.config.self_bleu_prompt or "").strip()
        if not prompt:
            return metrics

        num_samples = max(1, self.config.self_bleu_sample_count)
        score, samples_used = self._evaluate_self_bleu_for_prompt(prompt, sampling_params, num_samples)
        metrics["eval/self_bleu"] = score
        metrics["eval/self_bleu_samples"] = float(samples_used)
        return metrics

    def _evaluate_self_bleu_for_prompt(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> tuple[float, int]:
        """Generate rollouts from a single prompt and compute self-BLEU."""
        outputs = self.llm.generate([prompt] * num_samples, sampling_params)
        sequences: List[str] = []
        for output in outputs:
            completion = output.outputs[0].text
            cleaned = re.sub(r'[^ATCG]', '', completion.upper())
            if cleaned:
                sequences.append(cleaned)

        score = self._compute_self_bleu(sequences, self.config.self_bleu_max_n)
        return score, len(sequences)

    def _compute_self_bleu(self, sequences: List[str], max_n: int) -> float:
        """Simple self-BLEU implementation tailored for DNA tokens."""
        if len(sequences) < 2:
            return 0.0

        total_score = 0.0
        for idx, candidate in enumerate(sequences):
            references = [seq for i, seq in enumerate(sequences) if i != idx]
            total_score += self._calculate_bleu(candidate, references, max_n)
        return total_score / len(sequences)

    def _calculate_bleu(self, candidate: str, references: List[str], max_n: int) -> float:
        """Compute BLEU score for a single candidate against multiple references."""
        candidate_tokens = list(candidate)
        if not candidate_tokens or not references:
            return 0.0

        log_precision_sum = 0.0
        valid_precisions = 0
        for n in range(1, max_n + 1):
            cand_ngrams = self._ngram_counts(candidate_tokens, n)
            total_cand_ngrams = sum(cand_ngrams.values())
            if total_cand_ngrams == 0:
                continue

            ref_ngrams = self._max_reference_ngrams(references, n)
            overlap = sum(
                min(count, ref_ngrams.get(ngram, 0)) for ngram, count in cand_ngrams.items()
            )
            precision = (overlap + 1) / (total_cand_ngrams + 1)
            log_precision_sum += math.log(precision)
            valid_precisions += 1

        if valid_precisions == 0:
            return 0.0

        geometric_mean = math.exp(log_precision_sum / valid_precisions)

        candidate_len = len(candidate_tokens)
        if candidate_len == 0:
            return 0.0

        closest_ref_len = self._closest_reference_length(candidate_len, references)
        bp = 1.0
        if candidate_len < closest_ref_len:
            bp = math.exp(1 - closest_ref_len / candidate_len)
        return bp * geometric_mean

    @staticmethod
    def _ngram_counts(tokens: List[str], n: int) -> Counter:
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    @staticmethod
    def _max_reference_ngrams(references: List[str], n: int) -> Counter:
        max_counts: Counter = Counter()
        for reference in references:
            tokens = list(reference)
            ref_counts = Counter(
                tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
            )
            for ngram, count in ref_counts.items():
                max_counts[ngram] = max(max_counts.get(ngram, 0), count)
        return max_counts

    @staticmethod
    def _closest_reference_length(candidate_len: int, references: List[str]) -> int:
        best_len = len(references[0]) if references else 0
        best_diff = abs(best_len - candidate_len)
        for reference in references:
            ref_len = len(reference)
            diff = abs(ref_len - candidate_len)
            if diff < best_diff or (diff == best_diff and ref_len < best_len):
                best_len = ref_len
                best_diff = diff
        return best_len