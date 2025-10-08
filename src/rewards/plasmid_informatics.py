import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

import httpx
import requests
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack, NonTensorData
from torchrl.data.tensor_specs import CompositeSpec, Unbounded
from torchrl.envs import Transform

from ..config import config

logger = logging.getLogger(__name__)

class RewardTransform(Transform):
    """Assign rewards by calling external scoring endpoints in parallel."""

    def __init__(
        self,
        rewards_server_url: Optional[str] = None,
        timeout_s: float = 60.0,
        max_workers: Optional[int] = None,
        log_timings: bool = False,
    ):
        super().__init__(in_keys=[], out_keys=["reward"])

        self.rewards_server_url = (rewards_server_url or config.informatics_server_url).rstrip("/")
        self.timeout_s = timeout_s
        self.client = httpx.Client(
            base_url=self.rewards_server_url,
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=True,
        )
        #defaults for evaluating plasmif 
        self.require = {"ori": None, "amr": None, "mcs": True, "promoter": None}
        self.weights = {"ori": 0.30, "amr": 0.30, "mcs": 0.20, "promoter": 0.20}
        self.gc = {"target": 0.55, "weight": 0.05, "tolerance": 0.10}
        # length bonus configuration: up to 0.25 at or below target length, decays slightly for longer sequences
        self.length = {"target": 1000, "weight": 0.25, "decay_per_bp": 1e-4, "penalize_shorter": False}
        self._test_connection()

        # Exact paths & params per your curl commands
        self._endpoints: List[Dict] = [
            {"name": "amrfinder", "path": "/amrfinder/text", "params": {"is_protein": "false", "format": "json"}},
            {"name": "prodigal",  "path": "/prodigal/text",  "params": {"mode": "auto",   "format": "json"}},
            {"name": "plannotate","path": "/plannotate/fast","params": {}},
        ]

        # Plain text in, accept json or text back
        self._headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "application/json, text/plain; q=0.9, */*; q=0.1",
        }
        self._max_workers = max_workers or max(1, len(self._endpoints))
        self._log_timings = log_timings
        self._sample_executor = ThreadPoolExecutor(max_workers=self._max_workers)

    def _test_connection(self):
        try:
            r = requests.get(self.rewards_server_url + "/health", timeout=5)
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to rewards server: {e}")

    def _post_text(self, path: str, params: Dict, text: str, name: str) -> Dict:
        try:
            resp = self.client.post(
                path, params=params, content=text.encode("utf-8"), headers=self._headers
            )
            ok = (resp.status_code == 200)
            if not ok:
                snippet = text[:120]
                if len(text) > 120:
                    snippet += "…"
                try:
                    preview = (resp.text or "")[:300]
                except Exception:
                    preview = "<unreadable>"
                logger.error(
                    "[%s] %s -> %s | %s | payload_snippet=%s",
                    name,
                    path,
                    resp.status_code,
                    preview,
                    snippet,
                )
            try:
                response_data = resp.json()
            except Exception as json_err:
                if ok:
                    logger.warning("[%s] %s JSON parse error: %s", name, path, json_err)
                response_data = {}
            return {"status": ok, "name": name, "reponse": response_data}
        except Exception as e:
            logger.error("[%s] %s call failed: %s", name, path, e)
            return {"status": False, "name": name, "reponse": {}}

    def _score_single(self, idx: int, text: str, overrides: dict | None) -> tuple[int, float]:
        if not text:
            return idx, 0.0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Scoring sample idx=%s length=%s overrides=%s",
                idx,
                len(text),
                bool(overrides),
            )

        with ThreadPoolExecutor(max_workers=len(self._endpoints)) as executor:
            future_to_cfg = {
                executor.submit(
                    self._post_text,
                    cfg["path"],
                    cfg.get("params", {}),
                    text,
                    cfg["name"],
                ): cfg
                for cfg in self._endpoints
            }
            results = []
            for future in as_completed(future_to_cfg):
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": False, "name": "<exception>", "reponse": {}, "error": str(e)}
                results.append(result)

        try:
            reward = self.combine_rewards(results, overrides=overrides)
        except Exception as e:
            logger.warning("Reward calculation failed for text: %s", e)
            reward = 0.0

        # Apply length-based component (string length, not tokens)
        # Maximum of 0.25 when sequence length is at or below target; gently decays for longer sequences.
        try:
            lconf = dict(self.length)
            if overrides and isinstance(overrides, dict) and "length" in overrides:
                lconf.update(overrides.get("length") or {})

            target_len = max(0, int(lconf.get("target", 1000)))
            weight = float(lconf.get("weight", 0.25))
            decay_per_bp = float(lconf.get("decay_per_bp", 1e-4))
            penalize_shorter = bool(lconf.get("penalize_shorter", False))

            seq_len = len(text)
            if seq_len <= target_len and not penalize_shorter:
                length_bonus = weight
            else:
                delta = max(0, seq_len - target_len) if not penalize_shorter else abs(seq_len - target_len)
                factor = max(0.0, 1.0 - decay_per_bp * float(delta))
                length_bonus = weight * factor
            reward += float(length_bonus)
        except Exception:
            # If anything goes wrong with length bonus, skip it silently
            pass

        if logger.isEnabledFor(logging.DEBUG):
            status_map = {
                str(res.get("name", "?")): bool(res.get("status", False))
                for res in results
            }
            logger.debug(
                "Sample idx=%s component_status=%s reward=%.4f",
                idx,
                status_map,
                reward,
            )

        return idx, reward

    def combine_rewards(self, info_dicts: List[Dict], overrides: dict | None = None) -> float:
        """
        Combine amrfinder, prodigal, and plannotate responses into a single reward.
        Scoring:
          - Presence of ORI, AMR gene, MCS, promoter (weighted), each multiplied by percent identity if available.
          - Small GC proximity bonus (around 55%) as a tie-breaker.
        Extensible via self.require / self.weights / self.gc.
        """
        require = dict(self.require)
        weights = dict(self.weights)
        gcconf  = dict(self.gc)
    
        # Merge per-example overrides if provided
        if overrides:
            if "require" in overrides: require.update(overrides["require"] or {})
            if "weights" in overrides: weights.update(overrides["weights"] or {})
            if "gc" in overrides:      gcconf.update(overrides["gc"] or {})

        # filter out failed calls (False) and normalize shape
        records = [r for r in info_dicts if isinstance(r, dict) and r.get("status") is not False]
        by_name = {r.get("name"): (r.get("reponse") or {}) for r in records}
    
        amr = by_name.get("amrfinder") or {}
        prodigal = by_name.get("prodigal") or {}
        plannotate = by_name.get("plannotate") or []
    
        # helpers
        def _lc(s): return str(s or "").lower()
        def _contains_any(text: str, needles) -> bool:
            t = _lc(text)
            return any(_lc(n) in t for n in (needles or []))
        def _best_pident(entries) -> float:
            """Return best percent identity (0..1) among provided plannotate entries with 'pident'."""
            best = 0.0
            for e in entries:
                try:
                    p = float(e.get("pident", 100.0)) / 100.0
                except Exception:
                    p = 1.0
                best = max(best, max(0.0, min(1.0, p)))
            return best if best > 0 else 1.0  # default to 1.0 if none provided
    
        # ---- ORI / MCS / Promoter from plannotate ----
        ori_present, mcs_present, prom_present = False, False, False
        ori_pident, mcs_pident, prom_pident = 1.0, 1.0, 1.0
    
        if isinstance(plannotate, list):
            # buckets
            ori_entries, mcs_entries, prom_entries = [], [], []
    
            for feat in plannotate:
                name = str(feat.get("Feature") or "")
                desc = str(feat.get("Description") or "")
                typ  = str(feat.get("Type") or "")
                text = f"{name} {desc} {typ}"
    
                # ORI
                if self.require["ori"] is None:
                    if typ.lower() == "rep_origin" or _contains_any(text, ["ori", "colE1", "pmb1", "pbr322", "puc"]):
                        ori_entries.append(feat)
                else:
                    if _contains_any(text, self.require["ori"]):
                        ori_entries.append(feat)
    
                # MCS
                if self.require["mcs"] is True:
                    if _contains_any(text, ["mcs", "multiple cloning site"]):
                        mcs_entries.append(feat)
                elif isinstance(self.require["mcs"], list):
                    if _contains_any(text, self.require["mcs"]):
                        mcs_entries.append(feat)
    
                # Promoter
                if self.require["promoter"] is None:
                    if typ.lower() == "promoter" or _contains_any(text, ["promoter"]):
                        prom_entries.append(feat)
                else:
                    if _contains_any(text, self.require["promoter"]):
                        prom_entries.append(feat)

            if ori_entries:
                ori_present = True
                ori_pident = _best_pident(ori_entries)
            if mcs_entries:
                mcs_present = True
                mcs_pident = _best_pident(mcs_entries)
            if prom_entries:
                prom_present = True
                prom_pident = _best_pident(prom_entries)
    
        # ---- AMR from amrfinder ----
        amr_present, amr_pident = False, 1.0
        if isinstance(amr, dict):
            hits = amr.get("genes", []) or []
            candidate_hits = []
            for g in hits:
                cls = str(g.get("class") or "")
                sym = str(g.get("element_symbol") or "")
                nm  = str(g.get("element_name") or "")
                hay = f"{cls} {sym} {nm}"
                if self.require["amr"] is None or _contains_any(hay, self.require["amr"]):
                    candidate_hits.append(g)
            if candidate_hits:
                amr_present = True
                # pick best identity among matching hits
                best = 1.0
                for g in candidate_hits:
                    try:
                        pid = float(g.get("percent_identity_to_reference", 100.0)) / 100.0
                    except Exception:
                        pid = 1.0
                    best = max(best, max(0.0, min(1.0, pid)))
                amr_pident = best

        # ---- main score (presence × weight × identity) ----
        w = self.weights
        main = 0.0
        if ori_present:   main += w.get("ori", 0.0) * float(ori_pident)
        if amr_present:   main += w.get("amr", 0.0) * float(amr_pident)
        if mcs_present:   main += w.get("mcs", 0.0) * float(mcs_pident)
        if prom_present:  main += w.get("promoter", 0.0) * float(prom_pident)
        main = min(main, 1.0)  # keep things tidy
    
        # ---- GC tie-breaker from prodigal ----
        gc_bonus = 0.0
        if isinstance(prodigal, dict):
            meta = prodigal.get("metadata", {}) or {}
            gc_raw = meta.get("model_gc_cont") or meta.get("gc_cont")
            if gc_raw is not None:
                try:
                    s = str(gc_raw).strip().replace("%", "")
                    v = float(s)
                    gc = v / 100.0 if "%" in str(gc_raw) or v > 1.0 else v
                    target = float(self.gc["target"])
                    tol = max(1e-6, float(self.gc["tolerance"]))
                    dist = abs(gc - target)
                    norm = max(0.0, 1.0 - (dist / tol))   # 1 at target, 0 at >= tolerance
                    gc_bonus = float(self.gc["weight"]) * norm
                except Exception:
                    gc_bonus = 0.0
    
        total = main + gc_bonus
        return float(total)


    def _call(self, td: TensorDict) -> TensorDict:
        def _extract_strings(obj) -> list[str]:
            if obj is None:
                return []
            if isinstance(obj, str):
                return [obj]
            if isinstance(obj, NonTensorData):
                return _extract_strings(obj.data)
            if isinstance(obj, NonTensorStack):
                values: list[str] = []
                for item in obj:
                    values.extend(_extract_strings(item))
                return values
            if isinstance(obj, (list, tuple)):
                values: list[str] = []
                for item in obj:
                    values.extend(_extract_strings(item))
                return values
            return [str(obj)]

        llm_texts: list[str] = []
        keys = td.keys(True)
        if ("text", "response") in keys:
            llm_texts = _extract_strings(td.get(("text", "response")))
        elif ("text", "full") in keys:
            llm_texts = _extract_strings(td.get(("text", "full")))
        elif "text" in keys:
            llm_texts = _extract_strings(td.get("text"))
        elif "query" in keys:
            llm_texts = _extract_strings(td.get("query"))

        if not llm_texts:
            zero = torch.zeros(td.batch_size + (1,), dtype=torch.float32)
            if td.device is not None:
                zero = zero.to(td.device)
            td["reward"] = zero
            logger.debug("RewardTransform received empty text; assigning zero reward")
            return td

        # Align to batch size if we only got a single flattened string
        expected = td.numel()
        if len(llm_texts) != expected:
            if len(llm_texts) == 1:
                llm_texts = llm_texts * expected
            else:
                llm_texts = llm_texts[:expected]

        def _clean_sequence(raw: str) -> str:
            text = str(raw)
            # split FastChat-style or OpenAI-style special tokens
            tokens = text.split("<|im_start|>")
            segments = []
            for tok in tokens:
                if not tok:
                    continue
                if "|>" in tok:
                    _, remainder = tok.split("|>", 1)
                else:
                    remainder = tok
                segments.append(remainder)
            text = "".join(segments)
            # Remove explicit end tokens
            text = text.replace("<|im_end|>", "")
            # Keep only nucleotide characters and standard ambiguity codes
            allowed = set("ACGTURYKMSWBDHVNXacgturykmswbdhvnx")
            cleaned = "".join(ch for ch in text if ch in allowed)
            cleaned = cleaned.upper()
            # Ensure we don't return an empty sequence
            return cleaned

        llm_texts = [_clean_sequence(s) for s in llm_texts]

        # 2) extract per-example overrides (align shape)
        overrides = td.get("reward_params", None)
        if overrides is None or isinstance(overrides, dict):
            overrides_list = [overrides] * len(llm_texts)
        else:
            overrides_list = list(overrides)
            if len(overrides_list) != len(llm_texts):
                overrides_list = [None] * len(llm_texts)
    
        start_time = time.perf_counter() if self._log_timings else None

        futures = []
        for idx, (text, ov) in enumerate(zip(llm_texts, overrides_list)):
            future = self._sample_executor.submit(self._score_single, idx, text, ov)
            futures.append(future)
        rewards: list[float] = [0.0] * len(futures)
        failures = 0
        for future in futures:
            try:
                idx, reward_val = future.result()
                rewards[idx] = float(reward_val)
            except Exception as exc:
                failures += 1
                logger.warning("Reward computation task failed: %s", exc)

        successes = len(rewards) - failures
        if logger.isEnabledFor(logging.DEBUG):
            preview = ", ".join(f"{val:.3f}" for val in rewards[:3])
            logger.debug(
                "RewardTransform batch complete: size=%d successes=%d failures=%d preview=[%s]",
                len(rewards),
                successes,
                failures,
                preview,
            )

        if self._log_timings and start_time is not None:
            elapsed = time.perf_counter() - start_time
            logger.info(
                "[RewardTransform] processed %d sequences in %.2fs (failures=%d)",
                len(rewards),
                elapsed,
                failures,
            )

        # 4) write back (vector or scalar)
        out = torch.as_tensor(rewards, dtype=torch.float32)
        # reshape to td batch
        if out.ndim == 1 and out.numel() == td.numel():
            out = out.view(td.batch_size + (1,))
        else:
            out = out.mean().view(td.batch_size + (1,))  # conservative fallback
        if td.device is not None:
            out = out.to(td.device)
        td["reward"] = out
        return td


    def transform_reward_spec(self, reward_spec: CompositeSpec) -> CompositeSpec:
        device = getattr(reward_spec, "device", torch.device("cpu"))
        reward_spec["reward"] = Unbounded(
            shape=reward_spec.shape + (1,),
            dtype=torch.float32,
            device=device,
        )
        return reward_spec

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
        try:
            self._sample_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def __del__(self):
        try:
            if not self.client.is_closed:
                self.client.close()
        except Exception:
            pass
        try:
            self._sample_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
