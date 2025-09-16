from torchrl.envs import Transform
import httpx
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import CompositeSpec, Unbounded

from ..config import config

class RewardTransform(Transform):
    """Assign rewards by calling external scoring endpoints in parallel."""

    def __init__(
        self,
        rewards_server_url: Optional[str] = None,
        timeout_s: float = 60.0,
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
                try:
                    preview = (resp.text or "")[:300]
                except Exception:
                    preview = "<unreadable>"
                print(f"[{name}] {path} -> {resp.status_code} | {preview}")
            return {"status": ok, "name": name, "reponse": resp.json()}
        except Exception as e:
            print(f"[{name}] {path} call failed: {e}")
            return {"status": False, "name": name, "reponse": {}}

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
        # 1) extract text(s)
        llm_texts: list[str]
        if "text" in td.keys(True):
            t = td["text"]
            llm_texts = [t if isinstance(t, str) else getattr(t, "response", str(t))]
        elif "query" in td.keys(True):
                q = td.get("query")
                # if query is a list (batch), use it as-is; else wrap
                llm_texts = list(q) if hasattr(q, "__iter__") and not isinstance(q, (str, bytes)) else [q]
        else:
            td["reward"] = torch.zeros(td.batch_size + (1,), dtype=torch.float32)
            return td
    
        # 2) extract per-example overrides (align shape)
        overrides = td.get("reward_params", None)
        if overrides is None or isinstance(overrides, dict):
            overrides_list = [overrides] * len(llm_texts)
        else:
            overrides_list = list(overrides)
            if len(overrides_list) != len(llm_texts):
                # safe fallback
                overrides_list = [None] * len(llm_texts)
    
        # 3) call endpoints per example (parallel using ThreadPoolExecutor)
        rewards: list[float] = []
        for text, ov in zip(llm_texts, overrides_list):
            # hit the three endpoints for THIS example in parallel
            with ThreadPoolExecutor(max_workers=len(self._endpoints)) as executor:
                future_to_cfg = {
                    executor.submit(self._post_text, cfg["path"], cfg.get("params", {}), text, cfg["name"]): cfg
                    for cfg in self._endpoints
                }
                results = []
                for future in as_completed(future_to_cfg):
                    result = future.result()
                    results.append(result)
            
            # combine with per-example params
            r = self.combine_rewards(results, overrides=ov)
            rewards.append(float(r))
    
        # 4) write back (vector or scalar)
        out = torch.as_tensor(rewards, dtype=torch.float32)
        # reshape to td batch
        if out.ndim == 1 and out.numel() == td.numel():
            out = out.view(td.batch_size + (1,))
        else:
            out = out.mean().view(td.batch_size + (1,))  # conservative fallback
        td["reward"] = out
        return td


    def transform_reward_spec(self, reward_spec: CompositeSpec) -> CompositeSpec:
        reward_spec["reward"] = Unbounded(shape=reward_spec.shape + (1,), dtype=torch.float32)
        return reward_spec

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    def __del__(self):
        try:
            if not self.client.is_closed:
                self.client.close()
        except Exception:
            pass