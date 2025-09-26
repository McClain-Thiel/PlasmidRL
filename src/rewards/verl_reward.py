import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Configuration
INFORMATICS_SERVER_URL = os.getenv("INFORMATICS_SERVER_URL", "http://server:8080")
TIMEOUT_SECONDS = 60.0
MAX_WORKERS = 3

# Default scoring weights and requirements
DEFAULT_WEIGHTS = {"ori": 0.30, "amr": 0.30, "mcs": 0.20, "promoter": 0.20}
DEFAULT_GC = {"target": 0.55, "weight": 0.05, "tolerance": 0.10}

# API endpoints configuration
ENDPOINTS = [
    {"name": "amrfinder", "path": "/amrfinder/text", "params": {"is_protein": "false", "format": "json"}},
    {"name": "prodigal", "path": "/prodigal/text", "params": {"mode": "auto", "format": "json"}},
    {"name": "plannotate", "path": "/plannotate/fast", "params": {}},
]

HEADERS = {
    "Content-Type": "text/plain; charset=utf-8",
    "Accept": "application/json, text/plain; q=0.9, */*; q=0.1",
}


def _post_to_endpoint(client: httpx.Client, endpoint: Dict, plasmid_text: str) -> Dict:
    """Make a POST request to a single endpoint."""
    try:
        resp = client.post(
            endpoint["path"],
            params=endpoint.get("params", {}),
            content=plasmid_text.encode("utf-8"),
            headers=HEADERS
        )
        
        success = (resp.status_code == 200)
        if not success:
            logger.warning(
                "[%s] Request failed with status %s: %s",
                endpoint["name"], resp.status_code, resp.text[:200]
            )
        
        try:
            response_data = resp.json()
        except Exception:
            if success:
                logger.warning("[%s] Failed to parse JSON response", endpoint["name"])
            response_data = {}
            
        return {
            "status": success,
            "name": endpoint["name"],
            "response": response_data
        }
        
    except Exception as e:
        logger.error("[%s] Request failed: %s", endpoint["name"], e)
        return {
            "status": False,
            "name": endpoint["name"],
            "response": {}
        }


def _make_parallel_requests(plasmid_text: str) -> List[Dict]:
    """Make parallel requests to all endpoints."""
    if not plasmid_text.strip():
        return []
    
    results = []
    
    with httpx.Client(
        base_url=INFORMATICS_SERVER_URL,
        timeout=httpx.Timeout(TIMEOUT_SECONDS),
        follow_redirects=True
    ) as client:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all requests
            future_to_endpoint = {
                executor.submit(_post_to_endpoint, client, endpoint, plasmid_text): endpoint
                for endpoint in ENDPOINTS
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_endpoint):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    endpoint = future_to_endpoint[future]
                    logger.error("[%s] Future failed: %s", endpoint["name"], e)
                    results.append({
                        "status": False,
                        "name": endpoint["name"],
                        "response": {}
                    })
    
    return results


def _contains_any(text: str, needles: Optional[List[str]]) -> bool:
    """Check if text contains any of the needle strings (case-insensitive)."""
    if not needles:
        return False
    text_lower = str(text or "").lower()
    return any(str(needle or "").lower() in text_lower for needle in needles)


def _best_percent_identity(entries: List[Dict]) -> float:
    """Extract the best percent identity from a list of entries."""
    best = 0.0
    for entry in entries:
        try:
            pident = float(entry.get("pident", 100.0)) / 100.0
            best = max(best, max(0.0, min(1.0, pident)))
        except (ValueError, TypeError):
            continue
    return best if best > 0 else 1.0


def _score_plannotate_features(plannotate_data: List[Dict]) -> tuple[bool, bool, bool, float, float, float]:
    """Score plannotate features for ORI, MCS, and promoter presence."""
    ori_present = mcs_present = prom_present = False
    ori_pident = mcs_pident = prom_pident = 1.0
    
    if not isinstance(plannotate_data, list):
        return ori_present, mcs_present, prom_present, ori_pident, mcs_pident, prom_pident
    
    ori_entries, mcs_entries, prom_entries = [], [], []
    
    for feat in plannotate_data:
        if not isinstance(feat, dict):
            continue
            
        name = str(feat.get("Feature", ""))
        desc = str(feat.get("Description", ""))
        typ = str(feat.get("Type", ""))
        text = f"{name} {desc} {typ}"
        
        # Check for ORI
        if typ.lower() == "rep_origin" or _contains_any(text, ["ori", "colE1", "pmb1", "pbr322", "puc"]):
            ori_entries.append(feat)
            
        # Check for MCS
        if _contains_any(text, ["mcs", "multiple cloning site"]):
            mcs_entries.append(feat)
            
        # Check for promoter
        if typ.lower() == "promoter" or _contains_any(text, ["promoter"]):
            prom_entries.append(feat)
    
    if ori_entries:
        ori_present = True
        ori_pident = _best_percent_identity(ori_entries)
    if mcs_entries:
        mcs_present = True
        mcs_pident = _best_percent_identity(mcs_entries)
    if prom_entries:
        prom_present = True
        prom_pident = _best_percent_identity(prom_entries)
    
    return ori_present, mcs_present, prom_present, ori_pident, mcs_pident, prom_pident


def _score_amr_genes(amr_data: Dict) -> tuple[bool, float]:
    """Score AMR genes from amrfinder data."""
    amr_present = False
    amr_pident = 1.0
    
    if not isinstance(amr_data, dict):
        return amr_present, amr_pident
    
    genes = amr_data.get("genes", [])
    if not genes:
        return amr_present, amr_pident
    
    amr_present = True
    best_identity = 1.0
    
    for gene in genes:
        if not isinstance(gene, dict):
            continue
        try:
            pident = float(gene.get("percent_identity_to_reference", 100.0)) / 100.0
            best_identity = max(best_identity, max(0.0, min(1.0, pident)))
        except (ValueError, TypeError):
            continue
    
    amr_pident = best_identity
    return amr_present, amr_pident


def _calculate_gc_bonus(prodigal_data: Dict) -> float:
    """Calculate GC content bonus from prodigal data."""
    if not isinstance(prodigal_data, dict):
        return 0.0
    
    metadata = prodigal_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0.0
    
    gc_raw = metadata.get("model_gc_cont") or metadata.get("gc_cont")
    if gc_raw is None:
        return 0.0
    
    try:
        gc_str = str(gc_raw).strip().replace("%", "")
        gc_value = float(gc_str)
        gc_percent = gc_value / 100.0 if "%" in str(gc_raw) or gc_value > 1.0 else gc_value
        
        target = DEFAULT_GC["target"]
        tolerance = DEFAULT_GC["tolerance"]
        weight = DEFAULT_GC["weight"]
        
        distance = abs(gc_percent - target)
        normalized = max(0.0, 1.0 - (distance / tolerance))
        
        return weight * normalized
        
    except (ValueError, TypeError):
        return 0.0


def _combine_results(api_results: List[Dict]) -> float:
    """Combine API results into a single reward score."""
    # Organize results by endpoint name
    by_name = {}
    successful_endpoints = []
    failed_endpoints = []
    
    for result in api_results:
        endpoint_name = result.get("name", "unknown")
        if result.get("status", False):
            by_name[endpoint_name] = result.get("response", {})
            successful_endpoints.append(endpoint_name)
        else:
            failed_endpoints.append(endpoint_name)
    
    logger.info("API call results: successful=%s, failed=%s", successful_endpoints, failed_endpoints)
    
    # Extract data from each endpoint
    plannotate_data = by_name.get("plannotate", [])
    amr_data = by_name.get("amrfinder", {})
    prodigal_data = by_name.get("prodigal", {})
    
    # Score each component
    ori_present, mcs_present, prom_present, ori_pident, mcs_pident, prom_pident = \
        _score_plannotate_features(plannotate_data)
    
    amr_present, amr_pident = _score_amr_genes(amr_data)
    
    # Calculate individual component scores
    weights = DEFAULT_WEIGHTS
    ori_score = weights["ori"] * ori_pident if ori_present else 0.0
    amr_score = weights["amr"] * amr_pident if amr_present else 0.0
    mcs_score = weights["mcs"] * mcs_pident if mcs_present else 0.0
    prom_score = weights["promoter"] * prom_pident if prom_present else 0.0
    
    # Main score (sum of components)
    main_score = ori_score + amr_score + mcs_score + prom_score
    main_score = min(main_score, 1.0)  # Cap at 1.0
    
    # Add GC content bonus
    gc_bonus = _calculate_gc_bonus(prodigal_data)
    
    total_score = main_score + gc_bonus
    
    # Log detailed component breakdown
    logger.info("=== REWARD COMPONENT BREAKDOWN ===")
    logger.info("ORI (origin):     present=%s, identity=%.2f%%, score=%.4f (weight=%.2f)", 
                ori_present, ori_pident * 100, ori_score, weights["ori"])
    logger.info("AMR (resistance): present=%s, identity=%.2f%%, score=%.4f (weight=%.2f)", 
                amr_present, amr_pident * 100, amr_score, weights["amr"])
    logger.info("MCS (cloning):    present=%s, identity=%.2f%%, score=%.4f (weight=%.2f)", 
                mcs_present, mcs_pident * 100, mcs_score, weights["mcs"])
    logger.info("Promoter:         present=%s, identity=%.2f%%, score=%.4f (weight=%.2f)", 
                prom_present, prom_pident * 100, prom_score, weights["promoter"])
    logger.info("GC content bonus: %.4f", gc_bonus)
    logger.info("Main score:       %.4f (capped at 1.0)", main_score)
    logger.info("TOTAL REWARD:     %.4f", total_score)
    logger.info("==================================")
    
    return float(total_score)


def get_plasmid_reward(
    plasmid: Optional[str] = None,
    *,
    solution_str: Optional[str] = None,
    data_source: Optional[str] = None,
    ground_truth: Optional[str] = None,
    extra_info: Optional[Dict] = None,
    **kwargs,
) -> float:
    """
    Calculate reward for a plasmid sequence by querying informatics endpoints.
    
    Args:
        plasmid: DNA sequence string (preferred)
        solution_str: Alternative field where VERL passes the generated text
        
    Returns:
        float: Reward score between 0.0 and ~1.05 (including GC bonus)
    """
    if plasmid is None or not str(plasmid).strip():
        plasmid = solution_str or ""

    if not plasmid.strip():
        logger.warning("Empty plasmid sequence provided")
        return 0.0
    
    sequence = plasmid.strip()
    
    # Log input sequence info
    logger.info("=== EVALUATING PLASMID SEQUENCE ===")
    logger.info("Sequence length: %d bp", len(sequence))
    logger.info("Sequence preview: %s%s", 
                sequence[:100], "..." if len(sequence) > 100 else "")
    logger.info("Querying informatics server: %s", INFORMATICS_SERVER_URL)
    
    try:
        # Make parallel API calls
        api_results = _make_parallel_requests(sequence)
        
        if not api_results:
            logger.warning("No successful API responses - returning 0.0")
            return 0.0
        
        # Combine results into final score
        reward = _combine_results(api_results)
        
        logger.info("=== FINAL RESULT ===")
        logger.info("Plasmid reward: %.4f", reward)
        logger.info("==================")
        
        return reward
        
    except Exception as e:
        logger.error("Failed to calculate plasmid reward: %s", e)
        logger.info("=== FINAL RESULT ===")
        logger.info("Plasmid reward: 0.0000 (error)")
        logger.info("==================")
        return 0.0


def compute_score(
    data_source: Optional[str],
    solution_str: Optional[str],
    ground_truth: Optional[str],
    extra_info: Optional[Dict] = None,
):
    """VERL-expected reward signature.

    Delegates to get_plasmid_reward using the generated solution string.
    You may incorporate data_source/ground_truth/extra_info if needed.
    """
    return get_plasmid_reward(solution_str=solution_str, data_source=data_source, ground_truth=ground_truth, extra_info=extra_info)