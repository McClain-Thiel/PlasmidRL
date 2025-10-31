from pydantic import BaseModel
from typing import Optional, List

class RewardConfig(BaseModel):

    punish_mode: bool = True # penaize violations of the reward config as opposed to just not rewarding them
    length_penalty: bool = True # penalize sequences that are too long or too short
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    location_aware: bool = True # reward sequences that are located in the correct location (e.g. prompoter then cds then terminatr)
    # Penalty factor applied when min/max constraints are violated (outside of range)
    violation_penalty_factor: float = 0.5
    
    ori_min: int = 1
    ori_max: int = 1
    allowed_oris: Optional[List[str]] = None
    ori_weight: float = 1.0

    promoter_min: int = 1
    promoter_max: int = 1
    allowed_promoters: Optional[List[str]] = None
    promoter_weight: float = 1.0

    terminator_min: int = 0
    terminator_max: int = 2
    allowed_terminators: Optional[List[str]] = None
    terminator_weight: float = 0.5

    marker_min: int = 1
    marker_max: int = 2
    allowed_markers: Optional[List[str]] = None
    marker_weight: float = 1.0

    cds_min: int = 1
    cds_max: int = 5
    allowed_cds: Optional[List[str]] = None
    cds_weight: float = 1.0

    # Location-aware cassette scoring constants (simplified)
    cassette_max_cassettes: int = 2
    cassette_order_points: float = 5.0
    cassette_proximity_points: float = 5.0
    cassette_max_points_per: float = 20.0
    # Single proximity threshold (bp) for awarding proximity points
    proximity_threshold_bp: int = 300
    # Final CDS location-aware bonus scale (added on top of count score, then clamped)
    location_bonus_scale: float = 0.5
    # Overlap merge threshold (fraction of the smaller interval that must overlap)
    overlap_merge_threshold: float = 0.8

    def log_to_wandb(self):
        raise NotImplementedError("Not implemented")

