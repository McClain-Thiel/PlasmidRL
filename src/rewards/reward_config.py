from pydantic import BaseModel

class RewardConfig(BaseModel):

    punish_mode: bool = True # penaize violations of the reward config as opposed to just not rewarding them
    length_penalty: bool = True # penalize sequences that are too long or too short
    location_aware: bool = True # reward sequences that are located in the correct location (e.g. prompoter then cds then terminatr)
    
    ori_min: int = 1
    ori_max: int = 2
    allowed_oris: Optional[List[str]] = None
    ori_weight: float = 1.0

    promoter_min: int = 1
    promoter_max: int = 2
    allowed_promoters: Optional[List[str]] = None
    promoter_weight: float = 1.0

    terminator_min: int = 1
    terminator_max: int = 2
    allowed_terminators: Optional[List[str]] = None
    terminator_weight: float = 1.0

    marker_min: int = 1
    marker_max: int = 2
    allowed_markers: Optional[List[str]] = None
    marker_weight: float = 1.0

    cds_min: int = 1
    cds_max: int = 2
    allowed_cds: Optional[List[str]] = None
    cds_weight: float = 1.0

