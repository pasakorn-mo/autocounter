from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TrackState:
    object_id: int
    last_pos: Tuple[int, int]
    last_ts: float

    lane_id: Optional[str] = None

    crossed_l1_ts: Optional[float] = None
    crossed_l2_ts: Optional[float] = None

    finalized: bool = False
