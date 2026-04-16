from dataclasses import dataclass, field
from typing import Optional, List

@dataclass(slots=True)
class PlayerStats:
    name: str
    score: str
    sets: List[str] = field(default_factory=list)
    is_serving: bool = False

@dataclass(slots=True)
class MatchUpdate:
    match_id: str
    p1: PlayerStats
    p2: PlayerStats
    status: str  # e.g., "LIVE", "FINISHED"
    timestamp: float
    raw_event: Optional[str] = None
