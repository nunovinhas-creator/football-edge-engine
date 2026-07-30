from dataclasses import dataclass

@dataclass(slots=True)
class LiveMatchState:
    minute: int = 0

    home_xg_last5: float = 1.5
    away_conceded_xg_last5: float = 1.2

    home_style: str = "balanced"

    dangerous_attacks_10m: int = 0
    shots_on_target_10m: int = 0
    shots_10m: int = 0
    corners_10m: int = 0

    possession: float = 50.0

    previous_pressure: float = 0.0
    dangerous_attacks_10m: int = 0
    shots_10m: int = 0
