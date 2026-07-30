from dataclasses import dataclass


@dataclass(slots=True)
class LiveMatchState:

    minute: int = 0

    home_score: int = 0
    away_score: int = 0

    home_xg_last5: float = 1.5
    away_conceded_xg_last5: float = 1.2

    home_style: str = "balanced"

    dangerous_attacks_10m: int = 0
    shots_on_target_10m: int = 0
    shots_10m: int = 0
    corners_10m: int = 0

    possession: float = 50.0

    previous_pressure: float = 0.0

    # BSD incident features

    goals_last_15: int = 0
    last_goal_minute: int | None = None
    red_cards: int = 0
    game_state: str = "unknown"
