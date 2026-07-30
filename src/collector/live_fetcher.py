from src.models.live_state import LiveMatchState

class LiveDataCollector:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def parse_api_football_to_state(self, match_payload: dict) -> LiveMatchState:
        """
        Converte a estrutura de resposta típica da API-Football / Football-Data
        no formato padronizado LiveMatchState que o nosso motor exige.
        """
        fixture = match_payload.get("fixture", {})
        status = fixture.get("status", {})
        minute = status.get("elapsed", 0)

        stats = match_payload.get("statistics", {})
        home_stats = stats.get("home", {})

        possession = float(home_stats.get("possession", "50%").replace("%", ""))
        shots_on_target = home_stats.get("shots_on_target", 0)
        total_shots = home_stats.get("total_shots", 0)
        corners = home_stats.get("corners", 0)
        dangerous_attacks = home_stats.get("dangerous_attacks", 0)

        return LiveMatchState(
            minute=minute,
            possession=possession,
            dangerous_attacks_10m=dangerous_attacks,
            shots_on_target_10m=shots_on_target,
            shots_10m=total_shots,
            corners_10m=corners,
            previous_pressure=50.0
        )
