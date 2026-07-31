from src.live.engine import LiveGoalEngine
from src.engine.simulation import MonteCarloSimulator

from src.live.providers.api_match_provider import APIMatchProvider
from src.live.providers.api_odds_provider import APIOddsProvider


class LivePipeline:

    def __init__(
        self,
        match_provider=None,
        odds_provider=None
    ):

        self.live_engine = LiveGoalEngine()
        self.simulator = MonteCarloSimulator()

        self.match_provider = (
            match_provider
            if match_provider
            else APIMatchProvider()
        )

        self.odds_provider = (
            odds_provider
            if odds_provider
            else APIOddsProvider()
        )

    def calculate_dynamic_lambda(self, live_result):

        live_xg = live_result["estimated_xg_10m"]
        pressure = live_result["pressure"]

        base_lambda = 1.20

        xg_factor = live_xg * 0.30
        pressure_factor = (pressure / 100) * 0.60

        return round(
            min(base_lambda + xg_factor + pressure_factor, 4.0),
            2
        )

    def evaluate(self, match_id=1):

        match_state = self.match_provider.get_live_match(match_id)
        odds = self.odds_provider.get_live_odds(match_id)

        live_result = self.live_engine.predict_next_goal_probability(
            match_state
        )

        lambda_home = self.calculate_dynamic_lambda(
            live_result
        )

        simulation = self.simulator.run_match_simulation(
            current_minute=match_state.minute,
            current_home_score=0,
            current_away_score=0,
            home_lambda=lambda_home,
            away_lambda=0.80
        )

        return {
            "live": live_result,
            "odds": odds,
            "simulation": {
                "over_15": simulation.over_15_prob,
                "over_25": simulation.over_25_prob,
                "btts": simulation.btts_prob,
                "expected_home_goals": simulation.expected_goals_home,
                "expected_away_goals": simulation.expected_goals_away
            }
        }
