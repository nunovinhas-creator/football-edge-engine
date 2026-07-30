from src.live.engine import LiveGoalEngine
from src.engine.simulation import MonteCarloSimulator


class LivePipeline:
    """
    Orquestra os motores do Football Edge Engine v4.
    
    Liga:
    - LiveGoalEngine
    - Monte Carlo Simulation Engine
    """

    def __init__(self):
        self.live_engine = LiveGoalEngine()
        self.simulator = MonteCarloSimulator()

    def calculate_dynamic_lambda(self, live_result):
        """
        Converte pressão e xG live numa intensidade
        de golo compatível com o simulador.
        """

        live_xg = live_result["estimated_xg_10m"]
        pressure = live_result["pressure"]

        base_lambda = 1.20

        xg_factor = live_xg * 0.30
        pressure_factor = (pressure / 100) * 0.60

        lambda_live = base_lambda + xg_factor + pressure_factor

        return round(min(lambda_live, 4.0), 2)

    def evaluate(self, match_state):

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
            "simulation": {
                "over_15": simulation.over_15_prob,
                "over_25": simulation.over_25_prob,
                "btts": simulation.btts_prob,
                "expected_home_goals": simulation.expected_goals_home,
                "expected_away_goals": simulation.expected_goals_away
            }
        }
