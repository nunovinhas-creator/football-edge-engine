from src.live.engine import LiveGoalEngine
from src.engine.simulation import MonteCarloSimulator

from src.live.providers.api_match_provider import APIMatchProvider
from src.live.providers.api_odds_provider import APIOddsProvider

# FALLBACK_LAMBDA_HOME só é usado se o λ_home dinâmico não puder ser
# calculado (dados ao vivo ausentes/inválidos) — é o valor fixo que o
# dashboard usava antes de consumir o λ dinâmico (ver
# docs/AUDIT_MATEMATICA.md, secção Monte Carlo).
# FALLBACK_LAMBDA_AWAY mantém-se: não existe, atualmente, um cálculo
# dinâmico de λ_away no sistema, por isso é sempre este valor fixo (já era
# o valor usado por esta pipeline antes desta alteração).
FALLBACK_LAMBDA_HOME = 1.6
FALLBACK_LAMBDA_AWAY = 0.80


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
        """
        Calcula o λ_home dinâmico a partir da pressão e do xG ao vivo.

        Fallback: se `live_result` não tiver os campos esperados, ou estes
        não forem numéricos, devolve FALLBACK_LAMBDA_HOME em vez de
        propagar a excepção — a simulação nunca deve falhar por falta de
        dados ao vivo.
        """
        try:
            live_xg = float(live_result["estimated_xg_10m"])
            pressure = float(live_result["pressure"])
        except (KeyError, TypeError, ValueError):
            return FALLBACK_LAMBDA_HOME

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

        # Única origem da verdade para o λ_home usado pelo Monte Carlo.
        lambda_home = self.calculate_dynamic_lambda(
            live_result
        )

        # Sem fonte dinâmica de λ_away disponível no sistema atual — usa-se
        # o fallback fixo documentado no topo do módulo.
        lambda_away = FALLBACK_LAMBDA_AWAY

        simulation = self.simulator.run_match_simulation(
            current_minute=match_state.minute,
            current_home_score=0,
            current_away_score=0,
            home_lambda=lambda_home,
            away_lambda=lambda_away
        )

        return {
            "live": live_result,
            "odds": odds,
            "lambda": {
                "home": lambda_home,
                "away": lambda_away
            },
            "simulation": {
                "over_15": simulation.over_15_prob,
                "over_25": simulation.over_25_prob,
                "btts": simulation.btts_prob,
                "expected_home_goals": simulation.expected_goals_home,
                "expected_away_goals": simulation.expected_goals_away
            }
        }
