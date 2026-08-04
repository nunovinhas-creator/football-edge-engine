from src.live.engine import LiveGoalEngine
from src.engine.simulation import MonteCarloSimulator

from src.live.providers.api_match_provider import APIMatchProvider
from src.live.providers.api_odds_provider import APIOddsProvider

# FALLBACK_LAMBDA_HOME é devolvido por calculate_dynamic_lambda() sempre que
# os dados ao vivo estão ausentes/inválidos (para o λ_home OU para o λ_away,
# já que evaluate() reutiliza a mesma função/fallback para os dois — ver
# docs/AUDIT_MATEMATICA.md, secção Monte Carlo).
# FALLBACK_LAMBDA_AWAY deixou de ser o valor fixo de λ_away em evaluate()
# (agora dinâmico, ver evaluate()); mantido apenas como o valor legado
# documentado para quem ainda o importar.
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

        # λ_away dinâmico: reutiliza exatamente a mesma calculate_dynamic_lambda()
        # do λ_home, mas alimentada com a métrica ao vivo específica da equipa
        # visitante disponível em LiveMatchState (away_conceded_xg_last5). A
        # pressão ao vivo (`pressure`) não é medida separadamente por equipa
        # neste sistema, pelo que se reutiliza o mesmo valor já calculado pelo
        # Goal Engine para ambas as chamadas — nenhuma fórmula nova é criada.
        away_live_result = {
            "estimated_xg_10m": match_state.away_conceded_xg_last5,
            "pressure": live_result.get("pressure"),
        }
        lambda_away = self.calculate_dynamic_lambda(away_live_result)

        # Cartões vermelhos: LiveMatchState.red_cards é uma contagem agregada
        # do jogo (a integração BSD atual não distingue, nesta camada, a que
        # equipa pertence cada cartão). Como simplificação mínima e
        # documentada, o fator de inferioridade numérica é aplicado ao λ
        # restante da equipa visitante, sem tocar no Goal Engine nem no
        # adaptador BSD.
        if match_state.red_cards > 0:
            lambda_away = round(
                lambda_away * max(0.0, 1 - 0.15 * match_state.red_cards),
                2
            )

        simulation = self.simulator.run_match_simulation(
            current_minute=match_state.minute,
            current_home_score=match_state.home_score,
            current_away_score=match_state.away_score,
            home_lambda=lambda_home,
            away_lambda=lambda_away,
            match_id=match_id
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
