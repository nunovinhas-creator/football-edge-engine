from src.api.client import BzzoiroClient
from src.models.match import Match
from src.model.predictor import predict_probability
from src.collector.odds import OddsCollector
from src.engine.pregame_lambda import estimate_pregame_lambdas
from src.engine.lambda_estimator import estimate_lambda
from src.engine.value import estimate_pregame_probabilities


class EventCollector:

    def __init__(self):

        self.client = BzzoiroClient()
        self.odds = OddsCollector()


    def get_matches(self, limit=10):

        data = self.client.get(
            f"events/?limit={limit}"
        )

        matches = []


        for event in data["results"]:

            probability = predict_probability(
                event
            )


            market = self.odds.get_event_odds(
                event["id"]
            )


            h2h = event.get("head_to_head")

            # Golos esperados pré-jogo (lambda_home/mu_away) para o modelo
            # Dixon-Coles já existente (src/engine/dixon_coles.py). Estimador
            # por omissão: src/engine/lambda_estimator.py (usa mais da
            # granularidade já devolvida em head_to_head — golos por equipa,
            # jogos recentes — com encolhimento estatístico para a média de
            # liga em amostras pequenas; ver docs/05_lambda_estimator.md).
            # Fallback defensivo para o adaptador heurístico anterior
            # (src/engine/pregame_lambda.py, inalterado) caso o novo
            # estimador levante alguma exceção inesperada — a pipeline nunca
            # deve falhar por causa da estimação de lambda.
            try:
                lambda_home, mu_away = estimate_lambda(h2h)
            except Exception:
                lambda_home, mu_away = estimate_pregame_lambdas(h2h)

            dixon_coles_probabilities = estimate_pregame_probabilities(
                lambda_home,
                mu_away
            )

            matches.append(
                Match(
                    home=event["home_team"],
                    away=event["away_team"],
                    odds=market,
                    probability=probability,
                    league=str(
                        event["league_id"]
                    ),
                    xg_home=lambda_home,
                    xg_away=mu_away,
                    confidence=50,
                    h2h_matches=(h2h or {}).get(
                        "total_matches",
                        0
                    ),
                    dixon_coles_probabilities=dixon_coles_probabilities
                )
            )


        return matches
