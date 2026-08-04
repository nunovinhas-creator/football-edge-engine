from datetime import datetime

from src.collector.client import EventCollector

from src.engine.edge import (
    implied_probability,
    calculate_edge,
    calculate_ev
)

from src.engine.decision import make_decision
from src.engine.stake import calculate_stake
from src.engine.explanation import generate_explanation
from src.engine.ranking import create_ranking
from src.report.printer import print_report
from src.utils.telegram_notifier import send_telegram_alert


# Mapa entre o nome de mercado usado por match.odds ("HOME"/"DRAW"/"AWAY")
# e a chave devolvida por src/engine/value.py::estimate_pregame_probabilities()
# ("home"/"draw"/"away"), que por sua vez vem do modelo Dixon-Coles já
# existente (src/engine/dixon_coles.py). Ver docs/AUDIT_MATEMATICA.md.
MARKET_TO_DIXON_COLES_KEY = {
    "HOME": "home",
    "DRAW": "draw",
    "AWAY": "away"
}


def run_predict():

    collector = EventCollector()

    matches = collector.get_matches(10)

    print()
    print("JOGOS RECEBIDOS:", len(matches))
    print()

    for m in matches:
        print(m.to_dict())
        print("----------------")


    results = []


    for match in matches:

        # Conjunto completo das odds 1X2 já disponíveis para este jogo
        # (Melhoria #7 da auditoria matemática — remoção do overround):
        # usado como `market_odds` abaixo para que Edge seja calculado
        # contra a probabilidade "fair" (sem margem) sempre que houver
        # pelo menos 2 odds válidas entre HOME/DRAW/AWAY. Não faz novas
        # chamadas HTTP nem introduz odds novas — reaproveita `match.odds`.
        market_odds_1x2 = {
            m: match.odds[m]
            for m in ("HOME", "DRAW", "AWAY")
            if match.odds.get(m) is not None and match.odds[m] > 1.0
        }

        for market, odd in match.odds.items():

            if market not in [
                "HOME",
                "DRAW",
                "AWAY"
            ]:
                continue

            if odd is None or odd <= 1.0:
                continue

            market_probability = implied_probability(
                odd
            )

            # Probabilidade pré-jogo do modelo Dixon-Coles já existente
            # (src/engine/dixon_coles.py, via src/engine/value.py), calculada
            # por mercado (HOME/DRAW/AWAY) — em vez de reutilizar a mesma
            # probabilidade heurística de H2H (match.probability) para os 3
            # mercados, como acontecia antes desta integração. Fallback para
            # a heurística de H2H apenas se, por algum motivo, o Dixon-Coles
            # não tiver sido calculado para este jogo (nunca falha a pipeline).
            dixon_coles_key = MARKET_TO_DIXON_COLES_KEY.get(market)

            if match.dixon_coles_probabilities and dixon_coles_key in match.dixon_coles_probabilities:
                model_probability_fraction = match.dixon_coles_probabilities[dixon_coles_key]
            else:
                model_probability_fraction = match.probability / 100.0

            # Edge oficial: prob_model - probabilidade implícita de mercado,
            # em pontos percentuais (mesma convenção usada por DecisionEngine
            # e evaluate_live_market). Bug corrigido: antes chamava-se
            # calculate_edge(match.probability, market_probability), passando
            # uma probabilidade (0-1) no lugar da odd esperada pela função.
            edge = round(
                calculate_edge(
                    model_probability_fraction,
                    odd,
                    market_odds=market_odds_1x2
                ) * 100,
                2
            )

            ev = round(
                calculate_ev(
                    model_probability_fraction,
                    odd,
                    market_odds=market_odds_1x2
                ) * 100,
                2
            )

            market_probability = round(market_probability * 100, 2)


            decision = make_decision(
                edge,
                ev
            )


            h2h_matches = 0

            h2h_matches = getattr(
                match,
                "h2h_matches",
                0
            )


            if abs(edge) >= 8 and ev >= 15 and h2h_matches >= 5:

                confidence = "HIGH"
                confidence_reason = "Edge elevado, EV forte e histórico suficiente"

            elif abs(edge) >= 5 and ev >= 10:

                confidence = "MEDIUM"
                confidence_reason = "Edge alto mas amostra limitada"

            else:

                confidence = "LOW"
                confidence_reason = "Dados insuficientes"


            stake = calculate_stake(
                edge,
                confidence,
                match.h2h_matches
            )


            result = {

                "match":
                f"{match.home} vs {match.away}",

                "league":
                match.league,

                "market":
                market,

                "odd":
                odd,

                # Probabilidade efetivamente usada para calcular edge/ev acima
                # (Dixon-Coles por mercado, com fallback para a heurística de
                # H2H — ver comentário junto a model_probability_fraction).
                "model_probability":
                round(model_probability_fraction * 100, 2),

                "market_probability":
                market_probability,

                "edge":
                edge,

                "ev":
                ev,

                "confidence":
                confidence,

                "confidence_reason":
                confidence_reason,

                "h2h_matches":
                getattr(
                    match,
                    "h2h_matches",
                    0
                ),

                "stake":
                stake,

                "decision":
                decision
            }


            result["reasons"] = generate_explanation(
                {
                    "edge": edge,
                    "ev": ev,
                    "confidence": confidence,
                    "xg": None
                }
            )


            results.append(result)



    print()
    print("RESULTADOS CALCULADOS:")
    print()


    for r in results:
        print(r)
        print("----------------")



    ranking = create_ranking(results)


    print()
    print("🔥 VALUE BETS")
    print()


    for bet in ranking["value_bets"]:
        print_report(bet)

    send_telegram_bulletin(ranking["value_bets"])


def decision_alert_label(decision: str) -> str:
    """Traduz o texto de decisão já calculado (make_decision) num rótulo
    de alerta. Não recalcula nem altera a decisão em si."""

    if "BET" in decision:
        return "🟢 APOSTAR"
    if "WAIT" in decision:
        return "🟡 AGUARDAR"
    return "⚪ PASSAR"


def format_bet_alert(bet: dict, now: datetime) -> str:
    edge_sign = "+" if bet["edge"] > 0 else ""
    ev_sign = "+" if bet["ev"] > 0 else ""

    return (
        f"{decision_alert_label(bet['decision'])}\n\n"
        f"{bet['match']}\n\n"
        f"Mercado:\n{bet['market']}\n\n"
        f"Probabilidade:\n{bet['model_probability']}%\n\n"
        f"Odd:\n{bet['odd']}\n\n"
        f"Edge:\n{edge_sign}{bet['edge']}%\n\n"
        f"EV:\n{ev_sign}{bet['ev']}%\n\n"
        f"Stake:\n{bet['stake']}%\n\n"
        f"Hora:\n{now.strftime('%H:%M')}"
    )


def send_telegram_bulletin(value_bets, max_alerts: int = 5) -> None:
    """Envia o boletim diário para o Telegram: um alerta por oportunidade
    aprovada (`ranking["value_bets"]`, já calculado por create_ranking/
    is_valid_bet — sem alterar essa lógica de decisão), ou uma mensagem
    informativa quando não há nenhuma oportunidade nesta ronda."""

    now = datetime.now()

    if not value_bets:
        send_telegram_alert(
            f"ℹ️ Análise concluída ({now.strftime('%d/%m/%Y %H:%M')}).\n"
            "Nenhuma oportunidade cumpriu os critérios de EV+ nesta ronda."
        )
        return

    for bet in value_bets[:max_alerts]:
        send_telegram_alert(format_bet_alert(bet, now))
