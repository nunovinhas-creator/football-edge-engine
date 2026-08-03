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

        for market, odd in match.odds.items():

            if market not in [
                "HOME",
                "DRAW",
                "AWAY"
            ]:
                continue


            market_probability = implied_probability(
                odd
            )


            edge = calculate_edge(
                match.probability,
                market_probability
            )


            ev = calculate_ev(
                match.probability,
                odd
            )


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

                "model_probability":
                match.probability,

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
