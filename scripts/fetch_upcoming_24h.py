"""
Lista as oportunidades das próximas N horas (default: 24), com as
probabilidades Monte Carlo (Over 1.5 / Over 2.5 / BTTS) de cada jogo.

Usa exclusivamente `src.report.upcoming_matches.list_upcoming_opportunities`
— não recalcula nada, apenas imprime o que esse módulo já devolve.

Uso:
    BSD_API_KEY=... python3 scripts/fetch_upcoming_24h.py [--hours 24]
"""

import argparse

from src.report import upcoming_matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=24, help="Janela em horas (default: 24)")
    args = parser.parse_args()

    opportunities = upcoming_matches.list_upcoming_opportunities(hours=args.hours)

    print(f"\n⚽ Oportunidades nas próximas {args.hours}h: {len(opportunities)}\n")

    for opp in opportunities:
        card = opp["card"]
        mc = opp["snapshot"]["models"]["monte_carlo"]

        print("=" * 70)
        print(f"{opp['kickoff']['hour_label']} | {card['home_team']} vs {card['away_team']} | {card['competition']}")
        print(f"Decisão     : {opp['decision']['label']}")
        print(f"Engine Score: {opp['engine_score']['score']} {opp['star_rating']}")
        print(f"Monte Carlo : Over 1.5 = {mc['over_15']}%  Over 2.5 = {mc['over_25']}%  BTTS = {mc['btts']}%")
        print(f"Destaque    : {opp['monte_carlo_headline']}")
        print("=" * 70)

    if not opportunities:
        print("❄️ Nenhum jogo agendado nesta janela.")


if __name__ == "__main__":
    main()
