from src.live.providers.api_match_provider import APIMatchProvider
from src.engine.live_pipeline import LivePipeline

provider = APIMatchProvider()
pipeline = LivePipeline(match_provider=provider)

live = provider.get_live_matches()
events = live.get("events", [])

print(f"\n⚽ LIVE MATCHES: {len(events)}\n")

alerts = 0

for match in events:

    match_id = match["id"]

    try:

        analysis = pipeline.evaluate(match_id)

        live_data = analysis["live"]

        odds = analysis["odds"]["odds"]

        probability = live_data["next_goal_probability"]

        odd = odds.get("over_15_goals")

        if odd is None:
            continue

        implied = (1 / odd) * 100
        edge = round(probability - implied, 2)

        if probability >= 60 or edge >= 5:

            alerts += 1

            print("=" * 70)
            print(f"🚨 {match['home_team']} vs {match['away_team']}")
            print(f"Minute      : {live_data['minute']}")
            print(f"Goal Prob   : {probability}%")
            print(f"Bookie Odd  : {odd}")
            print(f"Edge        : {edge}%")
            print(f"Decision    : {live_data['recommendation']}")
            print("=" * 70)

    except Exception as e:

        print(match_id, e)

if alerts == 0:
    print("❄️ Nenhuma oportunidade encontrada neste momento.")
