from src.services.predictions import get_predictions

data = get_predictions(5)

for p in data["results"]:
    event = p["event"]
    markets = p["markets"]

    print("=" * 60)
    print(f'{event["home_team"]} vs {event["away_team"]}')
    print(event["event_date"])
    print()

    print(
        f'1X2: '
        f'H {markets["match_result"]["prob_home"]}% | '
        f'D {markets["match_result"]["prob_draw"]}% | '
        f'A {markets["match_result"]["prob_away"]}%'
    )

    print(f'xG: {markets["expected_goals"]["home"]} - {markets["expected_goals"]["away"]}')
    print(f'Over 2.5: {markets["over_under"]["prob_over_25"]}%')
    print(f'BTTS: {markets["btts"]["prob_yes"]}%')
    print(f'Score provável: {markets["score"]["most_likely"]}')
