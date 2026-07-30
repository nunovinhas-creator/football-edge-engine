from src.engine.value_scanner import scan_value_opportunities
from src.models.match import Match


matches = [

    Match(
        "Benfica",
        "Porto",
        2.10,
        55
    ),

    Match(
        "Milan",
        "Roma",
        1.90,
        58
    )

]


results = scan_value_opportunities(
    matches,
    live_probability=72,
    live_odd=2.10
)


print("====================")

for r in results:
    print(r)

print("====================")
