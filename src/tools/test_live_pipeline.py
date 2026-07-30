from src.engine.live_pipeline import LivePipeline
from src.models.live_state import LiveMatchState


state = LiveMatchState(
    minute=76,
    possession=65,
    dangerous_attacks_10m=18,
    shots_on_target_10m=5,
    shots_10m=12,
    corners_10m=4,
    previous_pressure=60
)


pipeline = LivePipeline()

result = pipeline.evaluate(state)

print("====================")
print(result)
print("====================")
