from src.engine.confidence import confidence_level


scores = [90, 75, 45]


for score in scores:
    print(
        score,
        confidence_level(score)
    )
