"""
Attack Score Engine

Calcula o potencial ofensivo de uma equipa numa escala de 0 a 100.
"""

class AttackScore:

    def calculate(
        self,
        pressure: float,
        shots: int,
        shots_on_target: int,
        dangerous_attacks: int,
    ) -> float:

        score = (
            pressure * 0.45 +
            shots * 2.0 +
            shots_on_target * 5.0 +
            dangerous_attacks * 1.0
        )

        return round(min(score, 100.0), 2)
