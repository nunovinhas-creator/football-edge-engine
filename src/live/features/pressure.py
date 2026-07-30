"""
Pressure Index Engine (v2)

Calcula um índice de pressão ofensiva entre 0 e 100.
"""

class PressureIndex:

    def calculate(
        self,
        dangerous_attacks: int,
        shots_on_target: int,
        shots: int,
        corners: int,
        possession: float,
    ) -> float:

        score = (
            dangerous_attacks * 1.2 +
            shots_on_target * 4.0 +
            shots * 1.5 +
            corners * 1.2 +
            max(possession - 50, 0) * 0.25
        )

        return round(min(score, 100.0), 2)
