"""
Momentum Engine

Deteta se a pressão está a aumentar ou a diminuir.
"""

class Momentum:

    def calculate(self, current: float, previous: float) -> str:

        diff = current - previous

        if diff >= 20:
            return "SURGING"

        elif diff >= 10:
            return "RISING"

        elif diff <= -20:
            return "COLLAPSING"

        elif diff <= -10:
            return "FALLING"

        return "STABLE"
