import numpy as np
from src.models.live_state import LiveMatchState

class LiveGoalEngine:
    def __init__(self):
        pass

    def calculate_pressure(self, match):
        # Exemplo de cálculo de raw_pressure (ajusta se tiveres fórmula específica)
        raw_pressure = getattr(match, 'raw_pressure', 0.0) if not isinstance(match, dict) else match.get('raw_pressure', 0.0)

        # Trata o previous_pressure com segurança se for um dict ou LiveMatchState
        if isinstance(match, dict):
            prev_pressure = match.get('previous_pressure', raw_pressure)
        else:
            prev_pressure = getattr(match, 'previous_pressure', raw_pressure)

        if prev_pressure is None:
            prev_pressure = raw_pressure

        smoothed_pressure = (raw_pressure * 0.7) + (prev_pressure * 0.3)
        return smoothed_pressure

    def predict_next_goal_probability(self, match_data):
        pressure = self.calculate_pressure(match_data)
        # Exemplo de cálculo simples de probabilidade
        p_goal = min(1.0, max(0.0, pressure / 100.0))
        return p_goal
