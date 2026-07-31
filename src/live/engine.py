import numpy as np
from src.models.live_state import LiveMatchState

class LiveGoalEngine:
    def __init__(self):
        pass

    def calculate_pressure(self, match: LiveMatchState) -> float:
        """Calcula a pressão ofensiva recente (últimos 10 min)."""
        attack_weight = match.dangerous_attacks_10m * 1.2
        shot_weight = match.shots_10m * 2.5
        target_weight = match.shots_on_target_10m * 4.0
        corner_weight = match.corners_10m * 1.5
        
        raw_pressure = attack_weight + shot_weight + target_weight + corner_weight
        smoothed_pressure = (raw_pressure * 0.7) + (match.previous_pressure * 0.3)
        return min(round(smoothed_pressure, 2), 100.0)

    def calculate_dominance_index(self, match: LiveMatchState) -> float:
        """
        Calcula o Índice de Domínio (0-100).
        Não depende só de posse de bola, mas do controlo territorial e perigo criado.
        """
        possession_factor = match.possession * 0.3
        attack_factor = min((match.dangerous_attacks_10m / 15.0) * 35.0, 35.0)
        shot_factor = min((match.shots_10m / 8.0) * 35.0, 35.0)
        
        dominance = possession_factor + attack_factor + shot_factor
        return round(min(dominance, 100.0), 1)

    def estimate_live_xg(self, match: LiveMatchState) -> float:
        """Estima o xG acumulado/esperado com base no volume de remates e perigo recente."""
        shot_xg = (match.shots_10m - match.shots_on_target_10m) * 0.08
        target_xg = match.shots_on_target_10m * 0.32
        corner_xg = match.corners_10m * 0.05
        
        estimated_10m_xg = shot_xg + target_xg + corner_xg
        return round(estimated_10m_xg, 2)

    def predict_next_goal_probability(self, match: LiveMatchState) -> dict:
        """Motor completo de análise live v2."""
        pressure = self.calculate_pressure(match)
        dominance = self.calculate_dominance_index(match)
        live_xg_10m = self.estimate_live_xg(match)
        
        # Base de xG histórico
        base_xg = (match.home_xg_last5 + match.away_conceded_xg_last5) / 2.0

        score_diff = abs(match.home_score - match.away_score)
        draw_bonus = 15 if score_diff == 0 else (8 if score_diff == 1 else 0)

        last_goal_bonus = 0
        if match.last_goal_minute is not None:
            last_goal_bonus = max(0,15-(match.minute-match.last_goal_minute))

        time_factor = min(match.minute/90.0,1.0)

        pressure_score = pressure
        pressure_score += draw_bonus
        pressure_score += last_goal_bonus
        pressure_score += min(base_xg*10,20)
        pressure_score += match.red_cards*5

        pressure_score = max(0,min(100,pressure_score))

        prob_pct = round(
            pressure_score*0.60 +
            dominance*0.15 +
            min(live_xg_10m*25,15) +
            time_factor*10,
            1
        )

        # Recomendação inteligente
        if prob_pct >= 55:
            rec = "🔥 BET (HIGH PROB)"
        elif prob_pct >= 35:
            rec = "⚠️ WAIT (PRESSURE BUILDING)"
        else:
            rec = "❄️ PASS (LOW ACTIVITY)"

        return {
            "minute": match.minute,
            "pressure": pressure,
            "dominance_index": dominance,
            "estimated_xg_10m": live_xg_10m,
            "next_goal_probability": prob_pct,
            "recommendation": rec
        }
