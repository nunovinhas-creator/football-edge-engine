import numpy as np
from src.models.live_state import LiveMatchState

class LiveGoalEngine:
    def __init__(self):
        pass

    def calculate_pressure(self, match: LiveMatchState) -> float:
        """Calcula a pressão ofensiva recente (últimos 10 min)."""
        attack_weight = getattr(match, "dangerous_attacks_10m", match.get("dangerous_attacks_10m", 0) if isinstance(match, dict) else 0) * 1.2
        shot_weight = getattr(match, "shots_10m", 0) * 2.5
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
        
        # Base de xG histórico vs ao vivo
        base_xg = (match.home_xg_last5 + match.away_conceded_xg_last5) / 2.0
        
        # Fator tempo/minuto do jogo (jogo ganha urgência após os 60')
        time_factor = 1.0 + (match.minute / 90.0) * 0.25
        
        # Fórmula de Probabilidade Combinada
        prob = ((pressure / 100.0) * 0.35) + \
               ((dominance / 100.0) * 0.25) + \
               (min(live_xg_10m / 1.5, 1.0) * 0.40)
               
        prob = min(prob * time_factor, 0.95)
        prob_pct = round(prob * 100, 1)

        # Recomendação inteligente
        if prob_pct >= 70.0:
            rec = "🔥 BET (HIGH PROB)"
        elif prob_pct >= 50.0:
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
