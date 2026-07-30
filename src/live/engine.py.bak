import numpy as np

class LiveGoalEngine:
    def __init__(self, weight_history=0.2, weight_tactics=0.2, weight_live=0.6):
        self.w_hist = weight_history
        self.w_tact = weight_tactics
        self.w_live = weight_live

    def predict_next_goal_probability(self, match_data):
        hist_baseline = (match_data.get('home_xg_last5', 1.5) + match_data.get('away_conceded_xg_last5', 1.2)) / 2.0
        tactical_factor = 1.25 if match_data.get('home_style') == 'high_press' else 1.0
        
        da_rate = match_data.get('dangerous_attacks_10m', 0) / 10.0
        sot_rate = match_data.get('shots_on_target_10m', 0)
        corners_rate = match_data.get('corners_10m', 0)
        live_pressure = (da_rate * 0.4) + (sot_rate * 0.5) + (corners_rate * 0.2)

        combined_score = (hist_baseline * self.w_hist) + (tactical_factor * self.w_tact) + (live_pressure * self.w_live)
        raw_prob = 1 / (1 + np.exp(-(combined_score - 1.2) * 2.5))
        return float(np.clip(raw_prob, 0.05, 0.95))
