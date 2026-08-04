import numpy as np
import os
import pickle
from dataclasses import dataclass
from src.models.live_state import LiveMatchState
from src.live.engine import LiveGoalEngine
from src.live.ml_predictor import MLGoalPredictor

@dataclass
class MLPredictionResult:
    goal_probability: float
    confidence_score: float
    model_used: str

class LiveMLPredictor:
    def __init__(self, model_path: str = "models_data/xgboost_live_v1.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()

        # Pipeline real (src/training/train_model.py, treinado sobre
        # data/live_history.db). Usado preferencialmente em predict();
        # o modelo sintetico acima so serve de fallback se este nao existir.
        self.goal_engine = LiveGoalEngine()
        try:
            self.real_predictor = MLGoalPredictor()
        except FileNotFoundError:
            self.real_predictor = None

    def _load_model(self):
        """Carrega o modelo XGBoost pré-treinado se existir no projeto."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None

    def extract_features(self, match: LiveMatchState) -> np.ndarray:
        """
        Converte o estado do jogo num vetor numérico de Features para a IA.
        """
        shots_ratio = match.shots_on_target_10m / max(1, match.shots_10m)
        danger_intensity = match.dangerous_attacks_10m * (match.possession / 100.0)
        xg_diff = match.home_xg_last5 - match.away_conceded_xg_last5

        features = [
            match.minute,
            match.dangerous_attacks_10m,
            match.shots_on_target_10m,
            match.shots_10m,
            match.corners_10m,
            match.possession,
            match.previous_pressure,
            shots_ratio,
            danger_intensity,
            xg_diff
        ]
        return np.array(features).reshape(1, -1)

    def predict(self, match: LiveMatchState, live_odd_over: float) -> MLPredictionResult:
        # pressure/live_xg sao recalculados aqui com o mesmo LiveGoalEngine
        # que ja alimenta o resto do dashboard (mesma funcao pura, mesmo
        # match) - valores reais e identicos aos mostrados em "Pressure"/
        # "xG 10m", nao inventados. live_odd_over vem da odd de mercado real
        # do caller: LiveMatchState nao tem um campo equivalente.
        if self.real_predictor is not None:
            live_result = self.goal_engine.predict_next_goal_probability(match)
            data = {
                "current_minute": match.minute,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "dangerous_attacks_10m": match.dangerous_attacks_10m,
                "shots_on_target_10m": match.shots_on_target_10m,
                "corners_10m": match.corners_10m,
                "live_odd_over": live_odd_over,
                "pressure": live_result["pressure"],
                "live_xg": live_result["estimated_xg_10m"],
                "red_cards": match.red_cards,
                "possession": match.possession,
            }
            result = self.real_predictor.predict(data)
            prob = result["goal_probability_15m"] / 100.0
            return MLPredictionResult(
                goal_probability=result["goal_probability_15m"],
                confidence_score=round(abs(prob - 0.5) * 200, 1),
                model_used="LiveGoalModel (dados reais, LightGBM calibrado - src/training/train_model.py)"
            )

        features = self.extract_features(match)

        # Se houver um modelo treinado em ficheiro, usamos a inferência XGBoost
        if self.model is not None:
            raw_prob = float(self.model.predict_proba(features)[0][1])
            return MLPredictionResult(
                goal_probability=round(raw_prob * 100.0, 1),
                confidence_score=92.0,
                model_used="XGBoost Live v1.0 (Calibrated)"
            )
        
        # Fallback Heurístico Base (Enquanto o modelo não é treinado em dataset gigante)
        base_score = (
            (match.shots_on_target_10m * 0.25) +
            (match.dangerous_attacks_10m * 0.05) +
            (match.corners_10m * 0.08)
        )
        prob = min(max(base_score * 10.0, 5.0), 95.0)
        return MLPredictionResult(
            goal_probability=round(prob, 1),
            confidence_score=65.0,
            model_used="Heuristic Rule-Based Fallback"
        )
