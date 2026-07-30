import numpy as np
import os
import pickle
from dataclasses import dataclass
from src.models.live_state import LiveMatchState

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

    def predict(self, match: LiveMatchState) -> MLPredictionResult:
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
