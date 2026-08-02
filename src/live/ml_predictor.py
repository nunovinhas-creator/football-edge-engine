import json
import os

import joblib
import pandas as pd

DEFAULT_THRESHOLD = 0.5
THRESHOLD_CONFIG_PATH = "models/live_goal_model_threshold.json"


class MLGoalPredictor:

    def __init__(self):

        path="models/live_goal_model.pkl"

        if not os.path.exists(path):
            raise FileNotFoundError(
                "Modelo não existe: "+path
            )

        self.model=joblib.load(path)
        self.threshold=self._load_threshold()


    def _load_threshold(self):
        if not os.path.exists(THRESHOLD_CONFIG_PATH):
            return DEFAULT_THRESHOLD

        try:
            with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            return float(config.get("threshold", DEFAULT_THRESHOLD))
        except (json.JSONDecodeError, TypeError, ValueError, OSError):
            return DEFAULT_THRESHOLD


    def predict(self,data):

        df=pd.DataFrame([data])

        prob=self.model.predict_proba(df)[0][1]

        return {
            "goal_probability_15m":
                round(prob*100,2),

            "signal":
                "BET"
                if prob>self.threshold
                else "WAIT"
        }
