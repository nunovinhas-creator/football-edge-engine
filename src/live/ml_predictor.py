import joblib
import os
import pandas as pd


class MLGoalPredictor:

    def __init__(self):

        path="models/live_goal_model.pkl"

        if not os.path.exists(path):
            raise FileNotFoundError(
                "Modelo não existe: "+path
            )

        self.model=joblib.load(path)


    def predict(self,data):

        df=pd.DataFrame([data])

        prob=self.model.predict_proba(df)[0][1]

        return {
            "goal_probability_15m":
                round(prob*100,2),

            "signal":
                "BET"
                if prob>0.55
                else "WAIT"
        }
