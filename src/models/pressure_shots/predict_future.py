import joblib
import pandas as pd


model = joblib.load(
    "research/pressure_shots/models/random_forest_shots.pkl"
)


jogo = pd.DataFrame([
    {
        "attack_avg_last5": 250,
        "dangerous_attack_avg_last5": 120,
        "ball_safe_avg_last5": 300
    }
])


previsao = model.predict(jogo)


print(
    f"Remates previstos: {previsao[0]:.1f}"
)
