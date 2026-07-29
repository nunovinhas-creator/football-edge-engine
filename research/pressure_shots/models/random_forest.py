import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/processed/pressure_shots/history_features.csv")

df = df.dropna()

features = [
    "attack_avg_last5",
    "dangerous_attack_avg_last5",
    "ball_safe_avg_last5",
]

X = df[features]
y = df["total_shots"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

print("Modelo treinado")

print("Importância das variáveis:")

for f, i in zip(features, model.feature_importances_):
    print(f"{f}: {i:.4f}")


joblib.dump(
    model,
    "research/pressure_shots/models/random_forest_shots.pkl"
)

print("Modelo guardado")
