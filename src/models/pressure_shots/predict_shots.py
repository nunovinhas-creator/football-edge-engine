import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/processed/pressure_shots/history_features.csv")

features = [
    "attack_avg_last5",
    "dangerous_attack_avg_last5",
    "ball_safe_avg_last5",
]

target = "total_shots"

df = df.dropna()

X = df[features]
y = df[target]

model = LinearRegression()
model.fit(X, y)

print("Coeficientes:")
for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.4f}")

print(f"\nIntercepto: {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")
