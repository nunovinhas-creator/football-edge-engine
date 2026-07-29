import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


df = pd.read_csv(
    "data/processed/pressure_shots/history_features.csv"
)


features = [
    "attack_avg_last5",
    "dangerous_attack_avg_last5",
    "ball_safe_avg_last5",
]


target = "total_shots"


df = df.dropna()


# treino/teste temporal
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


model.fit(
    train[features],
    train[target]
)


pred = model.predict(
    test[features]
)


mae = mean_absolute_error(
    test[target],
    pred
)


rmse = np.sqrt(
    mean_squared_error(
        test[target],
        pred
    )
)


print("Resultados Backtest")
print("-------------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


resultado = pd.DataFrame({
    "real": test[target],
    "previsto": pred
})


print(resultado.head(20))
