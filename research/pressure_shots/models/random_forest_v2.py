import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


df = pd.read_csv(
    "data/processed/pressure_shots/features_v2.csv"
)


features = [
    "attack_avg_last5",
    "dangerous_attack_avg_last5",
    "ball_safe_avg_last5",
    "total_shots_avg_last5",
    "shots_on_target_avg_last5",
    "attack_difference",
    "dangerous_attack_difference",
    "ball_safe_difference",
    "is_home",
]


target = "total_shots"


df = df.dropna()


split = int(len(df) * 0.8)


train = df.iloc[:split]
test = df.iloc[split:]


model = RandomForestRegressor(
    n_estimators=300,
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


print("Random Forest v2")
print("----------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


importances = pd.Series(
    model.feature_importances_,
    index=features
)


print("\nImportância:")
print(
    importances.sort_values(
        ascending=False
    )
)
