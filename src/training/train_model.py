import pandas as pd
import joblib

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


df=pd.read_csv(
"data/training_dataset.csv"
)


features=[
"current_minute",
"home_score",
"away_score",
"dangerous_attacks_10m",
"shots_on_target_10m",
"corners_10m",
"live_odd_over",
"pressure",
"live_xg",
"red_cards",
"possession"
]


X=df[features]

y=df["goal_in_next_15m"]


model=LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    class_weight="balanced"
)


X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


pred=model.predict_proba(
    X_test
)[:,1]


print(
"AUC:",
roc_auc_score(
y_test,
pred
)
)


joblib.dump(
model,
"models/live_goal_model.pkl"
)


print(
"Modelo guardado"
)
