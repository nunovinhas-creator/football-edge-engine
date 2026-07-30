import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

def generate_synthetic_training_data(n_samples: int = 5000):
    """
    Gera um dataset sintético baseado nas distribuições estatísticas de 
    jogos reais para calibrar o modelo base antes de injetar histórico real.
    """
    np.random.seed(42)
    
    minutes = np.random.randint(1, 90, n_samples)
    dangerous_attacks = np.random.randint(0, 25, n_samples)
    shots_on_target = np.random.randint(0, 8, n_samples)
    shots = shots_on_target + np.random.randint(0, 10, n_samples)
    corners = np.random.randint(0, 8, n_samples)
    possession = np.random.uniform(30.0, 70.0, n_samples)
    previous_pressure = np.random.uniform(10.0, 80.0, n_samples)
    
    shots_ratio = shots_on_target / np.maximum(1, shots)
    danger_intensity = dangerous_attacks * (possession / 100.0)
    xg_diff = np.random.uniform(-1.5, 1.5, n_samples)

    X = np.column_stack([
        minutes, dangerous_attacks, shots_on_target, shots, corners,
        possession, previous_pressure, shots_ratio, danger_intensity, xg_diff
    ])

    # Target: golo nos próximos 10m (1 ou 0)
    score_signal = (shots_on_target * 0.35) + (danger_intensity * 0.05) + (corners * 0.1)
    prob_signal = 1.0 / (1.0 + np.exp(- (score_signal - 2.0)))
    y = (np.random.uniform(0, 1, n_samples) < prob_signal).astype(int)

    return X, y

def train_and_save_model(model_path: str = "models_data/xgboost_live_v1.pkl"):
    print("🧠 A gerar dataset para treino do XGBoost...")
    X, y = generate_synthetic_training_data()

    print("⚡ A treinar o XGBoost Classifier + Calibração de Probabilidades...")
    base_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )

    # Calibração via Platt Scaling / Sigmoid
    calibrated_model = CalibratedClassifierCV(estimator=base_xgb, cv=3, method="sigmoid")
    calibrated_model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_model, f)

    print(f"✅ Modelo treinado e guardado com sucesso em: {model_path}")

if __name__ == "__main__":
    train_and_save_model()
