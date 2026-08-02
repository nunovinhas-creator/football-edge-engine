import json
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

DATASET_PATH = "data/training_dataset.csv"
MODEL_PATH = "models/live_goal_model.pkl"
METRICS_PATH = "models/live_goal_model_metrics.json"
FEATURE_IMPORTANCE_PATH = "models/live_goal_model_feature_importance.json"
REPORT_PATH = "models/train_validation_report.md"

TARGET_COLUMN = "goal_in_next_15m"
GROUP_COLUMN = "match_id"

FEATURES = [
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
    "possession",
]

RANDOM_STATE = 42
TEST_SIZE = 0.25

SINGLE_CLASS_WARNING = "WARNING: Test set contains only one class. Skipping ROC-AUC."


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    assert GROUP_COLUMN not in FEATURES, "match_id nao pode ser usado como feature"
    X = df[FEATURES]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]
    return X, y, groups


def group_split(X, y, groups):
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def report_split_stats(groups_train, groups_test, X_train, X_test):
    matches_train = set(groups_train.unique())
    matches_test = set(groups_test.unique())
    overlap = matches_train & matches_test

    stats = {
        "n_matches_train": len(matches_train),
        "n_matches_test": len(matches_test),
        "n_snapshots_train": len(X_train),
        "n_snapshots_test": len(X_test),
        "match_id_overlap": len(overlap),
    }

    print("=" * 60)
    print("VALIDACAO DO SPLIT (GroupShuffleSplit por match_id)")
    print("=" * 60)
    print(f"Numero de jogos no treino: {stats['n_matches_train']}")
    print(f"Numero de jogos no teste: {stats['n_matches_test']}")
    print(f"Numero de snapshots no treino: {stats['n_snapshots_train']}")
    print(f"Numero de snapshots no teste: {stats['n_snapshots_test']}")
    print(f"Numero de match_id repetidos entre treino e teste: {stats['match_id_overlap']}")
    print()

    assert stats["match_id_overlap"] == 0, (
        "Group leakage detectado: existem match_id presentes simultaneamente "
        "no treino e no teste."
    )

    return stats


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "BalancedRandomForest": BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            class_weight="balanced",
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
    }


def feature_importance_for(name, model, X_test, y_test, scoring="roc_auc"):
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        return {feat: float(val) for feat, val in zip(FEATURES, importances)}

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring=scoring,
    )
    return {feat: float(val) for feat, val in zip(FEATURES, result.importances_mean)}


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    results = {}
    fallback_metrics = {}
    importances = {}
    fitted_models = {}

    print("=" * 60)
    print("TREINO E AVALIACAO DOS MODELOS")
    print("=" * 60)

    # roc_auc_score (e a metrica "roc_auc" usada na permutation_importance)
    # exigem as duas classes presentes em y_test. Com GroupShuffleSplit, o
    # teste pode calhar so com uma classe, o que antes fazia o treino
    # rebentar. Aqui deteta-se isso uma unica vez (a condicao e global, o
    # mesmo y_test serve para todos os modelos) e usa-se um caminho
    # alternativo que nao depende de ROC-AUC.
    single_class_test = len(np.unique(y_test)) < 2
    if single_class_test:
        print(SINGLE_CLASS_WARNING)

    importance_scoring = "accuracy" if single_class_test else "roc_auc"

    for name, model in models.items():
        model.fit(X_train, y_train)

        if single_class_test:
            pred_label = model.predict(X_test)
            results[name] = None
            fallback_metrics[name] = {
                "accuracy": float(accuracy_score(y_test, pred_label)),
                "precision": float(precision_score(y_test, pred_label, zero_division=0)),
                "recall": float(recall_score(y_test, pred_label, zero_division=0)),
                "f1": float(f1_score(y_test, pred_label, zero_division=0)),
            }
            m = fallback_metrics[name]
            print(
                f"{name:25s} AUC=N/A Accuracy={m['accuracy']:.4f} "
                f"Precision={m['precision']:.4f} Recall={m['recall']:.4f} F1={m['f1']:.4f}"
            )
        else:
            pred = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, pred)
            results[name] = float(auc)
            print(f"{name:25s} AUC={auc:.4f}")

        importances[name] = feature_importance_for(
            name, model, X_test, y_test, scoring=importance_scoring
        )
        fitted_models[name] = model

    print()
    return results, fallback_metrics, importances, fitted_models, single_class_test


def write_metrics(split_stats, results, fallback_metrics, single_class_test, best_model_name):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = {
        **split_stats,
        "test_set_single_class": single_class_test,
    }
    if single_class_test:
        metrics["warning"] = SINGLE_CLASS_WARNING
        metrics["models_auc"] = {name: None for name in fallback_metrics}
        metrics["models_fallback_metrics"] = fallback_metrics
        metrics["best_model"] = best_model_name
        metrics["best_model_selection_metric"] = "f1"
        metrics["best_model_f1"] = fallback_metrics[best_model_name]["f1"]
    else:
        metrics["models_auc"] = results
        metrics["best_model"] = best_model_name
        metrics["best_model_selection_metric"] = "auc"
        metrics["best_model_auc"] = results[best_model_name]

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metricas guardadas em: {METRICS_PATH}")


def write_feature_importance(importances):
    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PATH), exist_ok=True)
    with open(FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2, ensure_ascii=False)
    print(f"Importancia de features guardada em: {FEATURE_IMPORTANCE_PATH}")


def write_validation_report(split_stats, results, fallback_metrics, single_class_test, best_model_name):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    lines = []
    lines.append("# Relatorio de Validacao do Treino (GroupShuffleSplit por match_id)")
    lines.append("")
    lines.append(
        "Split de treino/teste agrupado por `match_id`, garantindo que nenhum jogo "
        "tem snapshots simultaneamente no treino e no teste."
    )
    lines.append("")
    lines.append("## Estatisticas do split")
    lines.append("")
    lines.append(f"- Numero de jogos no treino: {split_stats['n_matches_train']}")
    lines.append(f"- Numero de jogos no teste: {split_stats['n_matches_test']}")
    lines.append(f"- Numero de snapshots no treino: {split_stats['n_snapshots_train']}")
    lines.append(f"- Numero de snapshots no teste: {split_stats['n_snapshots_test']}")
    lines.append(f"- Overlap de match_id entre treino e teste: {split_stats['match_id_overlap']}")
    lines.append("")

    if single_class_test:
        lines.append(f"> {SINGLE_CLASS_WARNING}")
        lines.append("")
        lines.append("## Metricas por modelo (ROC-AUC indisponivel)")
        lines.append("")
        lines.append("| Modelo | Accuracy | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|---|")
        sorted_fallback = sorted(
            fallback_metrics.items(), key=lambda item: -item[1]["f1"]
        )
        for name, m in sorted_fallback:
            marker = " (melhor modelo)" if name == best_model_name else ""
            lines.append(
                f"| {name}{marker} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                f"{m['recall']:.4f} | {m['f1']:.4f} |"
            )
        lines.append("")
        lines.append("## Melhor modelo")
        lines.append("")
        lines.append(
            f"`{best_model_name}` com F1 = {fallback_metrics[best_model_name]['f1']:.4f} "
            "(selecionado por F1 porque o conjunto de teste so contem uma classe; AUC indisponivel)"
        )
        lines.append("")
    else:
        sorted_results = sorted(results.items(), key=lambda item: -item[1])

        lines.append("## AUC por modelo")
        lines.append("")
        lines.append("| Modelo | AUC |")
        lines.append("|---|---|")
        for name, auc in sorted_results:
            marker = " (melhor modelo)" if name == best_model_name else ""
            lines.append(f"| {name}{marker} | {auc:.4f} |")
        lines.append("")

        lines.append("## Melhor modelo")
        lines.append("")
        lines.append(f"`{best_model_name}` com AUC = {results[best_model_name]:.4f}")
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Relatorio de validacao guardado em: {REPORT_PATH}")


def main():
    X, y, groups = load_dataset()

    X_train, X_test, y_train, y_test, groups_train, groups_test = group_split(
        X, y, groups
    )

    split_stats = report_split_stats(groups_train, groups_test, X_train, X_test)

    models = build_models()
    results, fallback_metrics, importances, fitted_models, single_class_test = (
        train_and_evaluate(models, X_train, y_train, X_test, y_test)
    )

    if single_class_test:
        best_model_name = max(
            fallback_metrics, key=lambda name: fallback_metrics[name]["f1"]
        )
        print(
            f"Melhor modelo: {best_model_name} "
            f"(F1={fallback_metrics[best_model_name]['f1']:.4f}, AUC indisponivel)"
        )
    else:
        best_model_name = max(results, key=results.get)
        print(f"Melhor modelo: {best_model_name} (AUC={results[best_model_name]:.4f})")
    print()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(fitted_models[best_model_name], MODEL_PATH)
    print(f"Modelo guardado em: {MODEL_PATH}")

    write_metrics(split_stats, results, fallback_metrics, single_class_test, best_model_name)
    write_feature_importance(importances)
    write_validation_report(
        split_stats, results, fallback_metrics, single_class_test, best_model_name
    )


if __name__ == "__main__":
    main()
