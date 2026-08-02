import json
import os
import sys

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
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False

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
N_SPLITS = 5

SINGLE_CLASS_TEST_WARNING = (
    "WARNING: Fold {fold} test set contains only one class. "
    "Skipping ROC-AUC for this fold."
)
SINGLE_CLASS_TRAIN_WARNING = (
    "WARNING: Fold {fold} training set contains only one class "
    "(model cannot produce a positive-class probability). "
    "Skipping ROC-AUC for this fold."
)


class NoValidModelError(RuntimeError):
    """Nenhum modelo produziu um unico fold valido em toda a validacao cruzada."""


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    assert GROUP_COLUMN not in FEATURES, "match_id nao pode ser usado como feature"
    X = df[FEATURES]
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]
    return X, y, groups


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


def build_cv_splitter():
    """
    Devolve (splitter, nome) para a validacao cruzada agrupada por match_id.
    Preferencia por StratifiedGroupKFold (estratifica goal_in_next_15m
    respeitando os grupos); cai para GroupKFold se a versao do scikit-learn
    instalada nao tiver StratifiedGroupKFold, ou se essa versao nao aceitar
    shuffle/random_state.
    """
    if HAS_STRATIFIED_GROUP_KFOLD:
        try:
            return (
                StratifiedGroupKFold(
                    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
                ),
                "StratifiedGroupKFold",
            )
        except TypeError:
            return StratifiedGroupKFold(n_splits=N_SPLITS), "StratifiedGroupKFold"

    try:
        return (
            GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
            "GroupKFold",
        )
    except TypeError:
        return GroupKFold(n_splits=N_SPLITS), "GroupKFold"


def feature_importance_for(model, X_test, y_test, scoring="roc_auc"):
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


def _short_error(exc):
    message = str(exc).strip() or repr(exc)
    return message if len(message) <= 200 else message[:200] + "..."


def _warn_operation_failure(model_name, fold_idx, operation, exc):
    reason = f"{operation}: {_short_error(exc)}"
    print(f"WARNING: model={model_name} fold={fold_idx} operation={operation} failed: {reason}")
    return reason


def evaluate_fold(model, model_name, X_train, y_train, X_test, y_test, fold_idx, skip_auc_reason):
    """
    Avalia um modelo num fold isolando cada etapa: uma excecao em qualquer
    etapa nunca propaga para fora desta funcao. fit()/predict() falharem
    invalida o fold inteiro para este modelo (sem previsoes nao ha metricas
    possiveis); predict_proba()/AUC e a importancia de features falharem
    nao invalidam o fold - ficam apenas marcados como indisponiveis.
    """
    result = {
        "fold": fold_idx,
        "valid": True,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "auc": None,
        "failure_reason": None,
        "auc_skipped_reason": None,
    }
    importance = None

    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = _warn_operation_failure(model_name, fold_idx, "fit", exc)
        return result, importance

    try:
        pred_label = model.predict(X_test)
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = _warn_operation_failure(model_name, fold_idx, "predict", exc)
        return result, importance

    try:
        result["accuracy"] = float(accuracy_score(y_test, pred_label))
        result["precision"] = float(precision_score(y_test, pred_label, zero_division=0))
        result["recall"] = float(recall_score(y_test, pred_label, zero_division=0))
        result["f1"] = float(f1_score(y_test, pred_label, zero_division=0))
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = _warn_operation_failure(
            model_name, fold_idx, "classification_metrics", exc
        )
        return result, importance

    if skip_auc_reason is not None:
        result["auc_skipped_reason"] = skip_auc_reason
    else:
        try:
            pred_proba = model.predict_proba(X_test)[:, 1]
            result["auc"] = float(roc_auc_score(y_test, pred_proba))
        except Exception as exc:
            result["auc_skipped_reason"] = _warn_operation_failure(
                model_name, fold_idx, "roc_auc", exc
            )

    importance_scoring = "accuracy" if result["auc"] is None else "roc_auc"
    try:
        importance = feature_importance_for(model, X_test, y_test, scoring=importance_scoring)
    except Exception as exc:
        _warn_operation_failure(model_name, fold_idx, "feature_importance", exc)
        importance = None

    return result, importance


def cross_validate(X, y, groups):
    splitter, splitter_name = build_cv_splitter()
    fold_splits = list(splitter.split(X, y, groups=groups))
    n_splits = len(fold_splits)

    model_names = list(build_models().keys())
    fold_reports = []
    per_model_fold_metrics = {name: [] for name in model_names}
    per_model_fold_importance = {name: [] for name in model_names}

    print("=" * 60)
    print(
        f"VALIDACAO CRUZADA ({splitter_name}, n_splits={n_splits}, "
        "agrupada por match_id)"
    )
    print("=" * 60)

    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits, start=1):
        # Cada fold e completamente independente: X/y/groups sao fatiados de
        # novo a partir dos indices deste fold, e build_models() (chamado
        # abaixo) devolve instancias novas e nao treinadas - nada e
        # partilhado entre folds nem entre modelos.
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

        train_matches = set(groups_train.unique())
        test_matches = set(groups_test.unique())
        overlap = train_matches & test_matches

        single_class_test = len(np.unique(y_test)) < 2
        single_class_train = len(np.unique(y_train)) < 2

        skip_auc_reason = None
        if single_class_test:
            skip_auc_reason = "single_class_test"
        elif single_class_train:
            skip_auc_reason = "single_class_train"

        fold_report = {
            "fold": fold_idx,
            "n_matches_train": len(train_matches),
            "n_matches_test": len(test_matches),
            "n_snapshots_train": len(X_train),
            "n_snapshots_test": len(X_test),
            "match_id_overlap": len(overlap),
            "test_set_single_class": single_class_test,
            "train_set_single_class": single_class_train,
        }
        fold_reports.append(fold_report)

        print(
            f"Fold {fold_idx}: jogos treino={fold_report['n_matches_train']} "
            f"jogos teste={fold_report['n_matches_test']} "
            f"snapshots treino={fold_report['n_snapshots_train']} "
            f"snapshots teste={fold_report['n_snapshots_test']} "
            f"overlap match_id={fold_report['match_id_overlap']}"
        )

        # Garantia estrutural de nao-leakage: isto nao e uma falha "por
        # modelo/fold" recuperavel, e uma violacao do invariante que motivou
        # todo este pipeline - mantem-se um hard-stop.
        assert fold_report["match_id_overlap"] == 0, (
            f"Group leakage detectado no fold {fold_idx}: existem match_id "
            "presentes simultaneamente no treino e no teste."
        )

        if single_class_test:
            print(SINGLE_CLASS_TEST_WARNING.format(fold=fold_idx))
        if single_class_train:
            print(SINGLE_CLASS_TRAIN_WARNING.format(fold=fold_idx))

        models = build_models()
        for name, model in models.items():
            metrics, importance = evaluate_fold(
                model, name, X_train, y_train, X_test, y_test, fold_idx, skip_auc_reason
            )
            per_model_fold_metrics[name].append(metrics)
            per_model_fold_importance[name].append(importance)

    print()
    return (
        splitter_name,
        n_splits,
        fold_reports,
        per_model_fold_metrics,
        per_model_fold_importance,
    )


def aggregate_metrics(per_model_fold_metrics):
    summary = {}
    excluded_models = {}

    for name, fold_metrics in per_model_fold_metrics.items():
        valid_folds = [fm for fm in fold_metrics if fm["valid"]]
        invalid_folds = [fm for fm in fold_metrics if not fm["valid"]]
        invalid_reasons = [
            {"fold": fm["fold"], "reason": fm["failure_reason"]} for fm in invalid_folds
        ]

        if not valid_folds:
            excluded_models[name] = {
                "n_folds_total": len(fold_metrics),
                "n_folds_valid": 0,
                "n_folds_invalid": len(invalid_folds),
                "invalid_fold_reasons": invalid_reasons,
            }
            summary[name] = None
            continue

        agg = {
            "n_folds_total": len(fold_metrics),
            "n_folds_valid": len(valid_folds),
            "n_folds_invalid": len(invalid_folds),
            "invalid_fold_reasons": invalid_reasons,
        }
        for metric in ("accuracy", "precision", "recall", "f1"):
            values = [fm[metric] for fm in valid_folds]
            agg[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        auc_values = [fm["auc"] for fm in valid_folds if fm["auc"] is not None]
        if auc_values:
            agg["auc"] = {
                "mean": float(np.mean(auc_values)),
                "std": float(np.std(auc_values)),
                "n_folds": len(auc_values),
            }
        else:
            agg["auc"] = {"mean": None, "std": None, "n_folds": 0}

        summary[name] = agg

    return summary, excluded_models


def aggregate_importance(per_model_fold_importance):
    aggregated = {}
    for name, fold_importances in per_model_fold_importance.items():
        valid_importances = [fi for fi in fold_importances if fi is not None]
        if not valid_importances:
            aggregated[name] = None
            continue

        feature_values = {feat: [] for feat in FEATURES}
        for fold_importance in valid_importances:
            for feat, val in fold_importance.items():
                feature_values[feat].append(val)
        aggregated[name] = {
            feat: float(np.mean(vals)) for feat, vals in feature_values.items()
        }
    return aggregated


def select_best_model(metrics_summary):
    available = {name: summary for name, summary in metrics_summary.items() if summary is not None}

    if not available:
        raise NoValidModelError(
            "Nenhum modelo produziu um unico fold valido em toda a validacao cruzada."
        )

    any_auc_available = any(
        summary["auc"]["mean"] is not None for summary in available.values()
    )

    if any_auc_available:
        scored = {
            name: summary["auc"]["mean"]
            for name, summary in available.items()
            if summary["auc"]["mean"] is not None
        }
        best_model_name = max(scored, key=scored.get)
        return best_model_name, "auc_mean", scored[best_model_name]

    scored = {name: summary["f1"]["mean"] for name, summary in available.items()}
    best_model_name = max(scored, key=scored.get)
    return best_model_name, "f1_mean", scored[best_model_name]


def write_metrics(
    splitter_name,
    n_splits,
    n_matches_total,
    n_snapshots_total,
    fold_reports,
    per_model_fold_metrics,
    metrics_summary,
    excluded_models,
    best_model_name,
    selection_metric,
    best_score,
):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = {
        "cv_method": splitter_name,
        "n_splits": n_splits,
        "n_matches_total": n_matches_total,
        "n_snapshots_total": n_snapshots_total,
        "folds": fold_reports,
        "models": {
            name: {
                "per_fold": per_model_fold_metrics[name],
                "mean": metrics_summary.get(name),
            }
            for name in per_model_fold_metrics
        },
        "excluded_models": excluded_models,
        "best_model": best_model_name,
        "best_model_selection_metric": selection_metric,
        "best_model_score": best_score,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metricas guardadas em: {METRICS_PATH}")


def write_feature_importance(importances):
    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PATH), exist_ok=True)
    with open(FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2, ensure_ascii=False)
    print(f"Importancia de features guardada em: {FEATURE_IMPORTANCE_PATH}")


def write_validation_report(
    splitter_name,
    n_splits,
    n_matches_total,
    n_snapshots_total,
    fold_reports,
    per_model_fold_metrics,
    metrics_summary,
    excluded_models,
    best_model_name,
    selection_metric,
    best_score,
):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    def fmt(stat):
        return f"{stat['mean']:.4f} +/- {stat['std']:.4f}"

    lines = []
    lines.append("# Relatorio de Validacao Cruzada (Group K-Fold por match_id)")
    lines.append("")
    lines.append(
        f"Metodo de validacao cruzada: `{splitter_name}` com {n_splits} folds, "
        "agrupados por `match_id` (nenhum jogo aparece simultaneamente em treino "
        "e teste em nenhum fold)."
    )
    lines.append("")
    lines.append(f"- Total de jogos no dataset: {n_matches_total}")
    lines.append(f"- Total de snapshots no dataset: {n_snapshots_total}")
    lines.append("")

    lines.append("## Estatisticas por fold")
    lines.append("")
    lines.append(
        "| Fold | Jogos treino | Jogos teste | Snapshots treino | Snapshots teste "
        "| Overlap match_id | Teste com 1 classe | Treino com 1 classe |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for fr in fold_reports:
        single_class_flag = "SIM (AUC ignorada)" if fr["test_set_single_class"] else "Nao"
        train_single_class_flag = (
            "SIM (AUC ignorada)" if fr["train_set_single_class"] else "Nao"
        )
        lines.append(
            f"| {fr['fold']} | {fr['n_matches_train']} | {fr['n_matches_test']} | "
            f"{fr['n_snapshots_train']} | {fr['n_snapshots_test']} | "
            f"{fr['match_id_overlap']} | {single_class_flag} | {train_single_class_flag} |"
        )
    lines.append("")

    if excluded_models:
        lines.append("## Modelos excluidos da comparacao")
        lines.append("")
        lines.append(
            "Estes modelos nao produziram um unico fold valido e foram excluidos "
            "da selecao do melhor modelo."
        )
        lines.append("")
        for name, info in excluded_models.items():
            lines.append(f"### {name}")
            lines.append("")
            lines.append(
                f"- Folds validos: {info['n_folds_valid']}/{info['n_folds_total']} "
                f"(ignorados: {info['n_folds_invalid']})"
            )
            for reason in info["invalid_fold_reasons"]:
                lines.append(f"  - Fold {reason['fold']}: {reason['reason']}")
            lines.append("")

    if not metrics_summary:
        lines.append("## Resultado")
        lines.append("")
        lines.append(
            "Nenhum modelo produziu um unico fold valido em toda a validacao "
            "cruzada. Nenhum modelo foi treinado ou guardado."
        )
        lines.append("")
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Relatorio de validacao guardado em: {REPORT_PATH}")
        return

    lines.append("## Metricas medias por modelo (media +/- desvio-padrao entre folds validos)")
    lines.append("")
    lines.append(
        "| Modelo | Folds validos | Folds ignorados | Accuracy | Precision | Recall | F1 | ROC-AUC |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    sort_key = "auc" if selection_metric == "auc_mean" else "f1"
    available_summaries = [
        (name, summary) for name, summary in metrics_summary.items() if summary is not None
    ]
    sorted_models = sorted(
        available_summaries,
        key=lambda item: -(
            item[1][sort_key]["mean"] if item[1][sort_key]["mean"] is not None else -1
        ),
    )
    for name, summary in sorted_models:
        marker = " (melhor modelo)" if name == best_model_name else ""
        auc_stat = summary["auc"]
        auc_str = (
            f"{auc_stat['mean']:.4f} +/- {auc_stat['std']:.4f} (n={auc_stat['n_folds']})"
            if auc_stat["mean"] is not None
            else "N/A"
        )
        lines.append(
            f"| {name}{marker} | {summary['n_folds_valid']} | {summary['n_folds_invalid']} | "
            f"{fmt(summary['accuracy'])} | {fmt(summary['precision'])} | "
            f"{fmt(summary['recall'])} | {fmt(summary['f1'])} | {auc_str} |"
        )
    lines.append("")

    lines.append("## Melhor modelo")
    lines.append("")
    criterio = "ROC-AUC media" if selection_metric == "auc_mean" else "F1 media"
    lines.append(
        f"`{best_model_name}` — selecionado por {criterio} = {best_score:.4f}"
        + (
            ""
            if selection_metric == "auc_mean"
            else " (ROC-AUC nao pode ser calculada em nenhum fold valido)"
        )
    )
    lines.append("")

    lines.append("## Metricas por fold (detalhe)")
    lines.append("")
    for name in per_model_fold_metrics:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Fold | Valido | Accuracy | Precision | Recall | F1 | ROC-AUC | Motivo |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for fm in per_model_fold_metrics[name]:
            if not fm["valid"]:
                lines.append(
                    f"| {fm['fold']} | Nao | - | - | - | - | - | {fm['failure_reason']} |"
                )
                continue
            auc_cell = f"{fm['auc']:.4f}" if fm["auc"] is not None else "N/A"
            motivo = fm["auc_skipped_reason"] or ""
            lines.append(
                f"| {fm['fold']} | Sim | {fm['accuracy']:.4f} | {fm['precision']:.4f} | "
                f"{fm['recall']:.4f} | {fm['f1']:.4f} | {auc_cell} | {motivo} |"
            )
        lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Relatorio de validacao guardado em: {REPORT_PATH}")


def main():
    X, y, groups = load_dataset()

    n_matches_total = int(groups.nunique())
    n_snapshots_total = int(len(X))

    (
        splitter_name,
        n_splits,
        fold_reports,
        per_model_fold_metrics,
        per_model_fold_importance,
    ) = cross_validate(X, y, groups)

    metrics_summary, excluded_models = aggregate_metrics(per_model_fold_metrics)
    importance_summary = aggregate_importance(per_model_fold_importance)

    if excluded_models:
        print("=" * 60)
        print("MODELOS EXCLUIDOS (sem nenhum fold valido)")
        print("=" * 60)
        for name, info in excluded_models.items():
            print(f"{name}: {info['n_folds_invalid']}/{info['n_folds_total']} folds falharam")
        print()

    try:
        best_model_name, selection_metric, best_score = select_best_model(metrics_summary)
    except NoValidModelError as exc:
        print("=" * 60)
        print(f"ERRO: {exc}")
        print("Nenhum modelo foi treinado nem guardado.")
        print("=" * 60)

        write_metrics(
            splitter_name,
            n_splits,
            n_matches_total,
            n_snapshots_total,
            fold_reports,
            per_model_fold_metrics,
            {},
            excluded_models,
            None,
            None,
            None,
        )
        write_feature_importance(importance_summary)
        write_validation_report(
            splitter_name,
            n_splits,
            n_matches_total,
            n_snapshots_total,
            fold_reports,
            per_model_fold_metrics,
            {},
            excluded_models,
            None,
            None,
            None,
        )
        sys.exit(1)

    print("=" * 60)
    print("RESUMO DA VALIDACAO CRUZADA (media +/- desvio-padrao entre folds validos)")
    print("=" * 60)
    for name, summary in metrics_summary.items():
        if summary is None:
            continue
        auc_stat = summary["auc"]
        auc_str = (
            f"{auc_stat['mean']:.4f}+/-{auc_stat['std']:.4f}(n={auc_stat['n_folds']})"
            if auc_stat["mean"] is not None
            else "N/A"
        )
        print(
            f"{name:25s} folds_validos={summary['n_folds_valid']}/{summary['n_folds_total']} "
            f"AUC={auc_str:26s} "
            f"Accuracy={summary['accuracy']['mean']:.4f}+/-{summary['accuracy']['std']:.4f} "
            f"Precision={summary['precision']['mean']:.4f}+/-{summary['precision']['std']:.4f} "
            f"Recall={summary['recall']['mean']:.4f}+/-{summary['recall']['std']:.4f} "
            f"F1={summary['f1']['mean']:.4f}+/-{summary['f1']['std']:.4f}"
        )
    print()

    criterio = "ROC-AUC media" if selection_metric == "auc_mean" else "F1 media"
    print(f"Melhor modelo: {best_model_name} (criterio={criterio}, score={best_score:.4f})")
    print()

    # Retreino final do modelo vencedor com 100% dos dados disponiveis
    # (mesma classe e mesmos hiperparametros de build_models()). So chega
    # aqui um modelo com pelo menos um fold valido na validacao cruzada.
    # Protegido tambem por try/except: se mesmo assim falhar, nao gravamos
    # um .pkl parcial e terminamos com erro claro (regra 6/7 do pedido).
    final_models = build_models()
    final_model = final_models[best_model_name]
    try:
        final_model.fit(X, y)
    except Exception as exc:
        print("=" * 60)
        print(
            f"ERRO: retreino final de {best_model_name} com 100% dos dados falhou: "
            f"{_short_error(exc)}"
        )
        print("Nenhum modelo foi guardado.")
        print("=" * 60)

        write_metrics(
            splitter_name,
            n_splits,
            n_matches_total,
            n_snapshots_total,
            fold_reports,
            per_model_fold_metrics,
            metrics_summary,
            excluded_models,
            best_model_name,
            selection_metric,
            best_score,
        )
        write_feature_importance(importance_summary)
        write_validation_report(
            splitter_name,
            n_splits,
            n_matches_total,
            n_snapshots_total,
            fold_reports,
            per_model_fold_metrics,
            metrics_summary,
            excluded_models,
            best_model_name,
            selection_metric,
            best_score,
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Modelo vencedor re-treinado com 100% dos dados e guardado em: {MODEL_PATH}")

    write_metrics(
        splitter_name,
        n_splits,
        n_matches_total,
        n_snapshots_total,
        fold_reports,
        per_model_fold_metrics,
        metrics_summary,
        excluded_models,
        best_model_name,
        selection_metric,
        best_score,
    )
    write_feature_importance(importance_summary)
    write_validation_report(
        splitter_name,
        n_splits,
        n_matches_total,
        n_snapshots_total,
        fold_reports,
        per_model_fold_metrics,
        metrics_summary,
        excluded_models,
        best_model_name,
        selection_metric,
        best_score,
    )


if __name__ == "__main__":
    main()
