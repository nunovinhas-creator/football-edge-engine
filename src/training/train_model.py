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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
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
THRESHOLD_CONFIG_PATH = "models/live_goal_model_threshold.json"
THRESHOLD_REPORT_PATH = "models/threshold_optimization_report.md"
CALIBRATION_METRICS_PATH = "models/calibration_metrics.json"
CALIBRATION_REPORT_PATH = "models/calibration_report.md"

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

# --- Calibracao de probabilidades ---
# Metodo usado por CalibratedClassifierCV. Unico local de configuracao:
# "sigmoid" (Platt Scaling) ou "isotonic".
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_VERDICT_TOLERANCE = 0.01

# --- Otimizacao do threshold de decisao ---
# Metrica usada para escolher o threshold otimo. Alterar aqui para mudar a
# metrica sem tocar no resto do pipeline. Metricas disponiveis: "f1",
# "precision", "recall", "balanced_accuracy", "youden_j" (definidas em
# THRESHOLD_METRICS, que pode ser estendido com qualquer outra metrica do
# scikit-learn que aceite (y_true, y_pred)).
THRESHOLD_OPTIMIZATION_METRIC = "f1"
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 4)


def _youden_j_score(y_true, y_pred):
    # Youden's J = sensibilidade + especificidade - 1 = 2*balanced_accuracy - 1
    return 2.0 * balanced_accuracy_score(y_true, y_pred) - 1.0


THRESHOLD_METRICS = {
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score,
    "youden_j": _youden_j_score,
}

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


def _select_valid_folds(model_name, X, y, groups):
    """
    Calcula os folds de build_cv_splitter() (inalterada) e filtra os que
    tem y_train com uma so classe - condicao que impede qualquer modelo de
    produzir uma probabilidade de classe positiva valida nesse fold (mesma
    condicao que ja existia, em linha, dentro de compute_oof_probabilities).

    Usada tanto por compute_oof_probabilities() como por
    calibrate_best_model(), para que as previsoes out-of-fold brutas e
    calibradas sejam sempre calculadas sobre exatamente o mesmo conjunto de
    folds - por construcao (mesma funcao, mesmos argumentos), nao por
    coincidencia entre duas implementacoes separadas. Isto e o que garante,
    a montante, que a comparacao "antes vs depois" no relatorio de
    calibracao seja valida (ver tambem _oof_populations_comparable(), que
    verifica isto explicitamente a jusante).

    Devolve (valid_fold_splits, n_total_folds, n_skipped_folds). Quando
    todos os folds sao validos (caso normal), valid_fold_splits tem o mesmo
    conteudo que build_cv_splitter().split(...) devolveria diretamente -
    nenhum comportamento muda face ao anterior.
    """
    splitter, _ = build_cv_splitter()
    all_fold_splits = list(splitter.split(X, y, groups=groups))

    valid_fold_splits = [
        (train_idx, test_idx)
        for train_idx, test_idx in all_fold_splits
        if len(np.unique(y.iloc[train_idx])) >= 2
    ]

    n_total = len(all_fold_splits)
    n_skipped = n_total - len(valid_fold_splits)
    if n_skipped > 0:
        print(
            f"WARNING: model={model_name} operation=fold_selection "
            f"{n_skipped}/{n_total} fold(s) ignorado(s) (treino com uma so "
            "classe nesse fold) - usados apenas os restantes."
        )

    return valid_fold_splits, n_total, n_skipped


def compute_oof_probabilities(model_name, X, y, groups):
    """
    Recalcula previsoes out-of-fold para um unico modelo, usando exatamente
    o mesmo splitter e random_state da validacao cruzada (build_cv_splitter()
    e build_models(), ambas inalteradas) - reproduz os mesmos folds ja
    usados em cross_validate(). Cada previsao devolvida vem sempre de uma
    instancia treinada sem os dados desse fold, pelo que o conjunto agregado
    e seguro para otimizar o threshold (nunca usa o modelo final treinado
    com 100% dos dados). Nao interfere em cross_validate(), aggregate_metrics()
    nem select_best_model() - e um recalculo isolado, so para este fim.
    """
    valid_fold_splits, _, _ = _select_valid_folds(model_name, X, y, groups)

    y_true_all = []
    y_proba_all = []

    for train_idx, test_idx in valid_fold_splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = build_models()[model_name]
        try:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
        except Exception as exc:
            print(
                f"WARNING: model={model_name} operation=oof_predict_proba "
                f"failed: {_short_error(exc)} (fold ignorado na otimizacao do threshold)"
            )
            continue

        y_true_all.extend(y_test.tolist())
        y_proba_all.extend(proba.tolist())

    return np.array(y_true_all), np.array(y_proba_all)


def calibrate_best_model(model_name, X, y, groups):
    """
    Calibra as probabilidades do modelo vencedor com CalibratedClassifierCV
    em modo "cv" (ensemble=True), usando exatamente os mesmos folds da
    validacao cruzada (build_cv_splitter(), inalterada). Neste modo, cada
    membro do ensemble e treinado APENAS na porcao de treino do seu fold e
    calibrado APENAS na porcao de teste desse mesmo fold - nunca no
    conjunto usado para o treinar. Nao usa nem altera cross_validate(),
    aggregate_metrics() ou select_best_model(); e uma camada adicional.

    Antes de calibrar, filtra (via _select_valid_folds(), a mesma funcao
    usada por compute_oof_probabilities()) qualquer fold cujo treino tenha
    uma so classe. Esses folds nunca chegam a CalibratedClassifierCV, para
    que nenhum membro do ensemble seja treinado/calibrado com dados
    degenerados - nunca havera um membro que devolva probabilidades
    constantes por falta de classes.

    Devolve (calibrated_model, valid_fold_splits) se a calibracao for bem
    sucedida, ou (None, valid_fold_splits) se falhar (incluindo o caso de
    nao existir nenhum fold valido) - nesse caso o chamador deve usar o
    modelo original sem calibrar (regra 7).
    """
    valid_fold_splits, n_total, n_skipped = _select_valid_folds(model_name, X, y, groups)

    if not valid_fold_splits:
        print(
            f"WARNING: model={model_name} operation=calibration failed: "
            f"nenhum fold valido para calibrar (todos os {n_total} folds "
            "tem treino com uma so classe) - a usar o modelo original sem "
            "calibrar."
        )
        return None, valid_fold_splits

    raw_model = build_models()[model_name]
    try:
        calibrated_model = CalibratedClassifierCV(
            estimator=raw_model,
            method=CALIBRATION_METHOD,
            cv=valid_fold_splits,
            ensemble=True,
        )
        calibrated_model.fit(X, y)
    except Exception as exc:
        print(
            f"WARNING: model={model_name} operation=calibration failed: "
            f"{_short_error(exc)} - a usar o modelo original sem calibrar."
        )
        return None, valid_fold_splits

    return calibrated_model, valid_fold_splits


def compute_oof_probabilities_calibrated(calibrated_model, fold_splits, X, y):
    """
    Previsoes calibradas out-of-fold: cada amostra e avaliada apenas pelo
    membro do ensemble (de calibrate_best_model) cujo fold de calibracao a
    incluiu no conjunto de teste - nunca por um membro cuja base foi
    treinada com essa amostra. Mesma logica de seguranca de
    compute_oof_probabilities(), aplicada ao modelo ja calibrado.
    """
    y_true_all = []
    y_proba_all = []

    for member, (train_idx, test_idx) in zip(
        calibrated_model.calibrated_classifiers_, fold_splits
    ):
        try:
            proba = member.predict_proba(X.iloc[test_idx])[:, 1]
        except Exception as exc:
            print(
                f"WARNING: operation=oof_calibrated_predict_proba failed: "
                f"{_short_error(exc)} (fold ignorado)"
            )
            continue
        y_true_all.extend(y.iloc[test_idx].tolist())
        y_proba_all.extend(proba.tolist())

    return np.array(y_true_all), np.array(y_proba_all)


def _safe_brier_and_log_loss(y_true, y_proba):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None, None

    try:
        brier = float(brier_score_loss(y_true, y_proba))
    except Exception:
        brier = None

    try:
        ll = float(log_loss(y_true, y_proba, labels=[0, 1]))
    except Exception:
        ll = None

    return brier, ll


def _calibration_verdict(before, after, tol=CALIBRATION_VERDICT_TOLERANCE):
    if before is None or after is None:
        return None
    if before == 0:
        return "manteve" if after == 0 else "piorou"
    relative_change = (after - before) / before
    if relative_change < -tol:
        return "melhorou"
    if relative_change > tol:
        return "piorou"
    return "manteve"


def _oof_populations_comparable(raw_y_true, calibrated_y_true):
    """
    Verifica explicitamente se as previsoes OOF brutas ("antes") e
    calibradas ("depois") correspondem exatamente a mesma populacao de
    amostras, na mesma ordem - condicao necessaria para a comparacao
    Brier/Log Loss "antes vs depois" ser valida.

    Ambas sao construidas iterando os folds validos (_select_valid_folds)
    pela mesma ordem deterministica e extraindo y_test.tolist() por fold;
    por isso, sequencias de rotulos identicas em numero e valores implicam
    a mesma sequencia de amostras nas mesmas posicoes. Isto e verificado
    aqui em vez de apenas assumido, mesmo sabendo que ambos os lados usam
    _select_valid_folds() com os mesmos argumentos - protege contra
    qualquer divergencia residual (por exemplo, uma falha pontual de
    model.fit()/predict_proba() num fold que so afete um dos dois lados).

    Devolve (True, None) se comparavel, ou (False, motivo) caso contrario.
    """
    if len(raw_y_true) != len(calibrated_y_true):
        return False, (
            "numero de amostras out-of-fold diferente "
            f"(antes={len(raw_y_true)}, depois={len(calibrated_y_true)})"
        )
    if len(raw_y_true) == 0:
        return False, "nenhuma amostra out-of-fold disponivel em nenhum dos dois lados"
    if not np.array_equal(raw_y_true, calibrated_y_true):
        return False, (
            "sequencia de rotulos out-of-fold diferente entre o modelo "
            "original e o modelo calibrado (os folds/amostras usados nao "
            "coincidem)"
        )
    return True, None


def build_calibration_summary(
    model_name,
    calibration_succeeded,
    before_y_true,
    before_y_proba,
    after_y_true,
    after_y_proba,
):
    before_brier, before_log_loss = _safe_brier_and_log_loss(before_y_true, before_y_proba)
    summary = {
        "model_name": model_name,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_succeeded": calibration_succeeded,
        "before": {
            "n_oof_samples": int(len(before_y_true)),
            "brier_score": before_brier,
            "log_loss": before_log_loss,
        },
        "after": None,
        "oof_populations_comparable": None,
        "comparison_warning": None,
        "verdict_brier": None,
        "verdict_log_loss": None,
    }

    if not calibration_succeeded:
        return summary

    after_brier, after_log_loss = _safe_brier_and_log_loss(after_y_true, after_y_proba)
    summary["after"] = {
        "n_oof_samples": int(len(after_y_true)),
        "brier_score": after_brier,
        "log_loss": after_log_loss,
    }

    comparable, reason = _oof_populations_comparable(before_y_true, after_y_true)
    summary["oof_populations_comparable"] = comparable
    summary["comparison_warning"] = reason

    if comparable:
        summary["verdict_brier"] = _calibration_verdict(before_brier, after_brier)
        summary["verdict_log_loss"] = _calibration_verdict(before_log_loss, after_log_loss)
    else:
        print(
            f"WARNING: model={model_name} operation=calibration_comparison "
            f"not directly comparable: {reason}. O relatorio nao vai "
            "apresentar melhorou/piorou/manteve."
        )

    return summary


def write_calibration_metrics(summary):
    os.makedirs(os.path.dirname(CALIBRATION_METRICS_PATH), exist_ok=True)
    with open(CALIBRATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Metricas de calibracao guardadas em: {CALIBRATION_METRICS_PATH}")


def write_calibration_report(summary):
    os.makedirs(os.path.dirname(CALIBRATION_REPORT_PATH), exist_ok=True)

    def fmt(value):
        return f"{value:.4f}" if value is not None else "N/A"

    lines = []
    lines.append("# Relatorio de Calibracao de Probabilidades")
    lines.append("")
    lines.append(f"Modelo: `{summary['model_name']}`")
    lines.append(f"Metodo: `{summary['calibration_method']}`")
    lines.append(
        f"Calibracao bem sucedida: {'Sim' if summary['calibration_succeeded'] else 'Nao'}"
    )
    lines.append("")

    if not summary["calibration_succeeded"]:
        lines.append(
            "A calibracao falhou; o pipeline continuou automaticamente com o "
            "modelo original (sem calibrar). Ver o WARNING emitido no log de "
            "treino para o motivo exato da falha."
        )
        lines.append("")
        lines.append("## Brier Score / Log Loss (modelo original, out-of-fold)")
        lines.append("")
        lines.append(f"- Brier Score: {fmt(summary['before']['brier_score'])}")
        lines.append(f"- Log Loss: {fmt(summary['before']['log_loss'])}")
        lines.append(f"- Amostras out-of-fold: {summary['before']['n_oof_samples']}")
        lines.append("")
        with open(CALIBRATION_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Relatorio de calibracao guardado em: {CALIBRATION_REPORT_PATH}")
        return

    before = summary["before"]
    after = summary["after"]
    comparable = summary["oof_populations_comparable"]

    if comparable is False:
        lines.append(
            f"> AVISO: a comparacao antes/depois NAO e direta - "
            f"{summary['comparison_warning']}. Os valores abaixo sao "
            "apenas informativos; NAO representam a mesma populacao de "
            "amostras, por isso nao e apresentado nenhum veredito de "
            "melhorou/piorou/manteve."
        )
        lines.append("")

    def comentario(verdict):
        if verdict is not None:
            return verdict
        if comparable is False:
            return "Nao comparavel (ver aviso acima)"
        return "N/A"

    lines.append("## Brier Score (menor e melhor)")
    lines.append("")
    lines.append(f"- Antes (modelo original): {fmt(before['brier_score'])}")
    lines.append(f"- Depois (modelo calibrado): {fmt(after['brier_score'])}")
    lines.append(f"- Comentario: {comentario(summary['verdict_brier'])}")
    lines.append("")

    lines.append("## Log Loss (menor e melhor)")
    lines.append("")
    lines.append(f"- Antes (modelo original): {fmt(before['log_loss'])}")
    lines.append(f"- Depois (modelo calibrado): {fmt(after['log_loss'])}")
    lines.append(f"- Comentario: {comentario(summary['verdict_log_loss'])}")
    lines.append("")

    lines.append(
        f"Amostras out-of-fold usadas: {before['n_oof_samples']} (antes), "
        f"{after['n_oof_samples']} (depois)."
    )
    lines.append("")

    with open(CALIBRATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Relatorio de calibracao guardado em: {CALIBRATION_REPORT_PATH}")


def evaluate_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


def find_optimal_threshold(y_true, y_proba, metric_name=THRESHOLD_OPTIMIZATION_METRIC):
    """
    Procura, na grelha THRESHOLD_GRID (0.05 a 0.95, passo 0.01), o threshold
    que maximiza metric_name. Devolve None se os dados de validacao nao
    permitirem otimizar (sem amostras ou so uma classe presente) - nesse
    caso o chamador deve cair para DEFAULT_THRESHOLD.
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None

    metric_fn = THRESHOLD_METRICS[metric_name]

    best_threshold = None
    best_score = -np.inf
    scores_by_threshold = []

    for t in THRESHOLD_GRID:
        y_pred = (y_proba >= t).astype(int)
        score = float(metric_fn(y_true, y_pred))
        scores_by_threshold.append({"threshold": float(t), "score": score})
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return {
        "threshold": best_threshold,
        "metric": metric_name,
        "metric_value": best_score,
        "n_oof_samples": int(len(y_true)),
        "scores_by_threshold": scores_by_threshold,
    }


def write_threshold_config(threshold_result, model_name):
    os.makedirs(os.path.dirname(THRESHOLD_CONFIG_PATH), exist_ok=True)

    if threshold_result is None:
        config = {
            "threshold": DEFAULT_THRESHOLD,
            "optimization_metric": None,
            "optimization_metric_value": None,
            "model_name": model_name,
            "n_oof_samples": 0,
            "note": (
                "Otimizacao do threshold nao foi possivel (dados de validacao "
                "out-of-fold insuficientes ou com uma so classe). A usar o "
                f"threshold por omissao ({DEFAULT_THRESHOLD})."
            ),
        }
    else:
        config = {
            "threshold": threshold_result["threshold"],
            "optimization_metric": threshold_result["metric"],
            "optimization_metric_value": threshold_result["metric_value"],
            "model_name": model_name,
            "n_oof_samples": threshold_result["n_oof_samples"],
            "note": None,
        }

    with open(THRESHOLD_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Configuracao de threshold guardada em: {THRESHOLD_CONFIG_PATH}")


def write_threshold_report(threshold_result, model_name, y_true, y_proba):
    os.makedirs(os.path.dirname(THRESHOLD_REPORT_PATH), exist_ok=True)

    lines = []
    lines.append("# Relatorio de Otimizacao do Threshold de Decisao")
    lines.append("")
    lines.append(f"Modelo vencedor: `{model_name}`")
    lines.append("")

    if threshold_result is None:
        lines.append(
            "Nao foi possivel otimizar o threshold: os dados de validacao "
            "out-of-fold nao tinham amostras suficientes ou continham so "
            f"uma classe. A usar o threshold por omissao ({DEFAULT_THRESHOLD})."
        )
        lines.append("")
        with open(THRESHOLD_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Relatorio de threshold guardado em: {THRESHOLD_REPORT_PATH}")
        return

    optimal_threshold = threshold_result["threshold"]
    lines.append(f"Metrica de otimizacao: `{threshold_result['metric']}`")
    lines.append(f"Threshold otimo: {optimal_threshold:.4f}")
    lines.append(f"Valor da metrica no threshold otimo: {threshold_result['metric_value']:.4f}")
    lines.append(f"Amostras de validacao (out-of-fold) usadas: {threshold_result['n_oof_samples']}")
    lines.append("")

    default_eval = evaluate_at_threshold(y_true, y_proba, DEFAULT_THRESHOLD)
    optimal_eval = evaluate_at_threshold(y_true, y_proba, optimal_threshold)

    lines.append("## Comparacao: threshold 0.5 (default) vs threshold otimo")
    lines.append("")
    lines.append("| Threshold | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| 0.5 (default) | {default_eval['precision']:.4f} | "
        f"{default_eval['recall']:.4f} | {default_eval['f1']:.4f} |"
    )
    lines.append(
        f"| {optimal_threshold:.4f} (otimo) | {optimal_eval['precision']:.4f} | "
        f"{optimal_eval['recall']:.4f} | {optimal_eval['f1']:.4f} |"
    )
    lines.append("")

    def cm_table(title, cm):
        rows = [
            f"## Matriz de confusao — {title}",
            "",
            "| | Previsto: sem golo | Previsto: golo |",
            "|---|---|---|",
            f"| Real: sem golo | TN={cm['tn']} | FP={cm['fp']} |",
            f"| Real: golo | FN={cm['fn']} | TP={cm['tp']} |",
            "",
        ]
        return rows

    lines.extend(cm_table("threshold 0.5 (default)", default_eval["confusion_matrix"]))
    lines.extend(cm_table(f"threshold {optimal_threshold:.4f} (otimo)", optimal_eval["confusion_matrix"]))

    with open(THRESHOLD_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Relatorio de threshold guardado em: {THRESHOLD_REPORT_PATH}")


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

    # Calibracao das probabilidades do modelo vencedor - camada adicional,
    # nao interfere na validacao cruzada nem na selecao do melhor modelo
    # (ja concluidas acima). Ver calibrate_best_model().
    print("=" * 60)
    print(f"CALIBRACAO DE PROBABILIDADES (metodo={CALIBRATION_METHOD})")
    print("=" * 60)

    raw_oof_y_true, raw_oof_y_proba = compute_oof_probabilities(best_model_name, X, y, groups)
    calibrated_model, fold_splits = calibrate_best_model(best_model_name, X, y, groups)

    if calibrated_model is not None:
        calibrated_oof_y_true, calibrated_oof_y_proba = compute_oof_probabilities_calibrated(
            calibrated_model, fold_splits, X, y
        )
        calibration_succeeded = True
        final_model = calibrated_model
        threshold_y_true, threshold_y_proba = calibrated_oof_y_true, calibrated_oof_y_proba
        print(f"Calibracao concluida com sucesso ({CALIBRATION_METHOD}).")
    else:
        calibration_succeeded = False
        calibrated_oof_y_true, calibrated_oof_y_proba = None, None
        threshold_y_true, threshold_y_proba = raw_oof_y_true, raw_oof_y_proba
        print("A calibracao falhou - a usar o modelo original (nao calibrado).")

        # Fallback (regra 7): modelo original, re-treinado com 100% dos
        # dados disponiveis - exatamente o que o pipeline ja fazia antes
        # desta funcionalidade. Continua protegido: se ate isto falhar, nao
        # gravamos um .pkl parcial e terminamos com erro claro.
        final_model = build_models()[best_model_name]
        try:
            final_model.fit(X, y)
        except Exception as exc:
            print("=" * 60)
            print(
                f"ERRO: retreino do modelo original ({best_model_name}) com "
                f"100% dos dados falhou: {_short_error(exc)}"
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
    print()

    calibration_summary = build_calibration_summary(
        best_model_name,
        calibration_succeeded,
        raw_oof_y_true,
        raw_oof_y_proba,
        calibrated_oof_y_true,
        calibrated_oof_y_proba,
    )
    write_calibration_metrics(calibration_summary)
    write_calibration_report(calibration_summary)

    # Otimizacao do threshold de decisao (inalterada), agora alimentada
    # pelas previsoes out-of-fold do modelo CALIBRADO quando a calibracao
    # foi bem sucedida (ou do modelo original, se tiver falhado) - nunca o
    # modelo final gravado em .pkl aplicado aos dados que o treinaram, para
    # evitar leakage.
    print("=" * 60)
    print(f"OTIMIZACAO DO THRESHOLD (metrica={THRESHOLD_OPTIMIZATION_METRIC})")
    print("=" * 60)
    threshold_result = find_optimal_threshold(threshold_y_true, threshold_y_proba)
    if threshold_result is None:
        print(
            f"Nao foi possivel otimizar o threshold (amostras out-of-fold "
            f"insuficientes ou com uma so classe); a usar threshold por "
            f"omissao = {DEFAULT_THRESHOLD}"
        )
    else:
        print(
            f"Threshold otimo: {threshold_result['threshold']:.4f} "
            f"({THRESHOLD_OPTIMIZATION_METRIC}={threshold_result['metric_value']:.4f}, "
            f"n_oof_samples={threshold_result['n_oof_samples']})"
        )
    print()
    write_threshold_config(threshold_result, best_model_name)
    write_threshold_report(threshold_result, best_model_name, threshold_y_true, threshold_y_proba)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    if calibration_succeeded:
        print(f"Modelo vencedor calibrado guardado em: {MODEL_PATH}")
    else:
        print(
            f"Modelo vencedor (nao calibrado) re-treinado com 100% dos dados "
            f"e guardado em: {MODEL_PATH}"
        )

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
