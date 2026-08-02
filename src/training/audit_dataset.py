"""
Auditoria do dataset de treino (data/training_dataset.csv) para deteção
de data leakage.

Este script NAO treina nem altera o modelo de producao. Apenas analisa
o dataset e gera relatorios em models/dataset_leakage_report.{json,md}.

Uso:
    python src/training/audit_dataset.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATASET_PATH = "data/training_dataset.csv"
REPORT_JSON_PATH = "models/dataset_leakage_report.json"
REPORT_MD_PATH = "models/dataset_leakage_report.md"

TARGET_CANDIDATES = ["goal_next15", "goal_in_next_15m"]

LEAKAGE_AUC_THRESHOLD = 0.90

RANDOM_STATE = 42


def find_target_column(df):
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Nenhuma coluna alvo encontrada. Esperava uma de: {TARGET_CANDIDATES}"
    )


def show_overview(df):
    print("=" * 60)
    print("1. VISAO GERAL DO DATASET")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Colunas ({len(df.columns)}): {list(df.columns)}")
    print("\nTipos de dados:")
    print(df.dtypes)
    print()


def compute_feature_stats(df, feature_columns):
    print("=" * 60)
    print("2. ESTATISTICAS POR FEATURE")
    print("=" * 60)

    stats = {}
    for col in feature_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        col_stats = {
            "min": None if series.isna().all() else float(series.min()),
            "max": None if series.isna().all() else float(series.max()),
            "mean": None if series.isna().all() else float(series.mean()),
            "std": None if series.isna().all() else float(series.std()),
            "n_unique": int(df[col].nunique(dropna=True)),
        }
        stats[col] = col_stats
        print(
            f"{col:30s} min={col_stats['min']} max={col_stats['max']} "
            f"mean={col_stats['mean']} std={col_stats['std']} "
            f"n_unique={col_stats['n_unique']}"
        )
    print()
    return stats


def compute_correlations(df, feature_columns, target_col):
    print("=" * 60)
    print(f"3. CORRELACAO ABSOLUTA COM '{target_col}'")
    print("=" * 60)

    correlations = {}
    for col in feature_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        target = pd.to_numeric(df[target_col], errors="coerce")
        corr = series.corr(target)
        correlations[col] = None if pd.isna(corr) else float(abs(corr))

    for col, corr in sorted(
        correlations.items(), key=lambda item: (item[1] is None, -(item[1] or 0))
    ):
        print(f"{col:30s} abs_corr={corr}")
    print()
    return correlations


def train_single_feature_models(df, feature_columns, target_col):
    print("=" * 60)
    print("4. AUC POR FEATURE ISOLADA (RandomForest)")
    print("=" * 60)

    y = pd.to_numeric(df[target_col], errors="coerce")
    results = []

    for col in feature_columns:
        X = pd.to_numeric(df[col], errors="coerce").to_frame()
        valid_mask = X[col].notna() & y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if y_valid.nunique() < 2 or len(X_valid) < 10:
            results.append({"feature": col, "auc": None})
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_valid,
            y_valid,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_valid,
        )

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        results.append({"feature": col, "auc": float(auc)})

    results.sort(key=lambda item: (item["auc"] is None, -(item["auc"] or 0)))

    for item in results:
        flag = ""
        if item["auc"] is not None and item["auc"] > LEAKAGE_AUC_THRESHOLD:
            flag = "  *** POSSIVEL LEAKAGE ***"
        print(f"{item['feature']:30s} AUC={item['auc']}{flag}")
    print()

    return results


def build_report(df, target_col, feature_columns, stats, correlations, auc_results):
    leaking_features = [
        item["feature"]
        for item in auc_results
        if item["auc"] is not None and item["auc"] > LEAKAGE_AUC_THRESHOLD
    ]

    report = {
        "dataset_path": DATASET_PATH,
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "target_column": target_col,
        "feature_stats": stats,
        "correlation_with_target": correlations,
        "single_feature_auc": auc_results,
        "leakage_threshold": LEAKAGE_AUC_THRESHOLD,
        "possible_leakage_features": leaking_features,
        "conclusion": (
            f"*** POSSIVEL LEAKAGE *** detetado em {len(leaking_features)} feature(s): "
            f"{leaking_features}"
            if leaking_features
            else "Nenhuma feature isolada excedeu o limiar de AUC "
            f"({LEAKAGE_AUC_THRESHOLD}). Sem indicios de leakage evidente."
        ),
    }
    return report


def write_json_report(report):
    os.makedirs(os.path.dirname(REPORT_JSON_PATH), exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Relatorio JSON guardado em: {REPORT_JSON_PATH}")


def write_markdown_report(report):
    os.makedirs(os.path.dirname(REPORT_MD_PATH), exist_ok=True)

    lines = []
    lines.append("# Relatorio de Auditoria de Dataset - Deteccao de Leakage")
    lines.append("")
    lines.append(f"Dataset analisado: `{report['dataset_path']}`")
    lines.append("")
    lines.append("## 1. Visao geral")
    lines.append("")
    lines.append(f"- Linhas: {report['shape']['rows']}")
    lines.append(f"- Colunas: {report['shape']['columns']}")
    lines.append(f"- Coluna alvo: `{report['target_column']}`")
    lines.append("")
    lines.append("### Colunas e tipos")
    lines.append("")
    lines.append("| Coluna | Tipo |")
    lines.append("|---|---|")
    for col in report["columns"]:
        lines.append(f"| {col} | {report['dtypes'][col]} |")
    lines.append("")

    lines.append("## 2. Estatisticas por feature")
    lines.append("")
    lines.append("| Feature | Min | Max | Media | Std | Nº unicos |")
    lines.append("|---|---|---|---|---|---|")
    for col, s in report["feature_stats"].items():
        lines.append(
            f"| {col} | {s['min']} | {s['max']} | {s['mean']} | {s['std']} | {s['n_unique']} |"
        )
    lines.append("")

    lines.append("## 3. Correlacao absoluta com o alvo")
    lines.append("")
    lines.append("| Feature | Correlacao absoluta |")
    lines.append("|---|---|")
    sorted_corr = sorted(
        report["correlation_with_target"].items(),
        key=lambda item: (item[1] is None, -(item[1] or 0)),
    )
    for col, corr in sorted_corr:
        lines.append(f"| {col} | {corr} |")
    lines.append("")

    lines.append("## 4. AUC por feature isolada (RandomForest)")
    lines.append("")
    lines.append(f"Limiar de leakage: AUC > {report['leakage_threshold']}")
    lines.append("")
    lines.append("| Feature | AUC | Alerta |")
    lines.append("|---|---|---|")
    for item in report["single_feature_auc"]:
        alerta = (
            "*** POSSIVEL LEAKAGE ***"
            if item["auc"] is not None and item["auc"] > report["leakage_threshold"]
            else ""
        )
        lines.append(f"| {item['feature']} | {item['auc']} | {alerta} |")
    lines.append("")

    lines.append("## 5. Conclusao")
    lines.append("")
    lines.append(report["conclusion"])
    lines.append("")
    if report["possible_leakage_features"]:
        lines.append("Features assinaladas com possivel leakage:")
        lines.append("")
        for feat in report["possible_leakage_features"]:
            lines.append(f"- `{feat}`")
        lines.append("")

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Relatorio Markdown guardado em: {REPORT_MD_PATH}")


def main():
    df = pd.read_csv(DATASET_PATH)

    show_overview(df)

    target_col = find_target_column(df)
    feature_columns = [col for col in df.columns if col != target_col]

    stats = compute_feature_stats(df, feature_columns)
    correlations = compute_correlations(df, feature_columns, target_col)
    auc_results = train_single_feature_models(df, feature_columns, target_col)

    report = build_report(df, target_col, feature_columns, stats, correlations, auc_results)

    print("=" * 60)
    print("5. CONCLUSAO")
    print("=" * 60)
    print(report["conclusion"])
    print()

    write_json_report(report)
    write_markdown_report(report)


if __name__ == "__main__":
    main()
