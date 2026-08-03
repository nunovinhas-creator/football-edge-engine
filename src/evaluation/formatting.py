"""
Utilitário de formatação partilhado por `report.py` e `compare.py`:
converte um DataFrame numa tabela Markdown sem depender do pacote opcional
`tabulate` (que `DataFrame.to_markdown()` exigiria e que não está entre as
dependências do projeto).
"""

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Renderiza um DataFrame como tabela Markdown (GFM). Devolve um aviso se vazio."""
    if df is None or df.empty:
        return "_Sem dados._"

    columns = [str(c) for c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for _, row in df.iterrows():
        cells = ["" if pd.isna(value) else str(value) for value in row]
        body_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator, *body_lines])
