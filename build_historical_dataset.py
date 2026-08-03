#!/usr/bin/env python3
"""
CLI para correr o Historical Dataset Builder (`src.historical_dataset`) de
ponta a ponta: competições -> épocas -> jogos terminados -> odds ->
estatísticas -> dataset normalizado -> CSV/SQLite/Parquet +
`dataset_report.json`.

Não altera nenhum algoritmo de previsão — Poisson, Dixon-Coles, Monte
Carlo, Goal Engine, Machine Learning, Kelly, Edge e EV permanecem
exatamente como estão. Este script apenas percorre a BSD API e normaliza
os dados brutos (ver `docs/07_historical_dataset_builder.md` e
`docs/08_historical_dataset_workflow.md`).

A implementação (parsing de argumentos, logging de progresso, exportação e
relatório) vive em `src.historical_dataset.cli` — testável sem chamadas
reais à BSD API (ver `tests/historical_dataset/test_cli.py`). Este ficheiro
é apenas o ponto de entrada de linha de comandos.

Exemplos:

    # Todas as ligas ativas, sem checkpoint (execução única)
    python build_historical_dataset.py --output-dir data/historical

    # Uma única competição/época, com checkpoint/resume
    python build_historical_dataset.py --competition-id 38 --season-id 2025 \\
        --output all --resume true

    # Apenas duas ligas específicas, com checkpoint/resume (formato antigo)
    python build_historical_dataset.py --leagues 39,140 --checkpoint-dir data/historical/.checkpoint

    # Execução parcial, limitada a 200 jogos (ex. para testar rapidamente)
    python build_historical_dataset.py --max-events 200
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.historical_dataset.cli import main

if __name__ == "__main__":
    sys.exit(main())
