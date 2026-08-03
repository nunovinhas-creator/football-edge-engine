"""
Reexporta o gerador de dataset sintético usado pelos testes de integração.
A implementação vive em `src/backtest/historical/sample_data.py` para
poder ser reutilizada também pelo exemplo de execução
(`src/tools/run_backtest_example.py`) sem duplicar código nem fazer
`src` depender de `tests`.
"""

from src.backtest.historical.sample_data import generate_sample_dataset

__all__ = ["generate_sample_dataset"]
