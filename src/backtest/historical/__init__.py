"""
Backtesting Framework histórico.

Analisa milhares de jogos históricos e produz métricas objetivas de
desempenho (ROI, Yield, Drawdown, Profit Factor, Brier Score, Log Loss,
ECE, análises por segmento e por threshold de Edge/EV), SEM alterar o
motor de previsão existente (Poisson, Dixon-Coles, Monte Carlo, Goal
Engine, Kelly, Machine Learning) nem as definições oficiais de Edge/EV
(`src.engine.edge`).

Uso típico:

    from src.backtest.historical import BacktestEngine

    engine = BacktestEngine()
    report = engine.run(historical_rows)  # lista de dicts ou DataFrame

    report.print_summary()
    report.to_csv("output/backtest")
    report.generate_all_plots("output/backtest/plots")
"""

from .dataset import (
    filter_dataset,
    infer_market_result,
    load_games_from_csv,
    load_historical_dataset,
)
from .engine import BacktestEngine, run_backtest
from .evaluator import evaluate_bet, evaluate_bets
from .models import EvaluatedBet, HistoricalBet, load_historical_bets
from .report import BacktestReport
from .sample_data import generate_sample_dataset
from .staking import FlatStake, KellyStake, StakingStrategy

__all__ = [
    "BacktestEngine",
    "run_backtest",
    "evaluate_bet",
    "evaluate_bets",
    "EvaluatedBet",
    "HistoricalBet",
    "load_historical_bets",
    "BacktestReport",
    "generate_sample_dataset",
    "FlatStake",
    "KellyStake",
    "StakingStrategy",
    "load_historical_dataset",
    "load_games_from_csv",
    "filter_dataset",
    "infer_market_result",
]
