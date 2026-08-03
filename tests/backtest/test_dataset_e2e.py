"""
Teste end-to-end do pipeline de backtesting sobre jogos históricos REAIS.

Usa `examples/backtest/sample_real_games.csv` — um pequeno conjunto (8 jogos)
de resultados históricos verídicos (datas, equipas e resultados finais
reais e verificáveis publicamente: "6-1" Man Utd-Man City 2011,
"5-0" Barcelona-Real Madrid 2010, final da Champions League 2019, etc.).
As odds e as probabilidades do modelo associadas a cada jogo são
ilustrativas (o repositório não contém um arquivo de odds históricas nem
as probabilidades que o motor de previsão teria produzido nesse momento),
servindo apenas para exercitar o pipeline completo:

    CSV -> dataset.load_historical_dataset -> dataset.filter_dataset
         -> BacktestEngine.run -> BacktestReport (CSV/Excel/HTML/gráficos)

Nenhum algoritmo de previsão é alterado ou recalculado neste teste — o
resultado de cada mercado é derivado do resultado final real do jogo
(`infer_market_result`), e Edge/EV/Kelly continuam a ser calculados pelas
implementações oficiais (`src.engine.edge`, `src.engine.kelly`).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.backtest.historical import BacktestEngine, FlatStake
from src.backtest.historical.dataset import filter_dataset, load_historical_dataset

SAMPLE_REAL_GAMES_CSV = (
    Path(__file__).resolve().parent.parent.parent / "examples" / "backtest" / "sample_real_games.csv"
)


class TestHistoricalDatasetEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="backtest_e2e_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_sample_real_games_csv_exists(self):
        self.assertTrue(SAMPLE_REAL_GAMES_CSV.exists())

    def test_load_full_dataset_from_csv(self):
        df = load_historical_dataset(SAMPLE_REAL_GAMES_CSV)
        self.assertEqual(len(df), 8)
        for column in ["date", "competition", "home_team", "away_team", "market", "odd", "result", "engine_decision"]:
            self.assertIn(column, df.columns)
        # jogo real e bem conhecido: Man Utd 1-6 Man City (23/10/2011)
        row = df[df["home_team"] == "Manchester United"].iloc[0]
        self.assertEqual(row["away_team"], "Manchester City")
        self.assertEqual(row["market"], "AWAY")
        self.assertEqual(row["result"], "WIN")  # Man City venceu fora, mercado AWAY acertou

    def test_full_backtest_runs_end_to_end(self):
        df = load_historical_dataset(SAMPLE_REAL_GAMES_CSV)
        report = BacktestEngine(staking=FlatStake(unit=1.0)).run(df)

        self.assertEqual(len(report.all_bets), 8)
        for column in ["match", "date", "competition", "home_team", "away_team", "market", "edge", "ev", "result"]:
            self.assertIn(column, report.all_bets.columns)

        self.assertIn("roi_pct", report.global_metrics)
        self.assertEqual(report.global_metrics["n_bets"], len(report.placed_bets))

    def test_backtest_by_date_range(self):
        df = load_historical_dataset(SAMPLE_REAL_GAMES_CSV)
        filtered = filter_dataset(df, start_date="2017-01-01", end_date="2019-12-31")
        # Barcelona-PSG (2017), Liverpool-Barcelona (2019), final CL 2019
        self.assertEqual(len(filtered), 3)
        report = BacktestEngine().run(filtered)
        self.assertEqual(len(report.all_bets), 3)

    def test_backtest_by_competition(self):
        df = load_historical_dataset(SAMPLE_REAL_GAMES_CSV)
        filtered = filter_dataset(df, competition="Champions League")
        self.assertEqual(len(filtered), 4)
        report = BacktestEngine().run(filtered)
        self.assertTrue((report.all_bets["competition"] == "Champions League").all())

    def test_backtest_by_market(self):
        df = load_historical_dataset(SAMPLE_REAL_GAMES_CSV)
        filtered = filter_dataset(df, market="OVER_2.5")
        self.assertEqual(len(filtered), 2)
        report = BacktestEngine().run(filtered)
        self.assertTrue((report.all_bets["market"] == "OVER_2.5").all())
        self.assertTrue((report.all_bets["won"]).all())  # ambos os jogos tiveram >2.5 golos

    def test_report_exports_csv_excel_html_and_plots(self):
        df = load_historical_dataset(SAMPLE_REAL_GAMES_CSV)
        report = BacktestEngine(staking=FlatStake(unit=1.0)).run(df)

        csv_dir = os.path.join(self.tmp_dir, "csv")
        written_csv = report.to_csv(csv_dir)
        self.assertTrue(all(os.path.exists(p) for p in written_csv.values()))

        try:
            import openpyxl  # noqa: F401
            excel_path = report.to_excel(os.path.join(self.tmp_dir, "report.xlsx"))
            self.assertTrue(os.path.exists(excel_path))
        except ImportError:
            pass

        plots_dir = os.path.join(self.tmp_dir, "plots")
        try:
            import matplotlib  # noqa: F401
            written_plots = report.generate_all_plots(plots_dir)
            self.assertTrue(all(os.path.exists(p) for p in written_plots.values()))
        except ImportError:
            plots_dir = None

        html_path = report.to_html(os.path.join(self.tmp_dir, "report.html"), plots_dir=plots_dir)
        self.assertTrue(os.path.exists(html_path))
        self.assertGreater(os.path.getsize(html_path), 0)
        with open(html_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<html", content)
        self.assertIn("Resumo Global", content)


if __name__ == "__main__":
    unittest.main()
