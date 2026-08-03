"""
Teste de integração de `src/evaluation/report.py`: corre `evaluate(...)`
de ponta a ponta sobre um dataset sintético (ver
`tests.backtest.fixtures.generate_sample_dataset`) e valida que os
segmentos extra e as quatro exportações (CSV, Excel, HTML, Markdown) e os
gráficos adicionais funcionam sem erro.
"""

import os
import shutil
import tempfile
import unittest

from src.backtest.historical import FlatStake
from src.evaluation import EvaluationReport, evaluate
from tests.backtest.fixtures import generate_sample_dataset


class TestEvaluationReportEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_sample_dataset(n_games=150, seed=7)
        cls.report = evaluate(cls.dataset, staking=FlatStake(unit=1.0))
        cls.tmp_dir = tempfile.mkdtemp(prefix="evaluation_report_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_returns_evaluation_report_instance(self):
        self.assertIsInstance(self.report, EvaluationReport)

    def test_all_segment_tables_include_month_and_confidence(self):
        segments = self.report.all_segment_tables()
        for expected in [
            "by_competition", "by_market", "by_odd_range", "by_edge_range",
            "by_month", "by_confidence_range",
        ]:
            self.assertIn(expected, segments)

    def test_global_and_statistical_metrics_present(self):
        self.assertIn("roi_pct", self.report.global_metrics)
        self.assertIn("brier_score", self.report.statistical_metrics)

    def test_to_csv_writes_extra_segment_files(self):
        output_dir = os.path.join(self.tmp_dir, "csv")
        written = self.report.to_csv(output_dir)
        self.assertIn("segment_by_month", written)
        self.assertIn("segment_by_confidence_range", written)
        for path in written.values():
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_to_excel_writes_a_single_workbook(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl não está instalado")

        path = os.path.join(self.tmp_dir, "report.xlsx")
        result_path = self.report.to_excel(path)
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 0)

    def test_generate_all_plots_writes_more_than_the_original_five(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib não está instalado")

        output_dir = os.path.join(self.tmp_dir, "plots")
        written = self.report.generate_all_plots(output_dir)
        # 5 do BacktestReport + 6 novos deste módulo
        self.assertGreaterEqual(len(written), 10)
        for path in written.values():
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_to_html_embeds_plots_when_available(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib não está instalado")

        plots_dir = os.path.join(self.tmp_dir, "plots_for_html")
        self.report.generate_all_plots(plots_dir)
        html_path = os.path.join(self.tmp_dir, "report.html")
        result_path = self.report.to_html(html_path, plots_dir=plots_dir)
        self.assertTrue(os.path.exists(result_path))
        with open(result_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("base64", content)
        self.assertIn("by_month", content)

    def test_to_markdown_contains_summary_and_segments(self):
        md_path = os.path.join(self.tmp_dir, "report.md")
        result_path = self.report.to_markdown(md_path)
        self.assertTrue(os.path.exists(result_path))
        with open(result_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("Resumo Global", content)
        self.assertIn("by_confidence_range", content)

    def test_empty_dataset_does_not_crash_any_export(self):
        empty_report = evaluate([])
        output_dir = os.path.join(self.tmp_dir, "empty")
        empty_report.to_csv(output_dir)
        empty_report.to_markdown(os.path.join(output_dir, "report.md"))


if __name__ == "__main__":
    unittest.main()
