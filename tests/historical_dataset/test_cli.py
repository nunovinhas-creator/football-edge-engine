"""
Testes unitários do CLI oficial do Historical Dataset Builder
(src/historical_dataset/cli.py) e do wrapper `build_historical_dataset.py`.

Tudo mocado (BSDHistoricalClient, HistoricalDatasetBuilder, Checkpoint) —
nenhuma chamada real à BSD API, nenhuma rede. Cobre: parsing de
argumentos (incluindo os novos --competition-id/--season-id/--output/
--resume), seleção de formato de exportação, geração de
dataset_report.json, comportamento de --resume (checkpoint-dir por
omissão) e a garantia de que nada imprime a chave de API/tokens.
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.historical_dataset import cli
from src.historical_dataset.checkpoint import NullCheckpoint


def _sample_records(n=2):
    records = []
    for i in range(1, n + 1):
        records.append(
            {
                "event_id": i,
                "competition_id": 38,
                "competition": "Primeira Liga",
                "season_id": 2025,
                "season": "2024/2025",
                "home_team": f"Team {i}A",
                "away_team": f"Team {i}B",
                "date": "2024-01-0" + str(i),
                "odds_home": 1.9,
                "odds_draw": 3.2,
                "odds_away": 3.6,
            }
        )
    return records


class TestBuildParser(unittest.TestCase):

    def test_defaults(self):
        args = cli.build_parser().parse_args([])
        self.assertEqual(args.output, "all")
        self.assertEqual(args.resume, "false")
        self.assertEqual(args.output_dir, "data/historical")
        self.assertEqual(args.page_size, 100)
        self.assertIsNone(args.competition_id)
        self.assertIsNone(args.season_id)

    def test_accepts_competition_id_season_id_output_resume_page_size_output_dir(self):
        args = cli.build_parser().parse_args(
            [
                "--competition-id", "38",
                "--season-id", "2025",
                "--output", "all",
                "--resume", "true",
                "--page-size", "50",
                "--output-dir", "out/here",
            ]
        )
        self.assertEqual(args.competition_id, 38)
        self.assertEqual(args.season_id, 2025)
        self.assertEqual(args.output, "all")
        self.assertEqual(args.resume, "true")
        self.assertEqual(args.page_size, 50)
        self.assertEqual(args.output_dir, "out/here")

    def test_output_rejects_invalid_choice(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["--output", "xml"])

    def test_resume_rejects_invalid_choice(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["--resume", "yes"])


class TestProgressLogger(unittest.TestCase):

    def _capture(self, events, **kwargs):
        logger = cli.ProgressLogger(**kwargs)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for event_type, data in events:
                logger(event_type, data)
        return buf.getvalue()

    def test_prints_competition_and_season(self):
        out = self._capture(
            [
                ("competition_start", {"league_id": 38, "league_name": "Primeira Liga"}),
                ("season_start", {"season_id": 2025, "season_name": "2024/2025"}),
            ]
        )
        self.assertIn("Primeira Liga", out)
        self.assertIn("2024/2025", out)

    def test_prints_page_progress(self):
        out = self._capture([("page", {"page_number": 3, "items_count": 100})])
        self.assertIn("página 3", out)
        self.assertIn("100", out)

    def test_eta_unknown_without_max_events(self):
        out = self._capture([("event", {"games_processed": 1, "odds_processed": 1})], print_every=1)
        self.assertIn("desconhecido", out)

    def test_eta_estimated_with_max_events(self):
        out = self._capture(
            [("event", {"games_processed": 5, "odds_processed": 5})],
            max_events=10,
            print_every=1,
        )
        self.assertIn("ETA:", out)
        self.assertNotIn("desconhecido", out)

    def test_print_every_throttles_event_logging(self):
        events = [("event", {"games_processed": i, "odds_processed": i}) for i in range(1, 6)]
        out = self._capture(events, print_every=5)
        self.assertEqual(out.count("jogos processados"), 1)

    def test_never_prints_secret_values(self):
        logger = cli.ProgressLogger()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            logger("competition_start", {"league_id": 1, "league_name": "L", "api_key": "SHOULD_NOT_APPEAR"})
        self.assertNotIn("SHOULD_NOT_APPEAR", buf.getvalue())


class TestMain(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _run_main(self, extra_args, records=None, request_count=7):
        records = records if records is not None else _sample_records()

        fake_client_instance = MagicMock()
        fake_client_instance.request_count = request_count
        fake_client_instance.api_key = "SUPER_SECRET_DO_NOT_PRINT"

        captured_builder_kwargs = {}

        def _fake_builder_factory(**kwargs):
            captured_builder_kwargs.update(kwargs)
            fake_builder = MagicMock()
            fake_builder.build.return_value = iter(records)
            fake_builder.iter_competitions.return_value = iter([])
            return fake_builder

        buf = io.StringIO()
        with patch("src.historical_dataset.cli.BSDHistoricalClient", return_value=fake_client_instance), \
             patch("src.historical_dataset.cli.HistoricalDatasetBuilder", side_effect=_fake_builder_factory), \
             contextlib.redirect_stdout(buf):
            exit_code = cli.main(["--output-dir", self.tmp_dir] + extra_args)

        return exit_code, buf.getvalue(), captured_builder_kwargs, fake_client_instance

    def test_competition_id_and_leagues_together_is_rejected(self):
        with self.assertRaises(SystemExit):
            cli.main(["--competition-id", "38", "--leagues", "38,140", "--output-dir", self.tmp_dir])

    def test_output_csv_only_writes_csv(self):
        exit_code, out, _, _ = self._run_main(["--competition-id", "38", "--season-id", "2025", "--output", "csv"])

        self.assertEqual(exit_code, 0)
        self.assertTrue((Path(self.tmp_dir) / "historical_dataset.csv").exists())
        self.assertFalse((Path(self.tmp_dir) / "historical_dataset.sqlite").exists())
        self.assertFalse((Path(self.tmp_dir) / "historical_dataset.parquet").exists())

    def test_output_all_writes_csv_and_sqlite(self):
        exit_code, out, _, _ = self._run_main(["--output", "all"])

        self.assertEqual(exit_code, 0)
        self.assertTrue((Path(self.tmp_dir) / "historical_dataset.csv").exists())
        self.assertTrue((Path(self.tmp_dir) / "historical_dataset.sqlite").exists())

    def test_dataset_report_json_is_always_written(self):
        exit_code, out, _, client = self._run_main(
            ["--competition-id", "38", "--season-id", "2025", "--output", "csv"], request_count=42
        )

        report_path = Path(self.tmp_dir) / "dataset_report.json"
        self.assertTrue(report_path.exists())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["competition"], 38)
        self.assertEqual(report["season"], 2025)
        self.assertEqual(report["total_games"], 2)
        self.assertEqual(report["api_requests"], 42)
        self.assertIn("csv", report["output_files"])
        self.assertNotIn("sqlite", report["output_files"])

    def test_dataset_report_written_even_with_zero_games(self):
        exit_code, out, _, _ = self._run_main(["--output", "csv"], records=[])

        self.assertEqual(exit_code, 0)
        report_path = Path(self.tmp_dir) / "dataset_report.json"
        self.assertTrue(report_path.exists())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["total_games"], 0)
        self.assertFalse((Path(self.tmp_dir) / "historical_dataset.csv").exists())

    def test_resume_false_uses_null_checkpoint(self):
        with patch("src.historical_dataset.cli.Checkpoint") as mock_checkpoint_cls:
            self._run_main(["--output", "csv", "--resume", "false"])
            mock_checkpoint_cls.assert_not_called()

    def test_resume_true_without_checkpoint_dir_defaults_under_output_dir(self):
        with patch("src.historical_dataset.cli.Checkpoint") as mock_checkpoint_cls:
            mock_checkpoint_cls.return_value = NullCheckpoint()
            self._run_main(["--output", "csv", "--resume", "true"])

        mock_checkpoint_cls.assert_called_once()
        called_path = mock_checkpoint_cls.call_args[0][0]
        self.assertEqual(Path(called_path), Path(self.tmp_dir) / ".checkpoint")

    def test_resume_true_respects_explicit_checkpoint_dir(self):
        explicit_dir = str(Path(self.tmp_dir) / "custom_checkpoint")
        with patch("src.historical_dataset.cli.Checkpoint") as mock_checkpoint_cls:
            mock_checkpoint_cls.return_value = NullCheckpoint()
            self._run_main(["--output", "csv", "--resume", "true", "--checkpoint-dir", explicit_dir])

        called_path = mock_checkpoint_cls.call_args[0][0]
        self.assertEqual(called_path, explicit_dir)

    def test_season_ids_passed_through_to_builder(self):
        fake_client_instance = MagicMock()
        fake_client_instance.request_count = 1

        captured = {}

        def _fake_builder_factory(**kwargs):
            fake_builder = MagicMock()

            def _build(**build_kwargs):
                captured.update(build_kwargs)
                return iter(_sample_records())

            fake_builder.build.side_effect = _build
            return fake_builder

        with patch("src.historical_dataset.cli.BSDHistoricalClient", return_value=fake_client_instance), \
             patch("src.historical_dataset.cli.HistoricalDatasetBuilder", side_effect=_fake_builder_factory):
            cli.main(["--output-dir", self.tmp_dir, "--season-id", "2025", "--output", "csv"])

        self.assertEqual(captured.get("season_ids"), [2025])

    def test_progress_callback_wired_into_builder(self):
        _, _, builder_kwargs, _ = self._run_main(["--output", "csv"])
        self.assertIn("progress_callback", builder_kwargs)
        self.assertIsInstance(builder_kwargs["progress_callback"], cli.ProgressLogger)

    def test_never_prints_api_key_or_authorization(self):
        _, out, _, _ = self._run_main(["--competition-id", "38", "--output", "csv"])

        self.assertNotIn("SUPER_SECRET_DO_NOT_PRINT", out)
        self.assertNotIn("Authorization", out)
        self.assertNotIn("Token ", out)


if __name__ == "__main__":
    unittest.main()
