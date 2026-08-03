"""
Testes unitários dos exporters (src/historical_dataset/storage.py).

Cobre: exportação para CSV, SQLite (via `sqlite3` da biblioteca padrão) e
Parquet (se um motor estiver instalado no ambiente — o repositório traz
`pyarrow` como dependência transitiva do `streamlit`), `export_all`
devolvendo os três caminhos, e o caso de um dataset vazio.
"""

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.historical_dataset.storage import export_all, to_csv, to_dataframe, to_parquet, to_sqlite


def _sample_records():
    return [
        {"event_id": 1, "competition": "Premier League", "home_team": "A", "away_team": "B",
         "home_score": 2, "away_score": 1, "odds_home": 1.9},
        {"event_id": 2, "competition": "Premier League", "home_team": "C", "away_team": "D",
         "home_score": 0, "away_score": 0, "odds_home": 2.5},
    ]


class TestToDataFrame(unittest.TestCase):

    def test_list_of_dicts_converted(self):
        df = to_dataframe(_sample_records())
        self.assertEqual(len(df), 2)
        self.assertIn("event_id", df.columns)

    def test_dataframe_passthrough_is_a_copy(self):
        original = pd.DataFrame(_sample_records())
        df = to_dataframe(original)
        df.loc[0, "event_id"] = 999
        self.assertEqual(original.loc[0, "event_id"], 1)

    def test_empty_list_returns_empty_dataframe_with_normalized_columns(self):
        df = to_dataframe([])
        self.assertTrue(df.empty)
        self.assertIn("event_id", df.columns)
        self.assertIn("odds_home", df.columns)


class TestExporters(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_to_csv_writes_readable_file(self):
        path = to_csv(_sample_records(), self.tmp_dir / "out.csv")

        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "home_team"], "A")

    def test_to_csv_creates_missing_parent_dirs(self):
        path = to_csv(_sample_records(), self.tmp_dir / "nested" / "dir" / "out.csv")
        self.assertTrue(path.exists())

    def test_to_sqlite_writes_queryable_table(self):
        path = to_sqlite(_sample_records(), self.tmp_dir / "out.sqlite", table="historical_matches")

        conn = sqlite3.connect(path)
        try:
            rows = conn.execute("SELECT event_id, home_team FROM historical_matches ORDER BY event_id").fetchall()
        finally:
            conn.close()

        self.assertEqual(rows, [(1, "A"), (2, "C")])

    def test_to_sqlite_replaces_existing_table(self):
        path = self.tmp_dir / "out.sqlite"
        to_sqlite(_sample_records(), path)
        to_sqlite(_sample_records()[:1], path)

        conn = sqlite3.connect(path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM historical_matches").fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(count, 1)

    def test_to_parquet_writes_readable_file_when_engine_available(self):
        path = to_parquet(_sample_records(), self.tmp_dir / "out.parquet")

        if path is None:
            self.skipTest("Nenhum motor Parquet instalado neste ambiente")

        df = pd.read_parquet(path)
        self.assertEqual(len(df), 2)

    def test_export_all_writes_csv_and_sqlite_always(self):
        paths = export_all(_sample_records(), self.tmp_dir, base_name="ds")

        self.assertTrue(Path(paths["csv"]).exists())
        self.assertTrue(Path(paths["sqlite"]).exists())
        self.assertIn("parquet", paths)  # pode ser um Path ou None

    def test_export_all_handles_empty_dataset(self):
        paths = export_all([], self.tmp_dir, base_name="empty")

        df = pd.read_csv(paths["csv"])
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
