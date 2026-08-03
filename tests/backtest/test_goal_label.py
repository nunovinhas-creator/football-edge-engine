"""
Testes da implementação única e oficial do label `goal_in_next_15m`
(src/backtest/goal_label.py).

Cobrem a definição matemática: existe golo (aumento de
home_score+away_score) num snapshot posterior do mesmo jogo, em minuto
m' tal que m < m' <= m + 15.

Casos: golo exatamente aos 15 minutos (fronteira), antes dos 15, depois
dos 15, vários golos, e sem golos.
"""

import sqlite3
import unittest

from src.backtest.goal_label import LABEL_COLUMN, recompute_goal_in_next_15m


SCHEMA = """
CREATE TABLE match_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT,
    current_minute INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    goal_in_next_15m BOOLEAN DEFAULT NULL
)
"""


class GoalInNext15mTestCase(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(SCHEMA)

    def tearDown(self):
        self.conn.close()

    def _insert(self, match_id, minute, home_score, away_score):
        self.conn.execute(
            "INSERT INTO match_snapshots "
            "(match_id, current_minute, home_score, away_score) "
            "VALUES (?, ?, ?, ?)",
            (match_id, minute, home_score, away_score),
        )

    def _labels_for(self, match_id):
        recompute_goal_in_next_15m(self.conn)
        rows = self.conn.execute(
            f"SELECT current_minute, {LABEL_COLUMN} FROM match_snapshots "
            "WHERE match_id = ? ORDER BY current_minute",
            (match_id,),
        ).fetchall()
        return dict(rows)

    def test_goal_exactly_at_15_minutes_counts_as_positive(self):
        # snapshot em minuto 60, 0-0; golo registado no snapshot de minuto 75 (60+15)
        self._insert("m1", 60, 0, 0)
        self._insert("m1", 75, 1, 0)

        labels = self._labels_for("m1")

        self.assertEqual(labels[60], 1)

    def test_goal_before_15_minutes_counts_as_positive(self):
        # golo aos 5 minutos depois do snapshot (dentro da janela)
        self._insert("m2", 60, 0, 0)
        self._insert("m2", 65, 1, 0)

        labels = self._labels_for("m2")

        self.assertEqual(labels[60], 1)

    def test_goal_after_15_minutes_counts_as_negative(self):
        # golo aos 16 minutos depois do snapshot (fora da janela)
        self._insert("m3", 60, 0, 0)
        self._insert("m3", 76, 1, 0)

        labels = self._labels_for("m3")

        self.assertEqual(labels[60], 0)

    def test_multiple_goals_still_positive_once(self):
        # dois golos dentro da janela: label continua a ser 1 (não conta golos)
        self._insert("m4", 60, 0, 0)
        self._insert("m4", 66, 1, 0)
        self._insert("m4", 70, 2, 0)

        labels = self._labels_for("m4")

        self.assertEqual(labels[60], 1)

    def test_no_goals_at_all_is_negative(self):
        self._insert("m5", 60, 0, 0)
        self._insert("m5", 70, 0, 0)
        self._insert("m5", 80, 0, 0)

        labels = self._labels_for("m5")

        self.assertEqual(labels[60], 0)
        self.assertEqual(labels[70], 0)
        self.assertEqual(labels[80], 0)

    def test_goal_already_reflected_in_own_snapshot_does_not_count(self):
        # o próprio snapshot de minuto 60 já tem o golo (1-0): a label
        # olha para golos *depois* do minuto do snapshot, não no próprio.
        self._insert("m6", 60, 1, 0)

        labels = self._labels_for("m6")

        self.assertEqual(labels[60], 0)

    def test_rows_without_current_minute_are_left_untouched(self):
        self.conn.execute(
            "INSERT INTO match_snapshots "
            "(match_id, current_minute, home_score, away_score) "
            "VALUES (?, NULL, ?, ?)",
            ("m7", 0, 0),
        )

        recompute_goal_in_next_15m(self.conn)

        row = self.conn.execute(
            f"SELECT {LABEL_COLUMN} FROM match_snapshots WHERE match_id = ?",
            ("m7",),
        ).fetchone()

        self.assertIsNone(row[0])


if __name__ == "__main__":
    unittest.main()
