"""
Testes unitários do checkpoint/resume (src/historical_dataset/checkpoint.py).

Cobre: persistência de épocas concluídas e jogos processados em disco,
resume a partir de uma instância `Checkpoint` nova apontada para o mesmo
diretório (simulando reiniciar o processo), e o `NullCheckpoint` no-op
usado quando o utilizador não pede checkpoint/resume.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.historical_dataset.checkpoint import Checkpoint, NullCheckpoint


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_season_not_done_by_default(self):
        cp = Checkpoint(self.tmp_dir)
        self.addCleanup(cp.close)

        self.assertFalse(cp.is_season_done(39, 2024))

    def test_mark_season_done_persists_across_instances(self):
        cp1 = Checkpoint(self.tmp_dir)
        cp1.mark_season_done(39, 2024)
        cp1.close()

        cp2 = Checkpoint(self.tmp_dir)
        self.addCleanup(cp2.close)

        self.assertTrue(cp2.is_season_done(39, 2024))
        self.assertFalse(cp2.is_season_done(39, 2023))

    def test_mark_event_done_persists_across_instances(self):
        cp1 = Checkpoint(self.tmp_dir)
        cp1.mark_event_done(1001)
        cp1.mark_event_done(1002)
        cp1.close()

        cp2 = Checkpoint(self.tmp_dir)
        self.addCleanup(cp2.close)

        self.assertTrue(cp2.is_event_done(1001))
        self.assertTrue(cp2.is_event_done(1002))
        self.assertFalse(cp2.is_event_done(9999))

    def test_marking_same_event_twice_does_not_duplicate_log_lines(self):
        cp = Checkpoint(self.tmp_dir)
        cp.mark_event_done(42)
        cp.mark_event_done(42)
        cp.close()

        log_path = Path(self.tmp_dir) / "processed_events.log"
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(lines, ["42"])

    def test_creates_directory_if_missing(self):
        nested = Path(self.tmp_dir) / "a" / "b" / "checkpoint"
        cp = Checkpoint(nested)
        self.addCleanup(cp.close)

        cp.mark_season_done(1, 1)

        self.assertTrue(nested.exists())


class TestNullCheckpoint(unittest.TestCase):

    def test_never_reports_anything_as_done(self):
        cp = NullCheckpoint()

        self.assertFalse(cp.is_season_done(1, 1))
        self.assertFalse(cp.is_event_done(1))

        cp.mark_season_done(1, 1)
        cp.mark_event_done(1)

        self.assertFalse(cp.is_season_done(1, 1))
        self.assertFalse(cp.is_event_done(1))

    def test_close_is_a_no_op(self):
        cp = NullCheckpoint()
        cp.close()  # não deve levantar


if __name__ == "__main__":
    unittest.main()
