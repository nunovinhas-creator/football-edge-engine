"""
Testes unitários da deduplicação (src/historical_dataset/dedup.py).
"""

import unittest

from src.historical_dataset.dedup import Deduplicator, dedupe_records


class TestDeduplicator(unittest.TestCase):

    def test_unseen_key_is_not_duplicate(self):
        dedup = Deduplicator()
        self.assertFalse(dedup.is_duplicate(1))

    def test_seen_key_is_duplicate(self):
        dedup = Deduplicator()
        dedup.add(1)
        self.assertTrue(dedup.is_duplicate(1))
        self.assertFalse(dedup.is_duplicate(2))

    def test_seen_count(self):
        dedup = Deduplicator()
        dedup.add(1)
        dedup.add(2)
        dedup.add(1)  # repetido, não conta duas vezes
        self.assertEqual(dedup.seen_count(), 2)


class TestDedupeRecords(unittest.TestCase):

    def test_keeps_first_occurrence_by_default_key(self):
        records = [
            {"event_id": 1, "home_score": 1},
            {"event_id": 2, "home_score": 2},
            {"event_id": 1, "home_score": 99},  # duplicado, deve ser ignorado
        ]

        result = list(dedupe_records(records))

        self.assertEqual([r["event_id"] for r in result], [1, 2])
        self.assertEqual(result[0]["home_score"], 1)

    def test_custom_key_fn(self):
        records = [
            {"home_team": "A", "away_team": "B", "date": "2020-01-01"},
            {"home_team": "A", "away_team": "B", "date": "2020-01-01"},
            {"home_team": "C", "away_team": "D", "date": "2020-01-01"},
        ]

        result = list(dedupe_records(
            records,
            key_fn=lambda r: (r["home_team"], r["away_team"], r["date"]),
        ))

        self.assertEqual(len(result), 2)

    def test_empty_input_returns_empty(self):
        self.assertEqual(list(dedupe_records([])), [])


if __name__ == "__main__":
    unittest.main()
