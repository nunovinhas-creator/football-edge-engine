"""
Testes unitários do orquestrador (src/historical_dataset/builder.py).

Usa um cliente falso em memória (sem rede) que simula competições, épocas,
jogos, odds e estatísticas da BSD API, para validar: percurso completo
competição -> época -> jogos -> odds -> estatísticas, deduplicação,
checkpoint/resume (retomar sem repetir épocas/jogos já concluídos),
tratamento de falhas parciais (odds/stats indisponíveis não interrompem o
pipeline) e o limite opcional `max_events`.
"""

import shutil
import tempfile
import unittest

from src.historical_dataset.builder import HistoricalDatasetBuilder, _extract_seasons
from src.historical_dataset.checkpoint import Checkpoint
from src.historical_dataset.client import BSDAPIError


class FakeHistoricalClient:
    """
    Cliente falso em memória: simula /leagues/, /leagues/{id}/seasons/,
    /events/ (filtrado por league_id+season_id+status), /events/{id}/odds/
    e /events/{id}/stats/.
    """

    def __init__(self, leagues, seasons_by_league, events_by_season, odds_by_event=None, stats_by_event=None, odds_errors=None):
        self.leagues = leagues
        self.seasons_by_league = seasons_by_league
        self.events_by_season = events_by_season
        self.odds_by_event = odds_by_event or {}
        self.stats_by_event = stats_by_event or {}
        self.odds_errors = odds_errors or {}
        self.calls = []

    def get(self, endpoint, params=None):
        self.calls.append((endpoint, dict(params or {})))
        params = params or {}

        if endpoint == "leagues/":
            offset = params.get("offset", 0)
            limit = params.get("limit", 100)
            return self.leagues[offset:offset + limit]

        if endpoint.startswith("leagues/") and endpoint.endswith("/seasons/"):
            league_id = int(endpoint.split("/")[1])
            return self.seasons_by_league.get(league_id, [])

        if endpoint == "events/":
            league_id = params.get("league_id")
            season_id = params.get("season_id")
            offset = params.get("offset", 0)
            limit = params.get("limit", 100)
            events = self.events_by_season.get((league_id, season_id), [])
            return events[offset:offset + limit]

        if endpoint.endswith("/odds/") and "/comparison" not in endpoint:
            event_id = int(endpoint.split("/")[1])
            if event_id in self.odds_errors:
                raise self.odds_errors[event_id]
            return self.odds_by_event.get(event_id)

        if endpoint.endswith("/stats/"):
            event_id = int(endpoint.split("/")[1])
            return self.stats_by_event.get(event_id)

        raise AssertionError(f"Endpoint inesperado no FakeHistoricalClient: {endpoint}")


def _make_client():
    leagues = [{"id": 39, "name": "Premier League"}]
    seasons_by_league = {39: [{"id": 2024, "name": "2023/2024"}]}
    events_by_season = {
        (39, 2024): [
            {"id": 1, "league_id": 39, "season_id": 2024, "home_team": "A", "away_team": "B",
             "event_date": "2024-01-01T00:00:00Z", "status": "finished", "home_score": 2, "away_score": 1,
             "home_score_ht": 1, "away_score_ht": 0},
            {"id": 2, "league_id": 39, "season_id": 2024, "home_team": "C", "away_team": "D",
             "event_date": "2024-01-08T00:00:00Z", "status": "finished", "home_score": 0, "away_score": 0,
             "home_score_ht": 0, "away_score_ht": 0},
        ]
    }
    odds_by_event = {
        1: {"1x2": {"home": 1.9, "draw": 3.4, "away": 4.1}},
        2: {"1x2": {"home": 2.5, "draw": 3.1, "away": 2.9}},
    }
    stats_by_event = {
        1: {"home": {"yellow_cards": 1}, "away": {"yellow_cards": 2}},
        2: {"home": {"yellow_cards": 0}, "away": {"yellow_cards": 1}},
    }
    return FakeHistoricalClient(leagues, seasons_by_league, events_by_season, odds_by_event, stats_by_event)


class TestHistoricalDatasetBuilder(unittest.TestCase):

    def test_full_walk_produces_one_record_per_finished_event(self):
        client = _make_client()
        builder = HistoricalDatasetBuilder(client=client)

        records = builder.build_to_list()

        self.assertEqual(len(records), 2)
        self.assertEqual({r["event_id"] for r in records}, {1, 2})
        self.assertEqual(records[0]["competition"], "Premier League")
        self.assertEqual(records[0]["season"], "2023/2024")

    def test_odds_and_stats_are_merged_into_record(self):
        client = _make_client()
        builder = HistoricalDatasetBuilder(client=client)

        records = {r["event_id"]: r for r in builder.build_to_list()}

        self.assertEqual(records[1]["odds_home"], 1.9)
        self.assertEqual(records[1]["cards_home_yellow"], 1)
        self.assertEqual(records[2]["odds_away"], 2.9)

    def test_max_events_limits_output(self):
        client = _make_client()
        builder = HistoricalDatasetBuilder(client=client)

        records = builder.build_to_list(max_events=1)

        self.assertEqual(len(records), 1)

    def test_explicit_leagues_subset_is_respected(self):
        client = _make_client()
        client.leagues.append({"id": 999, "name": "Should Not Be Visited"})
        builder = HistoricalDatasetBuilder(client=client)

        records = builder.build_to_list(leagues=[{"id": 39, "name": "Premier League"}])

        self.assertEqual(len(records), 2)
        for endpoint, params in client.calls:
            self.assertNotEqual(params.get("league_id"), 999)

    def test_odds_failure_does_not_abort_pipeline(self):
        client = _make_client()
        client.odds_errors[1] = BSDAPIError(500, "https://example.test", "boom")
        errors = []
        builder = HistoricalDatasetBuilder(client=client, on_error=lambda stage, eid, exc: errors.append((stage, eid)))

        records = {r["event_id"]: r for r in builder.build_to_list()}

        self.assertEqual(len(records), 2)
        self.assertIsNone(records[1]["odds_home"])
        self.assertEqual(records[2]["odds_home"], 2.5)
        self.assertEqual(errors, [("odds", 1)])

    def test_404_odds_is_silently_treated_as_unavailable(self):
        client = _make_client()
        client.odds_errors[1] = BSDAPIError(404, "https://example.test", "not found")
        errors = []
        builder = HistoricalDatasetBuilder(client=client, on_error=lambda stage, eid, exc: errors.append((stage, eid)))

        records = {r["event_id"]: r for r in builder.build_to_list()}

        self.assertIsNone(records[1]["odds_home"])
        self.assertEqual(errors, [])  # 404 não é reportado como erro


class TestHistoricalDatasetBuilderSeasonFilter(unittest.TestCase):

    def test_season_ids_filters_out_other_seasons(self):
        client = _make_client()
        client.seasons_by_league[39].append({"id": 2023, "name": "2022/2023"})
        client.events_by_season[(39, 2023)] = [
            {"id": 99, "league_id": 39, "season_id": 2023, "home_team": "X", "away_team": "Y",
             "event_date": "2023-01-01T00:00:00Z", "status": "finished", "home_score": 1, "away_score": 1,
             "home_score_ht": 0, "away_score_ht": 0},
        ]
        builder = HistoricalDatasetBuilder(client=client)

        records = builder.build_to_list(season_ids=[2024])

        self.assertEqual({r["event_id"] for r in records}, {1, 2})
        self.assertFalse(any(endpoint == "events/" and params.get("season_id") == 2023 for endpoint, params in client.calls))

    def test_season_ids_none_keeps_all_seasons(self):
        client = _make_client()
        builder = HistoricalDatasetBuilder(client=client)

        records = builder.build_to_list(season_ids=None)

        self.assertEqual(len(records), 2)


class TestHistoricalDatasetBuilderProgressCallback(unittest.TestCase):

    def test_emits_expected_event_types_in_order(self):
        client = _make_client()
        events = []
        builder = HistoricalDatasetBuilder(
            client=client, progress_callback=lambda event_type, data: events.append((event_type, data))
        )

        builder.build_to_list()

        event_types = [e[0] for e in events]
        self.assertEqual(
            event_types,
            ["competition_start", "season_start", "page", "event", "event", "season_done"],
        )

    def test_event_progress_carries_running_counts(self):
        client = _make_client()
        events = []
        builder = HistoricalDatasetBuilder(
            client=client, progress_callback=lambda event_type, data: events.append((event_type, data))
        )

        builder.build_to_list()

        event_payloads = [data for etype, data in events if etype == "event"]
        self.assertEqual([e["games_processed"] for e in event_payloads], [1, 2])
        self.assertEqual([e["odds_processed"] for e in event_payloads], [1, 2])

    def test_page_event_reports_page_number_and_items_count(self):
        client = _make_client()
        events = []
        builder = HistoricalDatasetBuilder(
            client=client, progress_callback=lambda event_type, data: events.append((event_type, data))
        )

        builder.build_to_list()

        page_events = [data for etype, data in events if etype == "page"]
        self.assertEqual(page_events, [{"league_id": 39, "season_id": 2024, "page_number": 1, "items_count": 2}])

    def test_no_progress_callback_is_a_safe_no_op(self):
        client = _make_client()
        builder = HistoricalDatasetBuilder(client=client)  # sem progress_callback
        records = builder.build_to_list()
        self.assertEqual(len(records), 2)


class TestHistoricalDatasetBuilderCheckpointResume(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_resumed_run_skips_completed_season(self):
        client = _make_client()
        checkpoint = Checkpoint(self.tmp_dir)
        builder = HistoricalDatasetBuilder(client=client, checkpoint=checkpoint)

        first_run = builder.build_to_list()
        checkpoint.close()
        self.assertEqual(len(first_run), 2)

        # Simula reiniciar o processo: nova instância de Checkpoint sobre o mesmo diretório.
        resumed_checkpoint = Checkpoint(self.tmp_dir)
        self.addCleanup(resumed_checkpoint.close)
        client2 = _make_client()
        builder2 = HistoricalDatasetBuilder(client=client2, checkpoint=resumed_checkpoint)

        second_run = builder2.build_to_list()

        self.assertEqual(second_run, [])  # época já concluída, nada para reprocessar
        # events/ nunca deveria ter sido pedido de novo para a época já concluída
        self.assertFalse(any(endpoint == "events/" for endpoint, _ in client2.calls))

    def test_partial_run_marks_individual_events_done_for_resume(self):
        client = _make_client()
        checkpoint = Checkpoint(self.tmp_dir)
        builder = HistoricalDatasetBuilder(client=client, checkpoint=checkpoint)

        first_run = builder.build_to_list(max_events=1)
        checkpoint.close()
        self.assertEqual(len(first_run), 1)
        processed_id = first_run[0]["event_id"]

        resumed_checkpoint = Checkpoint(self.tmp_dir)
        self.addCleanup(resumed_checkpoint.close)
        client2 = _make_client()
        builder2 = HistoricalDatasetBuilder(client=client2, checkpoint=resumed_checkpoint)

        second_run = builder2.build_to_list()

        self.assertEqual(len(second_run), 1)
        self.assertNotEqual(second_run[0]["event_id"], processed_id)


class TestExtractSeasons(unittest.TestCase):
    """
    Testes de `builder._extract_seasons` — parser de
    `GET /api/v2/leagues/{league_id}/seasons/`.

    Cobre a forma real observada em produção (envelope `{"seasons": [...]}`,
    com `league_id`/`count` adicionais), a forma genérica de compatibilidade
    `{"results": [...]}`, um array JSON simples, e os casos degenerados
    (lista vazia, payload ausente/inesperado).
    """

    def test_seasons_key_returns_all_elements(self):
        payload = {
            "league_id": 8,
            "count": 18,
            "seasons": [{"id": 280, "year": 2025}, {"id": 279, "year": 2024}],
        }
        self.assertEqual(_extract_seasons(payload), [{"id": 280, "year": 2025}, {"id": 279, "year": 2024}])

    def test_seasons_key_with_many_elements_returns_all(self):
        seasons = [{"id": i, "year": 2000 + i} for i in range(18)]
        payload = {"league_id": 8, "count": 18, "seasons": seasons}
        result = _extract_seasons(payload)
        self.assertEqual(len(result), 18)
        self.assertEqual(result, seasons)

    def test_empty_seasons_list_returns_empty_list(self):
        payload = {"league_id": 8, "count": 0, "seasons": []}
        self.assertEqual(_extract_seasons(payload), [])

    def test_results_key_still_supported_for_backward_compatibility(self):
        payload = {"results": [{"id": 1, "year": 2020}]}
        self.assertEqual(_extract_seasons(payload), [{"id": 1, "year": 2020}])

    def test_seasons_key_takes_precedence_over_results_when_both_present(self):
        payload = {"seasons": [{"id": 1}], "results": [{"id": 2}]}
        self.assertEqual(_extract_seasons(payload), [{"id": 1}])

    def test_plain_list_returned_as_is(self):
        payload = [{"id": 1, "year": 2020}, {"id": 2, "year": 2021}]
        self.assertEqual(_extract_seasons(payload), payload)

    def test_empty_plain_list_returns_empty_list(self):
        self.assertEqual(_extract_seasons([]), [])

    def test_none_payload_returns_empty_list(self):
        self.assertEqual(_extract_seasons(None), [])

    def test_dict_without_seasons_or_results_returns_empty_list(self):
        self.assertEqual(_extract_seasons({"league_id": 8, "count": 0}), [])

    def test_unexpected_type_returns_empty_list(self):
        self.assertEqual(_extract_seasons("not a payload"), [])


class TestIterSeasonsParsing(unittest.TestCase):
    """
    Testes de integração leves de `HistoricalDatasetBuilder.iter_seasons`
    com um cliente falso que devolve o payload bruto (dict/list) tal como a
    BSD API o devolveria — ao contrário de `FakeHistoricalClient`
    (usado no resto deste ficheiro), que já devolve listas pré-extraídas.
    """

    class _RawPayloadClient:
        def __init__(self, payload):
            self.payload = payload
            self.calls = []

        def get(self, endpoint, params=None):
            self.calls.append(endpoint)
            return self.payload

    def test_real_seasons_envelope_is_parsed(self):
        payload = {
            "league_id": 8,
            "count": 18,
            "seasons": [{"id": 280, "year": 2025, "name": "2025/2026", "is_current": True}],
        }
        client = self._RawPayloadClient(payload)
        builder = HistoricalDatasetBuilder(client=client)

        seasons = builder.iter_seasons(8)

        self.assertEqual(seasons, [{"id": 280, "year": 2025, "name": "2025/2026", "is_current": True}])
        self.assertEqual(client.calls, ["leagues/8/seasons/"])

    def test_empty_seasons_envelope_returns_empty_list(self):
        client = self._RawPayloadClient({"league_id": 8, "count": 0, "seasons": []})
        builder = HistoricalDatasetBuilder(client=client)

        self.assertEqual(builder.iter_seasons(8), [])

    def test_generic_results_envelope_still_supported(self):
        client = self._RawPayloadClient({"results": [{"id": 5, "year": 2019}]})
        builder = HistoricalDatasetBuilder(client=client)

        self.assertEqual(builder.iter_seasons(38), [{"id": 5, "year": 2019}])


if __name__ == "__main__":
    unittest.main()
