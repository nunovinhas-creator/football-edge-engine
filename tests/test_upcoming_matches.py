"""
Testes da Sprint 2 — 🎯 Oportunidades das Próximas 24 Horas
(`src.report.upcoming_matches`).

Cobrem exatamente os requisitos pedidos pela sprint:
  - a janela [agora, agora+24h] é respeitada e a ordenação cronológica
    inicial da busca está correta;
  - o ranking final (Decisão -> Engine Score) está correto;
  - os filtros (Competição/Mercado/Engine Score/Decisão/Hora) funcionam;
  - a pesquisa por equipa/competição funciona;
  - `build_match_snapshot` (Goal Engine, Monte Carlo, Dixon-Coles,
    Machine Learning, Edge, EV, Kelly, Decision Engine) é chamado
    EXATAMENTE uma vez por jogo — nunca recalculado ao "expandir";
  - nenhuma probabilidade/Engine Score/decisão é alterada por esta
    camada de agregação face ao que `build_match_snapshot` devolveu;
  - o bloco de histórico de jogos semelhantes usa exclusivamente as
    funções já existentes do Backtesting Framework.
"""

import copy
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.report import upcoming_matches as um
from src.report.dashboard_data import DEFAULT_BOOKIE_ODD, build_match_snapshot
from src.report.historical_validation import (
    build_current_bet_profile,
    find_similar_bets,
    summarize_similar_bets,
)
from src.backtest.historical.statistics import brier_score


class FakeOddsCollector:
    """Nunca toca na rede: devolve as odds já fornecidas em `odds_by_event`."""

    def __init__(self, odds_by_event=None):
        self.odds_by_event = odds_by_event or {}

    def get_event_odds(self, event_id):
        return self.odds_by_event.get(event_id, {})


def make_event(event_id, home="Casa FC", away="Fora FC", hours_from_now=2.0,
                league_name="Liga Teste", h2h=None, now=None):
    now = now or datetime.now(timezone.utc)
    kickoff = now + timedelta(hours=hours_from_now)
    event = {
        "id": event_id,
        "home_team": home,
        "away_team": away,
        "event_date": kickoff.isoformat(),
        "league_name": league_name,
    }
    if h2h is not None:
        event["head_to_head"] = h2h
    return event


class TestSelectEventsInWindow(unittest.TestCase):
    def test_keeps_only_events_within_the_24h_window_and_sorts_chronologically(self):
        now = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)

        too_early = make_event(1, hours_from_now=-1.0, now=now)  # já começou
        soon = make_event(2, hours_from_now=20.0, now=now)
        very_soon = make_event(3, hours_from_now=1.0, now=now)
        too_late = make_event(4, hours_from_now=25.0, now=now)
        no_date = {"id": 5, "home_team": "X", "away_team": "Y"}

        result = um.select_events_in_window(
            [too_early, soon, very_soon, too_late, no_date], hours=24, now=now
        )

        ids_in_order = [event["id"] for event, _kickoff in result]
        self.assertEqual(ids_in_order, [3, 2])

    def test_boundary_events_are_included(self):
        now = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
        at_now = make_event(1, hours_from_now=0.0, now=now)
        at_horizon = make_event(2, hours_from_now=24.0, now=now)

        result = um.select_events_in_window([at_now, at_horizon], hours=24, now=now)

        self.assertEqual([event["id"] for event, _ in result], [1, 2])


class TestBuildPregameMatchData(unittest.TestCase):
    def test_zeroes_live_only_fields_for_a_match_that_has_not_started(self):
        event = make_event(42, home="FC Porto", away="Benfica")

        match_data = um.build_pregame_match_data(event)

        self.assertEqual(match_data["current_minute"], 0)
        self.assertEqual(match_data["home_score"], 0)
        self.assertEqual(match_data["away_score"], 0)
        self.assertEqual(match_data["dangerous_attacks_10m"], 0)
        self.assertEqual(match_data["shots_10m"], 0)
        self.assertEqual(match_data["shots_on_target_10m"], 0)
        self.assertEqual(match_data["corners_10m"], 0)
        self.assertEqual(match_data["previous_pressure"], 0.0)
        self.assertEqual(match_data["red_cards"], 0)
        self.assertEqual(match_data["home_team"], "FC Porto")
        self.assertEqual(match_data["away_team"], "Benfica")

    def test_uses_default_bookie_odd_when_no_odd_is_available(self):
        event = make_event(1)
        match_data = um.build_pregame_match_data(event, odd=None)
        self.assertEqual(match_data["live_odd_over"], DEFAULT_BOOKIE_ODD)

    def test_uses_provided_odd_when_available(self):
        event = make_event(1)
        match_data = um.build_pregame_match_data(event, odd=2.35)
        self.assertEqual(match_data["live_odd_over"], 2.35)

    def test_expected_goals_come_from_the_official_pregame_lambda_estimator(self):
        h2h = {
            "total_matches": 6,
            "home_goals": 10,
            "away_goals": 4,
            "avg_total_goals": 2.33,
            "home_win_rate": 66.0,
            "away_win_rate": 17.0,
        }
        event = make_event(1, h2h=h2h)

        from src.engine.lambda_estimator import estimate_lambda

        expected_home, expected_away = estimate_lambda(h2h)
        match_data = um.build_pregame_match_data(event, h2h=h2h)

        self.assertEqual(match_data["home_xg_last5"], expected_home)
        self.assertEqual(match_data["away_conceded_xg_last5"], expected_away)


class TestStarRating(unittest.TestCase):
    def test_official_thresholds(self):
        self.assertEqual(um.star_rating(100), "★★★★★")
        self.assertEqual(um.star_rating(90), "★★★★★")
        self.assertEqual(um.star_rating(89.9), "★★★★☆")
        self.assertEqual(um.star_rating(80), "★★★★☆")
        self.assertEqual(um.star_rating(79.9), "★★★☆☆")
        self.assertEqual(um.star_rating(70), "★★★☆☆")
        self.assertEqual(um.star_rating(69.9), "★★☆☆☆")
        self.assertEqual(um.star_rating(60), "★★☆☆☆")
        self.assertEqual(um.star_rating(59.9), "★☆☆☆☆")
        self.assertEqual(um.star_rating(0), "★☆☆☆☆")


class TestMainMonteCarloMarket(unittest.TestCase):
    def test_picks_the_highest_probability_market(self):
        mc = {"over_15": 55.0, "over_25": 78.0, "btts": 60.0}
        self.assertEqual(um.main_monte_carlo_market(mc), "Over 2.5 = 78%")

    def test_ties_prefer_the_first_candidate(self):
        mc = {"over_15": 80.0, "over_25": 80.0, "btts": 10.0}
        self.assertEqual(um.main_monte_carlo_market(mc), "Over 1.5 = 80%")


def _fake_opportunity(match_id, decision_label, score, hour=12, competition="Liga A",
                       market="Próximo Golo (15m)", home="Home", away="Away"):
    return {
        "match_id": match_id,
        "card": {"home_team": home, "away_team": away, "competition": competition},
        "decision": {"label": decision_label, "color": "ok"},
        "engine_score": {"score": score, "color": "ok"},
        "value": {"market": market},
        "kickoff": {"datetime": datetime(2026, 8, 5, hour, 0, tzinfo=timezone.utc), "hour_label": f"{hour:02d}:00"},
        "similar_games": {"n_bets": 0},
        "star_rating": um.star_rating(score),
        "monte_carlo_headline": "Over 1.5 = 50%",
    }


class TestSortOpportunities(unittest.TestCase):
    def test_orders_by_decision_group_then_engine_score_descending(self):
        opportunities = [
            _fake_opportunity(1, "🟡 AGUARDAR", 90),
            _fake_opportunity(2, "🟢 APOSTAR AGORA", 40),
            _fake_opportunity(3, "🔴 NÃO APOSTAR", 99),
            _fake_opportunity(4, "🟢 APOSTAR AGORA", 85),
            _fake_opportunity(5, "🟡 AGUARDAR", 95),
        ]

        result = um.sort_opportunities(opportunities)

        self.assertEqual([o["match_id"] for o in result], [4, 2, 5, 1, 3])


class TestFilterOpportunities(unittest.TestCase):
    def setUp(self):
        self.opportunities = [
            _fake_opportunity(1, "🟢 APOSTAR AGORA", 92, hour=10, competition="Liga A", market="Próximo Golo (15m)"),
            _fake_opportunity(2, "🟡 AGUARDAR", 55, hour=20, competition="Liga B", market="Over 1.5"),
            _fake_opportunity(3, "🔴 NÃO APOSTAR", 30, hour=2, competition="Liga A", market="Próximo Golo (15m)"),
        ]

    def test_filter_by_competition(self):
        result = um.filter_opportunities(self.opportunities, competition="Liga A")
        self.assertEqual({o["match_id"] for o in result}, {1, 3})

    def test_filter_by_market(self):
        result = um.filter_opportunities(self.opportunities, market="Over 1.5")
        self.assertEqual({o["match_id"] for o in result}, {2})

    def test_filter_by_min_engine_score(self):
        result = um.filter_opportunities(self.opportunities, min_engine_score=50)
        self.assertEqual({o["match_id"] for o in result}, {1, 2})

    def test_filter_by_decision(self):
        result = um.filter_opportunities(self.opportunities, decision="🔴 NÃO APOSTAR")
        self.assertEqual({o["match_id"] for o in result}, {3})

    def test_filter_by_hour_range(self):
        result = um.filter_opportunities(self.opportunities, hour_from=8, hour_to=12)
        self.assertEqual({o["match_id"] for o in result}, {1})

    def test_filter_by_hour_range_wrapping_midnight(self):
        result = um.filter_opportunities(self.opportunities, hour_from=22, hour_to=3)
        self.assertEqual({o["match_id"] for o in result}, {3})

    def test_combining_filters(self):
        result = um.filter_opportunities(
            self.opportunities, competition="Liga A", min_engine_score=50
        )
        self.assertEqual({o["match_id"] for o in result}, {1})

    def test_no_filters_returns_everything(self):
        result = um.filter_opportunities(self.opportunities)
        self.assertEqual(len(result), 3)


class TestSearchOpportunities(unittest.TestCase):
    def setUp(self):
        self.opportunities = [
            _fake_opportunity(1, "🟢 APOSTAR AGORA", 90, home="FC Porto", away="Sporting CP", competition="Liga Portugal"),
            _fake_opportunity(2, "🟡 AGUARDAR", 60, home="Real Madrid", away="Barcelona", competition="La Liga"),
        ]

    def test_matches_home_team_case_insensitive(self):
        result = um.search_opportunities(self.opportunities, "porto")
        self.assertEqual({o["match_id"] for o in result}, {1})

    def test_matches_away_team(self):
        result = um.search_opportunities(self.opportunities, "Barcelona")
        self.assertEqual({o["match_id"] for o in result}, {2})

    def test_matches_competition(self):
        result = um.search_opportunities(self.opportunities, "la liga")
        self.assertEqual({o["match_id"] for o in result}, {2})

    def test_empty_query_returns_everything(self):
        result = um.search_opportunities(self.opportunities, "")
        self.assertEqual(len(result), 2)

    def test_none_query_returns_everything(self):
        result = um.search_opportunities(self.opportunities, None)
        self.assertEqual(len(result), 2)

    def test_no_match_returns_empty(self):
        result = um.search_opportunities(self.opportunities, "Manchester")
        self.assertEqual(result, [])


class TestListUpcomingOpportunitiesReusesSnapshotsOnce(unittest.TestCase):
    """
    Requisito de performance da Sprint 2: 'os cálculos dos modelos devem
    ocorrer apenas uma vez por jogo. Nunca recalcular ao expandir um
    cartão. Reutilizar snapshots.'
    """

    def setUp(self):
        # Snapshot real (todos os motores oficiais, sem qualquer mock) usado
        # como molde para as respostas simuladas de build_match_snapshot.
        self.base_snap = build_match_snapshot(
            {
                "match_id": 0,
                "home_team": "Base Home",
                "away_team": "Base Away",
                "current_minute": 0,
                "home_score": 0,
                "away_score": 0,
                "home_xg_last5": 1.6,
                "away_conceded_xg_last5": 1.1,
            }
        )

    def _mock_build_match_snapshot(self, call_log):
        def _fake(match_data, competition="", status_label="", ml_predictor=None, goal_engine=None):
            call_log.append(match_data.get("match_id"))
            snap = copy.deepcopy(self.base_snap)
            snap["match_id"] = match_data.get("match_id")
            snap["card"]["home_team"] = match_data.get("home_team")
            snap["card"]["away_team"] = match_data.get("away_team")
            snap["card"]["competition"] = competition
            return snap

        return _fake

    def test_build_match_snapshot_called_exactly_once_per_event(self):
        now = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
        events = [
            make_event(1, home="A", away="B", hours_from_now=2.0, now=now),
            make_event(2, home="C", away="D", hours_from_now=5.0, now=now),
            make_event(3, home="E", away="F", hours_from_now=10.0, now=now),
        ]

        call_log = []
        with unittest.mock.patch(
            "src.report.upcoming_matches.build_match_snapshot",
            side_effect=self._mock_build_match_snapshot(call_log),
        ):
            opportunities = um.list_upcoming_opportunities(
                hours=24,
                ml_predictor=object(),
                goal_engine=object(),
                all_bets=pd.DataFrame(),
                odds_collector=FakeOddsCollector(),
                events=events,
            )

        self.assertEqual(sorted(call_log), [1, 2, 3])
        self.assertEqual(len(opportunities), 3)

        # A "análise detalhada" (equivalente a expandir o cartão) reutiliza
        # o MESMO objeto snapshot já construído — não invoca o motor de novo.
        for opp in opportunities:
            self.assertIs(opp["snapshot"]["card"], opp["card"])
            self.assertIs(opp["snapshot"]["decision"], opp["decision"])
            self.assertIs(opp["snapshot"]["engine_score"], opp["engine_score"])
            self.assertIs(opp["snapshot"]["value"], opp["value"])

        self.assertEqual(call_log.count(1), 1)
        self.assertEqual(call_log.count(2), 1)
        self.assertEqual(call_log.count(3), 1)

    def test_opportunity_does_not_alter_probabilities_engine_score_or_decision(self):
        now = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)
        events = [make_event(1, home="A", away="B", hours_from_now=2.0, now=now)]

        call_log = []
        with unittest.mock.patch(
            "src.report.upcoming_matches.build_match_snapshot",
            side_effect=self._mock_build_match_snapshot(call_log),
        ):
            opportunities = um.list_upcoming_opportunities(
                hours=24,
                ml_predictor=object(),
                goal_engine=object(),
                all_bets=pd.DataFrame(),
                odds_collector=FakeOddsCollector(),
                events=events,
            )

        opp = opportunities[0]
        expected_snap = copy.deepcopy(self.base_snap)
        expected_snap["match_id"] = 1

        self.assertEqual(opp["engine_score"]["score"], self.base_snap["engine_score"]["score"])
        self.assertEqual(opp["decision"]["label"], self.base_snap["decision"]["label"])
        self.assertEqual(opp["snapshot"]["models"], self.base_snap["models"])
        self.assertEqual(opp["value"]["edge_pct"], self.base_snap["value"]["edge_pct"])
        self.assertEqual(opp["value"]["kelly_pct"], self.base_snap["value"]["kelly_pct"])
        self.assertEqual(opp["star_rating"], um.star_rating(self.base_snap["engine_score"]["score"]))


class TestBuildSimilarGamesSummary(unittest.TestCase):
    """O bloco '📈 Histórico de Jogos Semelhantes' tem de usar exclusivamente
    as funções já existentes do Backtesting Framework — este teste garante
    que o resultado é idêntico a chamar essas funções diretamente (nenhuma
    fórmula nova, nenhum recálculo divergente)."""

    def setUp(self):
        self.snap = build_match_snapshot(
            {
                "match_id": 1,
                "home_team": "FC Porto",
                "away_team": "Sporting CP",
                "current_minute": 0,
                "home_score": 0,
                "away_score": 0,
                "home_xg_last5": 1.7,
                "away_conceded_xg_last5": 1.2,
                "live_odd_over": 1.90,
            },
            competition="Liga Portugal",
        )

        odd = self.snap["value"]["bookie_odd"]
        prob = self.snap["models"]["goal_engine"]["probability"] / 100.0
        rows = []
        for i in range(6):
            won = i % 2 == 0
            stake = 10.0
            profit = stake * (odd - 1) if won else -stake
            rows.append(
                {
                    "odd": odd,
                    "probability": prob,
                    "edge": 0.05,
                    "ev": 0.08,
                    "kelly": 0.02,
                    "stake": stake,
                    "won": won,
                    "profit": profit,
                    "competition": "Liga Portugal",
                    "date": f"2026-01-0{i + 1}",
                }
            )
        self.all_bets = pd.DataFrame(rows)

    def test_matches_direct_calls_to_the_backtesting_framework(self):
        result = um.build_similar_games_summary(self.snap, self.all_bets)

        profile = build_current_bet_profile(self.snap)
        search = find_similar_bets(profile, self.all_bets)
        summary = summarize_similar_bets(search["matches"])
        expected_brier = brier_score(search["matches"])

        self.assertEqual(result["n_bets"], summary["n_bets"])
        self.assertEqual(result["roi_pct"], summary["roi_pct"])
        self.assertEqual(result["yield_pct"], summary["yield_pct"])
        self.assertEqual(result["hit_rate_pct"], summary["hit_rate_pct"])
        self.assertEqual(result["clv_pct"], summary["avg_clv_percentage"])
        self.assertEqual(result["max_drawdown_pct"], summary["max_drawdown_pct"])
        self.assertEqual(result["brier_score"], expected_brier)
        self.assertGreater(result["n_bets"], 0)

    def test_empty_dataset_returns_zeroed_summary_without_error(self):
        result = um.build_similar_games_summary(self.snap, pd.DataFrame())
        self.assertEqual(result["n_bets"], 0)
        self.assertEqual(result["roi_pct"], 0.0)
        self.assertIsNone(result["clv_pct"])


class TestAvailableFilterValues(unittest.TestCase):
    def test_available_competitions_markets_and_decisions_are_deduplicated_and_sorted(self):
        opportunities = [
            _fake_opportunity(1, "🟢 APOSTAR AGORA", 90, competition="Liga B", market="Over 1.5"),
            _fake_opportunity(2, "🟡 AGUARDAR", 60, competition="Liga A", market="Over 1.5"),
            _fake_opportunity(3, "🟢 APOSTAR AGORA", 70, competition="Liga A", market="Próximo Golo (15m)"),
        ]

        self.assertEqual(um.available_competitions(opportunities), ["Liga A", "Liga B"])
        self.assertEqual(um.available_markets(opportunities), ["Over 1.5", "Próximo Golo (15m)"])
        self.assertEqual(um.available_decisions(opportunities), ["🟢 APOSTAR AGORA", "🟡 AGUARDAR"])


if __name__ == "__main__":
    unittest.main()
