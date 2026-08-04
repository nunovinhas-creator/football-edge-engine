"""
Testes do painel "📈 Validação Histórica da Aposta Atual"
(`src.report.historical_validation`).

Cobrem:
  - pesquisa de jogos semelhantes (bandas de tolerância, alargamento
    progressivo, filtro de competição);
  - reaproveitamento de `src.backtest.historical.metrics.summary_metrics`
    para o subconjunto encontrado (mesma função usada pelo Backtesting
    Global — nenhuma métrica financeira nova);
  - veredicto (positivo / negativo / dados insuficientes);
  - integração ponta-a-ponta com `build_match_snapshot` (o mesmo
    snapshot consumido pelo resto do Dashboard Pro) e com
    `run_demo_backtest` (o mesmo `BacktestReport` do painel de
    Backtesting Global).
"""

import unittest

import pandas as pd

from src.report.historical_validation import (
    CurrentBetProfile,
    build_comparison,
    build_current_bet_profile,
    build_historical_validation,
    build_validation_explanation,
    build_verdict,
    find_similar_bets,
    summarize_similar_bets,
)


def _historical_row(odd, probability, competition="Premier League", won=True, profit=None, stake=1.0, **kwargs):
    if profit is None:
        profit = stake * (odd - 1.0) if won else -stake
    row = {
        "match": "A vs B",
        "date": "2020-01-01",
        "market": "HOME",
        "competition": competition,
        "odd": odd,
        "probability": probability,
        "edge": round(probability - 1.0 / odd, 4),
        "ev": round(probability * odd - 1.0, 4),
        "kelly": 0.1,
        "stake": stake,
        "won": won,
        "profit": profit,
    }
    row.update(kwargs)
    return row


def _profile(**overrides) -> CurrentBetProfile:
    base = dict(
        market="Próximo Golo (15m)",
        odd=2.0,
        probability_pct=60.0,
        edge_pct=10.0,
        ev_pct=20.0,
        kelly_pct=5.0,
        confidence_label="Alta",
        confidence_score=80.0,
        consensus_label="Forte",
        consensus_gap=5.0,
        competition="Premier League",
    )
    base.update(overrides)
    return CurrentBetProfile(**base)


class TestFindSimilarBets(unittest.TestCase):

    def test_finds_bets_within_tight_tolerance(self):
        df = pd.DataFrame(
            [
                _historical_row(odd=2.05, probability=0.61, won=True),
                _historical_row(odd=1.95, probability=0.58, won=True),
                _historical_row(odd=2.1, probability=0.62, won=False),
                _historical_row(odd=9.0, probability=0.10, won=False),  # muito diferente
            ]
        )
        result = find_similar_bets(_profile(), df, min_sample=3)
        self.assertEqual(len(result["matches"]), 3)
        self.assertNotIn(9.0, result["matches"]["odd"].tolist())

    def test_widens_tolerance_when_sample_too_small(self):
        df = pd.DataFrame(
            [
                _historical_row(odd=2.0, probability=0.60, won=True),
                _historical_row(odd=3.5, probability=0.30, won=False, competition="La Liga"),
                _historical_row(odd=4.0, probability=0.25, won=False, competition="La Liga"),
            ]
        )
        result = find_similar_bets(_profile(), df, min_sample=3)
        self.assertGreaterEqual(len(result["matches"]), 1)
        self.assertIsNotNone(result["odd_tolerance_pct"])

    def test_empty_dataset_returns_empty_matches(self):
        result = find_similar_bets(_profile(), pd.DataFrame())
        self.assertTrue(result["matches"].empty)
        self.assertEqual(result["criteria_applied"], [])

    def test_lists_unavailable_criteria_never_fabricated(self):
        df = pd.DataFrame([_historical_row(odd=2.0, probability=0.60, won=True)])
        result = find_similar_bets(_profile(), df, min_sample=1)
        self.assertIn("λ (lambda) semelhantes", result["criteria_unavailable"])
        self.assertIn("Estado do marcador semelhante", result["criteria_unavailable"])


class TestSummarizeAndCompare(unittest.TestCase):

    def test_summary_reuses_official_metrics_columns(self):
        df = pd.DataFrame(
            [
                _historical_row(odd=2.0, probability=0.6, won=True),
                _historical_row(odd=2.0, probability=0.6, won=False),
            ]
        )
        summary = summarize_similar_bets(df)
        self.assertEqual(summary["n_bets"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertIn("roi_pct", summary)
        self.assertIn("max_drawdown_pct", summary)
        self.assertIn("profit_factor", summary)

    def test_comparison_has_no_historical_avg_when_no_matches(self):
        summary = summarize_similar_bets(pd.DataFrame())
        comparison = build_comparison(_profile(), summary)
        for row in comparison:
            self.assertIsNone(row["historical_avg"])


class TestVerdict(unittest.TestCase):

    def test_insufficient_sample_is_neutral(self):
        summary = {"n_bets": 2, "roi_pct": 50.0, "hit_rate_pct": 100.0}
        verdict = build_verdict(summary)
        self.assertEqual(verdict["color"], "warn")

    def test_positive_roi_with_enough_sample_is_positive(self):
        summary = {"n_bets": 10, "roi_pct": 15.0, "hit_rate_pct": 60.0, "max_drawdown_pct": -5.0}
        verdict = build_verdict(summary)
        self.assertEqual(verdict["color"], "ok")
        explanation = build_validation_explanation(verdict, summary)
        self.assertIn("ROI positivo", explanation)

    def test_negative_roi_with_enough_sample_is_negative(self):
        summary = {"n_bets": 10, "roi_pct": -8.0, "hit_rate_pct": 30.0, "max_drawdown_pct": -20.0}
        verdict = build_verdict(summary)
        self.assertEqual(verdict["color"], "off")
        explanation = build_validation_explanation(verdict, summary)
        self.assertIn("cautela", explanation)


class TestEndToEndWithRealEngineModules(unittest.TestCase):
    """
    Integração com o motor real (sem mocks): `build_match_snapshot`
    (Goal Engine, Monte Carlo, Dixon-Coles, ML, Edge, EV, Kelly, Decision
    Engine) e `run_demo_backtest` (BacktestEngine sobre o dataset real de
    demonstração) — os módulos matemáticos não são alterados nem
    substituídos, apenas consumidos.
    """

    def test_build_historical_validation_end_to_end(self):
        from src.report.dashboard_data import DEMO_MATCH_DATA, build_match_snapshot, run_demo_backtest

        snap = build_match_snapshot(DEMO_MATCH_DATA, competition="Premier League")
        report = run_demo_backtest()

        result = build_historical_validation(snap, report.all_bets)

        self.assertIn("profile", result)
        self.assertIn("search", result)
        self.assertIn("summary", result)
        self.assertIn("comparison", result)
        self.assertIn("verdict", result)
        self.assertIn("explanation", result)

        # A probabilidade/edge/ev/kelly da aposta atual têm de ser
        # EXATAMENTE as já calculadas pelo motor no snapshot — nenhum
        # valor novo é gerado por este módulo de apresentação.
        profile = result["profile"]
        self.assertEqual(profile.odd, snap["value"]["bookie_odd"])
        self.assertEqual(profile.edge_pct, snap["value"]["edge_pct"])
        self.assertEqual(profile.kelly_pct, snap["value"]["kelly_pct"])
        self.assertEqual(profile.probability_pct, snap["models"]["goal_engine"]["probability"])

        self.assertIn(result["verdict"]["color"], {"ok", "warn", "off"})


def test_module_only_imports_presentation_and_metrics_helpers():
    """
    Confirma, por inspeção do próprio módulo, que nenhuma fórmula
    matemática do motor (Dixon-Coles, Monte Carlo, Machine Learning,
    Kelly, Edge, EV, Decision Engine) é importada/recalculada aqui — só
    `src.backtest.historical.metrics` (as mesmas funções do Backtesting
    Global).
    """
    import src.report.historical_validation as module

    forbidden_modules = (
        "src.engine.dixon_coles",
        "src.engine.simulation",
        "src.engine.kelly",
        "src.engine.edge",
        "src.engine.decision",
        "src.model.ml_predictor",
        "src.live.engine",
    )
    with open(module.__file__, encoding="utf-8") as fh:
        import_lines = [
            line for line in fh.readlines() if line.strip().startswith(("import ", "from "))
        ]
    for forbidden in forbidden_modules:
        for line in import_lines:
            assert forbidden not in line, f"{forbidden} não deveria ser importado por historical_validation.py: {line!r}"


if __name__ == "__main__":
    unittest.main()
