"""
Testes das forças de ataque/defesa por equipa (`src/engine/team_strength.py`)
— Melhoria #5 da auditoria matemática: substituir o prior fixo do
estimador de lambda por forças de ataque/defesa calculadas a partir do
Historical Dataset Builder já existente.

Cobrem:
  - cálculo da força ofensiva (`compute_team_scoring_strength`, `attack_avg`);
  - cálculo da força defensiva (`compute_team_scoring_strength`, `defense_avg`);
  - ausência de fuga de informação (leakage): só jogos anteriores a `before`
    entram no cálculo;
  - fallback correto (`None`) quando a equipa não tem histórico suficiente;
  - estabilidade quando existem poucas observações (1 jogo);
  - `estimate_team_strength_priors`, a combinação ataque/defesa usada como
    prior dinâmico (Nível 0) por `lambda_estimator.py`.
"""

import unittest

import pandas as pd

from src.engine.lambda_estimator import RECENT_MATCH_DECAY_RATE, _exponential_decay_effective_n
from src.engine.team_strength import (
    TeamStrength,
    compute_team_scoring_strength,
    estimate_team_strength_priors,
)


def _df(rows):
    return pd.DataFrame(rows)


class TestComputeTeamScoringStrengthAttack(unittest.TestCase):
    """Força ofensiva: média (ponderada por recência) de golos MARCADOS pela equipa."""

    def test_single_match_attack_equals_goals_scored(self):
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 3, "away_score": 1,
             "date": pd.Timestamp("2024-01-01")},
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"))
        self.assertIsNotNone(strength)
        self.assertAlmostEqual(strength.attack_avg, 3.0)

    def test_attack_counts_goals_regardless_of_venue(self):
        # A marca 2 em casa, depois 4 fora — o ataque de A deve refletir
        # ambos os jogos, não só os jogos em casa.
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0,
             "date": pd.Timestamp("2024-01-01")},
            {"home_team": "C", "away_team": "A", "home_score": 1, "away_score": 4,
             "date": pd.Timestamp("2024-02-01")},
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"), decay_rate=0.0)
        # decay_rate=0 -> pesos uniformes -> média simples = (2+4)/2 = 3.0
        self.assertAlmostEqual(strength.attack_avg, 3.0)

    def test_recent_scoring_form_is_weighted_more_heavily(self):
        # Jogo mais recente (5 golos) deve pesar mais do que o mais antigo (0 golos).
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0,
             "date": pd.Timestamp("2024-01-01")},
            {"home_team": "A", "away_team": "B", "home_score": 5, "away_score": 0,
             "date": pd.Timestamp("2024-05-01")},
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"))
        # média simples seria 2.5; com decaimento, o jogo recente (5) domina.
        self.assertGreater(strength.attack_avg, 2.5)


class TestComputeTeamScoringStrengthDefense(unittest.TestCase):
    """Força defensiva: média (ponderada por recência) de golos SOFRIDOS pela equipa."""

    def test_single_match_defense_equals_goals_conceded(self):
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 3, "away_score": 1,
             "date": pd.Timestamp("2024-01-01")},
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"))
        self.assertAlmostEqual(strength.defense_avg, 1.0)

    def test_defense_counts_goals_conceded_regardless_of_venue(self):
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 1, "away_score": 2,
             "date": pd.Timestamp("2024-01-01")},  # A em casa, sofre 2
            {"home_team": "C", "away_team": "A", "home_score": 4, "away_score": 0,
             "date": pd.Timestamp("2024-02-01")},  # A fora, sofre 4
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"), decay_rate=0.0)
        self.assertAlmostEqual(strength.defense_avg, 3.0)  # (2+4)/2

    def test_attack_and_defense_are_independent_axes(self):
        # Equipa que marca muito e sofre pouco: as duas médias não podem
        # ser iguais nem confundidas uma com a outra.
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 4, "away_score": 0,
             "date": pd.Timestamp("2024-01-01")},
            {"home_team": "A", "away_team": "C", "home_score": 4, "away_score": 0,
             "date": pd.Timestamp("2024-02-01")},
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"), decay_rate=0.0)
        self.assertAlmostEqual(strength.attack_avg, 4.0)
        self.assertAlmostEqual(strength.defense_avg, 0.0)


class TestNoLeakage(unittest.TestCase):
    """Sem fuga de informação: só jogos com `date` estritamente anterior a `before` entram no cálculo."""

    def _df_with_future(self):
        return _df([
            {"home_team": "A", "away_team": "B", "home_score": 1, "away_score": 1,
             "date": pd.Timestamp("2024-01-01")},
            # este jogo é o próprio jogo a prever (ou um posterior) — não
            # pode influenciar a força calculada ANTES dele.
            {"home_team": "A", "away_team": "C", "home_score": 9, "away_score": 0,
             "date": pd.Timestamp("2024-06-01")},
        ])

    def test_matches_on_or_after_before_are_excluded(self):
        df = self._df_with_future()
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"))
        # só o jogo de 2024-01-01 (1 golo) deveria contar; o de 2024-06-01
        # (9 golos) tem de ficar de fora, ou o ataque calculado explodiria.
        self.assertAlmostEqual(strength.attack_avg, 1.0)

    def test_boundary_date_equal_to_before_is_excluded(self):
        # `before` é a data do próprio jogo a prever: um jogo NESSA mesma
        # data (fronteira) não deve ser tratado como "anterior".
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 7, "away_score": 0,
             "date": pd.Timestamp("2024-06-01")},
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"))
        self.assertIsNone(strength)

    def test_only_earlier_matches_shift_the_estimate_over_time(self):
        # A previsão calculada num instante posterior (depois de mais jogos
        # terem ocorrido) pode ser diferente da calculada num instante
        # anterior — mas nunca ao inverso (o passado não pode "ver" o
        # futuro).
        df = self._df_with_future()
        early = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-03-01"))
        late = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-12-01"))
        self.assertAlmostEqual(early.attack_avg, 1.0)
        self.assertGreater(late.attack_avg, early.attack_avg)

    def test_no_before_date_returns_none(self):
        df = self._df_with_future()
        self.assertIsNone(compute_team_scoring_strength(df, "A", None))
        self.assertIsNone(compute_team_scoring_strength(df, "A", pd.NaT))


class TestFallback(unittest.TestCase):
    """Fallback correto: `None` quando não há histórico suficiente para a equipa."""

    def test_team_not_in_dataset_returns_none(self):
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0,
             "date": pd.Timestamp("2024-01-01")},
        ])
        self.assertIsNone(compute_team_scoring_strength(df, "Z", pd.Timestamp("2024-06-01")))

    def test_empty_dataframe_returns_none(self):
        df = pd.DataFrame(columns=["home_team", "away_team", "home_score", "away_score", "date"])
        self.assertIsNone(compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01")))

    def test_missing_required_columns_returns_none_without_raising(self):
        df = pd.DataFrame([{"foo": 1}])
        self.assertIsNone(compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01")))

    def test_invalid_scores_are_skipped_not_raised(self):
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": None, "away_score": None,
             "date": pd.Timestamp("2024-01-01")},
        ])
        self.assertIsNone(compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01")))

    def test_estimate_team_strength_priors_none_when_either_team_unknown(self):
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 1,
             "date": pd.Timestamp("2024-01-01")},
        ])
        # "Z" nunca jogou no dataset -> não há força suficiente para os dois
        # lados da combinação ataque/defesa.
        self.assertIsNone(
            estimate_team_strength_priors(df, "A", "Z", pd.Timestamp("2024-06-01"))
        )
        self.assertIsNone(
            estimate_team_strength_priors(df, "Z", "A", pd.Timestamp("2024-06-01"))
        )


class TestStabilityWithFewObservations(unittest.TestCase):
    """Estabilidade quando existem poucas observações (não deve levantar exceção nem produzir números absurdos)."""

    def test_single_observation_is_stable_and_finite(self):
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 6, "away_score": 0,
             "date": pd.Timestamp("2024-01-01")},
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"))
        self.assertIsNotNone(strength)
        self.assertTrue(strength.attack_avg == strength.attack_avg)  # não é NaN
        self.assertTrue(strength.sample_size == strength.sample_size)
        self.assertGreater(strength.sample_size, 0.0)

    def test_effective_sample_size_matches_lambda_estimator_formula(self):
        # Reutiliza a MESMA amostra efetiva de decaimento já usada por
        # `lambda_estimator._exponential_decay_effective_n` — não uma
        # fórmula nova.
        df = _df([
            {"home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0,
             "date": pd.Timestamp("2024-01-01")},
            {"home_team": "A", "away_team": "C", "home_score": 2, "away_score": 1,
             "date": pd.Timestamp("2024-02-01")},
            {"home_team": "A", "away_team": "D", "home_score": 0, "away_score": 0,
             "date": pd.Timestamp("2024-03-01")},
        ])
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-06-01"))
        expected_n = _exponential_decay_effective_n(3, RECENT_MATCH_DECAY_RATE)
        self.assertAlmostEqual(strength.sample_size, expected_n)

    def test_many_observations_do_not_explode_sample_size(self):
        # n_eff satura (ver lambda_estimator) — não cresce sem limite com o
        # número de jogos históricos, mesmo para uma equipa com dezenas de
        # jogos no dataset.
        rows = [
            {"home_team": "A", "away_team": "B", "home_score": 1, "away_score": 1,
             "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * i)}
            for i in range(60)
        ]
        df = _df(rows)
        strength = compute_team_scoring_strength(df, "A", pd.Timestamp("2024-01-01"))
        self.assertLess(strength.sample_size, 10.0)


class TestEstimateTeamStrengthPriors(unittest.TestCase):
    """Combinação ataque/defesa usada como prior dinâmico (Nível 0) por `lambda_estimator.py`."""

    def _df(self):
        return _df([
            # A: ataque forte (marca 3), defesa fraca (sofre 2)
            {"home_team": "A", "away_team": "X", "home_score": 3, "away_score": 2,
             "date": pd.Timestamp("2024-01-01")},
            # B: ataque fraco (marca 0), defesa forte (sofre 0)
            {"home_team": "B", "away_team": "Y", "home_score": 0, "away_score": 0,
             "date": pd.Timestamp("2024-01-02")},
        ])

    def test_combines_attack_of_one_team_with_defense_of_the_other(self):
        df = self._df()
        priors = estimate_team_strength_priors(df, "A", "B", pd.Timestamp("2024-06-01"), decay_rate=0.0)
        self.assertIsNotNone(priors)
        # golos esperados de A (casa) = média(ataque de A=3.0, defesa de B=0.0) = 1.5
        self.assertAlmostEqual(priors["team_strength_home_goals"], 1.5)
        # golos esperados de B (fora) = média(ataque de B=0.0, defesa de A=2.0) = 1.0
        self.assertAlmostEqual(priors["team_strength_away_goals"], 1.0)

    def test_sample_size_is_bounded_by_the_weaker_known_team(self):
        df = pd.concat([
            self._df(),
            _df([
                {"home_team": "A", "away_team": "Z", "home_score": 2, "away_score": 1,
                 "date": pd.Timestamp("2024-02-01")},
            ]),
        ], ignore_index=True)
        priors = estimate_team_strength_priors(df, "A", "B", pd.Timestamp("2024-06-01"))
        # A tem 2 jogos, B tem 1 -> a amostra combinada não pode exceder a de B.
        b_strength = compute_team_scoring_strength(df, "B", pd.Timestamp("2024-06-01"))
        self.assertAlmostEqual(priors["team_strength_sample_size"], b_strength.sample_size)

    def test_returns_none_without_history_for_both_teams(self):
        df = pd.DataFrame(columns=["home_team", "away_team", "home_score", "away_score", "date"])
        self.assertIsNone(estimate_team_strength_priors(df, "A", "B", pd.Timestamp("2024-06-01")))


if __name__ == "__main__":
    unittest.main()
