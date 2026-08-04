"""
Testes unitários da normalização de jogo+odds+estatísticas
(src/historical_dataset/normalizer.py).

Cobre: mapeamento dos campos de EventDetailV2Schema (schema.yaml),
extração de odds 1X2/Over-Under/BTTS em formas plausíveis distintas,
extração de estatísticas por equipa com preservação de campos
desconhecidos em `extra_*`, e o caso "sem odds/sem estatísticas
disponíveis" (tudo a None, sem levantar exceção).
"""

import json
import unittest

from src.historical_dataset.normalizer import NORMALIZED_COLUMNS, normalize_event


def _event(**overrides):
    base = {
        "id": 555,
        "league_id": 39,
        "season_id": 2024,
        "home_team_id": 10,
        "home_team": "Benfica",
        "away_team_id": 20,
        "away_team": "Porto",
        "event_date": "2024-05-01T20:00:00Z",
        "status": "finished",
        "round_number": 33,
        "round_name": "Round 33",
        "home_score": 2,
        "away_score": 1,
        "home_score_ht": 1,
        "away_score_ht": 0,
        "venue_id": 7,
    }
    base.update(overrides)
    return base


class TestNormalizeEventCoreFields(unittest.TestCase):

    def test_maps_event_detail_fields(self):
        record = normalize_event(_event(), league={"id": 39, "name": "Primeira Liga"}, season={"id": 2024, "name": "2023/2024"})

        self.assertEqual(record["event_id"], 555)
        self.assertEqual(record["competition_id"], 39)
        self.assertEqual(record["competition"], "Primeira Liga")
        self.assertEqual(record["season_id"], 2024)
        self.assertEqual(record["season"], "2023/2024")
        self.assertEqual(record["home_team"], "Benfica")
        self.assertEqual(record["away_team"], "Porto")
        self.assertEqual(record["home_score"], 2)
        self.assertEqual(record["away_score"], 1)
        self.assertEqual(record["home_score_ht"], 1)
        self.assertEqual(record["away_score_ht"], 0)
        self.assertEqual(record["date"], "2024-05-01T20:00:00Z")

    def test_all_normalized_columns_present(self):
        record = normalize_event(_event())
        self.assertEqual(set(record.keys()), set(NORMALIZED_COLUMNS))

    def test_missing_odds_and_stats_are_none_not_raising(self):
        record = normalize_event(_event(), odds=None, stats=None)

        for col in ("odds_home", "odds_draw", "odds_away", "odds_btts_yes",
                     "cards_home_yellow", "corners_home"):
            self.assertIsNone(record[col])
        self.assertIsNone(record["bookmaker"])


class TestNormalizeOdds(unittest.TestCase):

    def test_nested_1x2_and_over_under_and_btts(self):
        odds = {
            "1x2": {"home": 1.9, "draw": 3.4, "away": 4.1},
            "over_under": {
                "2.5": {"over": 1.85, "under": 1.95},
                "1.5": {"over": 1.3, "under": 3.2},
            },
            "btts": {"yes": 1.7, "no": 2.1},
        }

        record = normalize_event(_event(), odds=odds)

        self.assertEqual(record["odds_home"], 1.9)
        self.assertEqual(record["odds_draw"], 3.4)
        self.assertEqual(record["odds_away"], 4.1)
        self.assertEqual(record["odds_over_2_5"], 1.85)
        self.assertEqual(record["odds_under_2_5"], 1.95)
        self.assertEqual(record["odds_over_1_5"], 1.3)
        self.assertEqual(record["odds_btts_yes"], 1.7)
        self.assertEqual(record["odds_btts_no"], 2.1)
        self.assertEqual(record["bookmaker"], "consensus")

    def test_flat_uppercase_1x2(self):
        odds = {"HOME": 2.0, "DRAW": 3.3, "AWAY": 3.6}

        record = normalize_event(_event(), odds=odds)

        self.assertEqual(record["odds_home"], 2.0)
        self.assertEqual(record["odds_draw"], 3.3)
        self.assertEqual(record["odds_away"], 3.6)

    def test_flat_over_under_keys(self):
        odds = {"over_2_5": 1.9, "under_2_5": 1.9}

        record = normalize_event(_event(), odds=odds)

        self.assertEqual(record["odds_over_2_5"], 1.9)
        self.assertEqual(record["odds_under_2_5"], 1.9)

    def test_bookmakers_available_from_comparison(self):
        comparison = {
            "markets": {
                "1x2": {"bet365": {"HOME": 1.9, "DRAW": 3.4, "AWAY": 4.1}, "pinnacle": {"HOME": 1.95}},
                "btts": {"bet365": {"YES": 1.7, "NO": 2.1}},
            }
        }

        record = normalize_event(_event(), odds={"1x2": {"home": 1.9, "draw": 3.4, "away": 4.1}}, odds_comparison=comparison)

        self.assertEqual(record["bookmakers_available"], "bet365,pinnacle")

    def test_no_comparison_leaves_bookmakers_available_none(self):
        record = normalize_event(_event(), odds={"1x2": {"home": 1.9, "draw": 3.4, "away": 4.1}})
        self.assertIsNone(record["bookmakers_available"])


class TestNormalizeOddsRealBSDShape(unittest.TestCase):
    """
    Forma real, plana, devolvida por `/events/{id}/odds/` (confirmada por
    execução real do builder — ver auditoria da normalização de odds):
    `home_win`/`draw`/`away_win`, `over_XX_goals`/`under_XX_goals` e
    `btts_yes`/`btts_no`. Distinta das formas "plausíveis" já cobertas em
    `TestNormalizeOdds` (nested `1x2`/`over_under`/`btts`, ou flat
    `HOME`/`DRAW`/`AWAY` e `over_2_5`/`under_2_5`).
    """

    REAL_ODDS = {
        "home_win": 1.12,
        "draw": 8.20,
        "away_win": 15.93,
        "over_15_goals": 1.05,
        "under_15_goals": 6.50,
        "over_25_goals": 1.36,
        "under_25_goals": 2.89,
        "over_35_goals": 2.75,
        "under_35_goals": 1.42,
        "btts_yes": 2.15,
        "btts_no": 1.63,
    }

    def test_extra_odds_maps_to_odds_home(self):
        record = normalize_event(_event(), odds=self.REAL_ODDS)
        self.assertEqual(record["odds_home"], 1.12)

    def test_extra_odds_maps_to_odds_draw(self):
        record = normalize_event(_event(), odds=self.REAL_ODDS)
        self.assertEqual(record["odds_draw"], 8.20)

    def test_extra_odds_maps_to_odds_away(self):
        record = normalize_event(_event(), odds=self.REAL_ODDS)
        self.assertEqual(record["odds_away"], 15.93)

    def test_over_under_all_thresholds_mapped(self):
        record = normalize_event(_event(), odds=self.REAL_ODDS)
        self.assertEqual(record["odds_over_1_5"], 1.05)
        self.assertEqual(record["odds_under_1_5"], 6.50)
        self.assertEqual(record["odds_over_2_5"], 1.36)
        self.assertEqual(record["odds_under_2_5"], 2.89)
        self.assertEqual(record["odds_over_3_5"], 2.75)
        self.assertEqual(record["odds_under_3_5"], 1.42)

    def test_btts_mapped(self):
        record = normalize_event(_event(), odds=self.REAL_ODDS)
        self.assertEqual(record["odds_btts_yes"], 2.15)
        self.assertEqual(record["odds_btts_no"], 1.63)

    def test_no_odds_available_leaves_all_odds_columns_none(self):
        """Ausência de odds (jogo sem odds publicadas) não deve falhar nem inventar valores."""
        record = normalize_event(_event(), odds=None)
        for col in (
            "odds_home", "odds_draw", "odds_away",
            "odds_over_1_5", "odds_under_1_5",
            "odds_over_2_5", "odds_under_2_5",
            "odds_over_3_5", "odds_under_3_5",
            "odds_btts_yes", "odds_btts_no",
        ):
            self.assertIsNone(record[col])
        self.assertIsNone(record["extra_odds"])
        self.assertIsNone(record["bookmaker"])

    def test_empty_odds_json_leaves_all_odds_columns_none(self):
        """JSON vazio (`{}`) — distinto de `None` — também não deve inventar valores."""
        record = normalize_event(_event(), odds={})
        for col in (
            "odds_home", "odds_draw", "odds_away",
            "odds_over_1_5", "odds_under_1_5",
            "odds_over_2_5", "odds_under_2_5",
            "odds_over_3_5", "odds_under_3_5",
            "odds_btts_yes", "odds_btts_no",
        ):
            self.assertIsNone(record[col])
        self.assertIsNone(record["extra_odds"])

    def test_extra_odds_preserves_raw_json_unchanged(self):
        """`extra_odds` deve continuar a refletir o payload bruto tal como veio da API."""
        record = normalize_event(_event(), odds=self.REAL_ODDS)
        self.assertEqual(json.loads(record["extra_odds"]), self.REAL_ODDS)

    def test_old_nested_shape_still_supported(self):
        """Compatibilidade com datasets antigos: a forma nested já suportada continua a funcionar."""
        odds = {
            "1x2": {"home": 1.9, "draw": 3.4, "away": 4.1},
            "over_under": {"2.5": {"over": 1.85, "under": 1.95}},
            "btts": {"yes": 1.7, "no": 2.1},
        }
        record = normalize_event(_event(), odds=odds)
        self.assertEqual(record["odds_home"], 1.9)
        self.assertEqual(record["odds_draw"], 3.4)
        self.assertEqual(record["odds_away"], 4.1)
        self.assertEqual(record["odds_over_2_5"], 1.85)
        self.assertEqual(record["odds_under_2_5"], 1.95)
        self.assertEqual(record["odds_btts_yes"], 1.7)
        self.assertEqual(record["odds_btts_no"], 2.1)

    def test_old_flat_uppercase_shape_still_supported(self):
        """Compatibilidade com datasets antigos: a forma flat HOME/DRAW/AWAY continua a funcionar."""
        odds = {"HOME": 2.0, "DRAW": 3.3, "AWAY": 3.6, "over_2_5": 1.9, "under_2_5": 1.9}
        record = normalize_event(_event(), odds=odds)
        self.assertEqual(record["odds_home"], 2.0)
        self.assertEqual(record["odds_draw"], 3.3)
        self.assertEqual(record["odds_away"], 3.6)
        self.assertEqual(record["odds_over_2_5"], 1.9)
        self.assertEqual(record["odds_under_2_5"], 1.9)


class TestNormalizeOddsRealBSDEnvelope(unittest.TestCase):
    """
    Forma *realmente* devolvida pela BSD API: o dict plano de
    `TestNormalizeOddsRealBSDShape` vem, na prática, embrulhado sob a
    chave "odds" — `{"odds": {home_win: ..., ...}}` — confirmado por
    código de produção ativo (`main.py live`):
    `analysis["odds"]["odds"]["over_15_goals"]` em `src/cli/live.py` e
    `scripts/live_scanner.py`, onde `analysis["odds"]` é o JSON devolvido
    sem alterações por `APIOddsProvider.get_live_odds()` (mesma chamada a
    `GET /events/{id}/odds/` usada por este builder). Este invólucro não
    era desembrulhado antes desta correção, pelo que todas as colunas de
    odds ficavam a None mesmo com os aliases corretos.
    """

    WRAPPED_REAL_ODDS = {"odds": dict(TestNormalizeOddsRealBSDShape.REAL_ODDS)}

    def test_unwraps_odds_envelope_for_1x2(self):
        record = normalize_event(_event(), odds=self.WRAPPED_REAL_ODDS)
        self.assertEqual(record["odds_home"], 1.12)
        self.assertEqual(record["odds_draw"], 8.20)
        self.assertEqual(record["odds_away"], 15.93)

    def test_unwraps_odds_envelope_for_over_under(self):
        record = normalize_event(_event(), odds=self.WRAPPED_REAL_ODDS)
        self.assertEqual(record["odds_over_1_5"], 1.05)
        self.assertEqual(record["odds_under_1_5"], 6.50)
        self.assertEqual(record["odds_over_2_5"], 1.36)
        self.assertEqual(record["odds_under_2_5"], 2.89)
        self.assertEqual(record["odds_over_3_5"], 2.75)
        self.assertEqual(record["odds_under_3_5"], 1.42)

    def test_unwraps_odds_envelope_for_btts(self):
        record = normalize_event(_event(), odds=self.WRAPPED_REAL_ODDS)
        self.assertEqual(record["odds_btts_yes"], 2.15)
        self.assertEqual(record["odds_btts_no"], 1.63)

    def test_unwraps_odds_envelope_sets_bookmaker_consensus(self):
        record = normalize_event(_event(), odds=self.WRAPPED_REAL_ODDS)
        self.assertEqual(record["bookmaker"], "consensus")

    def test_extra_odds_preserves_raw_envelope_unchanged(self):
        """`extra_odds` deve continuar a guardar o payload bruto (com invólucro), não o desembrulhado."""
        record = normalize_event(_event(), odds=self.WRAPPED_REAL_ODDS)
        self.assertEqual(json.loads(record["extra_odds"]), self.WRAPPED_REAL_ODDS)

    def test_empty_odds_envelope_leaves_columns_none(self):
        """`{"odds": {}}` (jogo sem odds publicadas) não deve inventar valores nem marcar bookmaker."""
        record = normalize_event(_event(), odds={"odds": {}})
        for col in (
            "odds_home", "odds_draw", "odds_away",
            "odds_over_1_5", "odds_under_1_5",
            "odds_over_2_5", "odds_under_2_5",
            "odds_over_3_5", "odds_under_3_5",
            "odds_btts_yes", "odds_btts_no",
        ):
            self.assertIsNone(record[col])
        self.assertIsNone(record["bookmaker"])

    def test_already_unwrapped_shape_still_supported(self):
        """Compatibilidade: passar o dict já desembrulhado diretamente continua a funcionar."""
        record = normalize_event(_event(), odds=dict(TestNormalizeOddsRealBSDShape.REAL_ODDS))
        self.assertEqual(record["odds_home"], 1.12)
        self.assertEqual(record["odds_away"], 15.93)


class TestNormalizeStats(unittest.TestCase):

    def test_home_away_containers_with_known_aliases(self):
        stats = {
            "home": {"yellow_cards": 2, "red_cards": 0, "corners": 6, "shots_total": 14, "shots_on_target": 5,
                      "possession": 55, "fouls": 10, "offsides": 1},
            "away": {"yellow_cards": 3, "red_cards": 1, "corners": 3, "shots_total": 8, "shots_on_target": 2,
                      "possession": 45, "fouls": 14, "offsides": 2},
        }

        record = normalize_event(_event(), stats=stats)

        self.assertEqual(record["cards_home_yellow"], 2)
        self.assertEqual(record["cards_home_red"], 0)
        self.assertEqual(record["cards_away_yellow"], 3)
        self.assertEqual(record["cards_away_red"], 1)
        self.assertEqual(record["corners_home"], 6)
        self.assertEqual(record["corners_away"], 3)
        self.assertEqual(record["shots_home"], 14)
        self.assertEqual(record["shots_on_target_away"], 2)
        self.assertEqual(record["possession_home"], 55)
        self.assertEqual(record["fouls_away"], 14)
        self.assertEqual(record["offsides_home"], 1)

    def test_alias_variants_are_recognized(self):
        stats = {
            "home": {"yellowcards": 1, "redcards": 0, "corner_kicks": 4, "total_shots": 9,
                      "shots_on_goal": 3, "ball_possession": 60, "fouls_committed": 8},
            "away": {},
        }

        record = normalize_event(_event(), stats=stats)

        self.assertEqual(record["cards_home_yellow"], 1)
        self.assertEqual(record["corners_home"], 4)
        self.assertEqual(record["shots_home"], 9)
        self.assertEqual(record["shots_on_target_home"], 3)
        self.assertEqual(record["possession_home"], 60)
        self.assertEqual(record["fouls_home"], 8)

    def test_unknown_stat_fields_preserved_in_extra_columns(self):
        stats = {
            "home": {"yellow_cards": 1, "xg": 1.85, "big_chances_created": 3},
            "away": {"yellow_cards": 2, "xg": 0.9},
        }

        record = normalize_event(_event(), stats=stats)

        extra_home = json.loads(record["extra_stats_home"])
        extra_away = json.loads(record["extra_stats_away"])

        self.assertEqual(extra_home["xg"], 1.85)
        self.assertEqual(extra_home["big_chances_created"], 3)
        self.assertNotIn("yellow_cards", extra_home)
        self.assertEqual(extra_away["xg"], 0.9)

    def test_match_level_extra_stats_preserved(self):
        stats = {
            "home": {"yellow_cards": 1},
            "away": {"yellow_cards": 1},
            "momentum": [1, 2, 3],
            "average_positions": {"home": [], "away": []},
        }

        record = normalize_event(_event(), stats=stats)

        extra_match = json.loads(record["extra_match_stats"])
        self.assertIn("momentum", extra_match)
        self.assertIn("average_positions", extra_match)

    def test_teams_list_shape_supported(self):
        stats = {
            "teams": [
                {"is_home": True, "yellow_cards": 1},
                {"is_home": False, "yellow_cards": 2},
            ]
        }

        record = normalize_event(_event(), stats=stats)

        self.assertEqual(record["cards_home_yellow"], 1)
        self.assertEqual(record["cards_away_yellow"], 2)


class TestNormalizeStatsRealBSDEnvelope(unittest.TestCase):
    """
    Forma *realmente* devolvida pela BSD API: `/events/{id}/stats/` vem
    embrulhado sob a chave "stats" — `{"stats": {"home": ..., "away": ...},
    ...outros campos ao nível do jogo}` — confirmado por
    `research/pressure_shots/build_raw_table.py`
    (`stats.get("stats").get("home")`), validado contra respostas reais
    da API (74 ligas testadas, ver `research/pressure_shots/README.md`).
    Este invólucro não era desembrulhado antes desta correção, pelo que
    `_team_containers` nunca encontrava "home"/"away" e todas as colunas
    de estatísticas ficavam a None.
    """

    WRAPPED_STATS = {
        "stats": {
            "home": {"yellow_cards": 2, "corners": 6, "shots_total": 14, "possession": 55},
            "away": {"yellow_cards": 3, "corners": 3, "shots_total": 8, "possession": 45},
        },
        "shotmap": [{"minute": 10, "team": "home"}],
        "momentum": [1, 2, 3],
    }

    def test_unwraps_stats_envelope_for_team_stats(self):
        record = normalize_event(_event(), stats=self.WRAPPED_STATS)

        self.assertEqual(record["cards_home_yellow"], 2)
        self.assertEqual(record["cards_away_yellow"], 3)
        self.assertEqual(record["corners_home"], 6)
        self.assertEqual(record["corners_away"], 3)
        self.assertEqual(record["shots_home"], 14)
        self.assertEqual(record["possession_away"], 45)

    def test_unwraps_stats_envelope_preserves_match_level_extras(self):
        """shotmap/momentum (irmãos de "stats" no payload real) continuam em extra_match_stats."""
        record = normalize_event(_event(), stats=self.WRAPPED_STATS)

        extra_match = json.loads(record["extra_match_stats"])
        self.assertIn("shotmap", extra_match)
        self.assertIn("momentum", extra_match)
        self.assertNotIn("stats", extra_match)

    def test_empty_stats_envelope_leaves_columns_none(self):
        record = normalize_event(_event(), stats={"stats": {}})
        for col in ("cards_home_yellow", "corners_home", "shots_home", "possession_home"):
            self.assertIsNone(record[col])

    def test_already_unwrapped_shape_still_supported(self):
        """Compatibilidade: passar {"home": ..., "away": ...} diretamente (sem invólucro) continua a funcionar."""
        stats = {
            "home": {"yellow_cards": 1, "corners": 5},
            "away": {"yellow_cards": 2, "corners": 2},
        }
        record = normalize_event(_event(), stats=stats)
        self.assertEqual(record["cards_home_yellow"], 1)
        self.assertEqual(record["corners_away"], 2)


if __name__ == "__main__":
    unittest.main()
