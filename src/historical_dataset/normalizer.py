"""Normalização de jogos, odds e estatísticas da BSD API para um registo plano único.

`schema.yaml` documenta a forma exata da resposta para `/events/`,
`/events/{id}/` e `/leagues/{id}/seasons/` (ver `EventDetailV2Schema` /
`_SeasonSchema`), mas marca `/events/{id}/odds/`, `/events/{id}/odds/comparison/`
e `/events/{id}/stats/` como "No response body" — a forma exata destas três
respostas não está documentada na especificação OpenAPI incluída no
repositório. Este módulo assume, com base nas descrições textuais desses
endpoints (ver `docs/07_historical_dataset_builder.md`, secção
"Limitações"), uma forma plausível e faz a extração de forma defensiva:
qualquer campo que não corresponda a um dos aliases conhecidos é
preservado (não descartado) nas colunas `extra_*` como JSON.

Nenhuma probabilidade, edge, EV, Kelly ou output de modelo é calculado
aqui — apenas leitura/normalização de dados brutos já existentes na API.
"""

import json
from typing import Any, Dict, Optional, Tuple

# Ordem canónica das colunas do dataset normalizado (usada pelos exporters
# em storage.py para produzir CSV/SQLite/Parquet com um esquema estável).
NORMALIZED_COLUMNS = [
    "event_id",
    "competition_id",
    "competition",
    "season_id",
    "season",
    "round_number",
    "round_name",
    "date",
    "status",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_score",
    "away_score",
    "home_score_ht",
    "away_score_ht",
    "venue_id",
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_over_1_5",
    "odds_under_1_5",
    "odds_over_2_5",
    "odds_under_2_5",
    "odds_over_3_5",
    "odds_under_3_5",
    "odds_btts_yes",
    "odds_btts_no",
    "bookmaker",
    "bookmakers_available",
    "cards_home_yellow",
    "cards_home_red",
    "cards_away_yellow",
    "cards_away_red",
    "corners_home",
    "corners_away",
    "shots_home",
    "shots_away",
    "shots_on_target_home",
    "shots_on_target_away",
    "possession_home",
    "possession_away",
    "fouls_home",
    "fouls_away",
    "offsides_home",
    "offsides_away",
    "extra_stats_home",
    "extra_stats_away",
    "extra_match_stats",
    "extra_odds",
]

# Aliases conhecidos por estatística, procurados (case-insensitive) dentro
# do sub-dicionário por equipa devolvido por /events/{id}/stats/. Qualquer
# campo do payload que não corresponda a nenhum destes aliases é preservado
# em `extra_stats_home`/`extra_stats_away` (ver `_leftover`).
STAT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "yellow_cards": ("yellow_cards", "yellowcards", "cards_yellow", "yellow"),
    "red_cards": ("red_cards", "redcards", "cards_red", "red"),
    "corners": ("corners", "corner_kicks"),
    "shots_total": ("shots_total", "total_shots", "shots"),
    "shots_on_target": ("shots_on_target", "shots_on_goal", "on_target_shots"),
    "possession": ("possession", "ball_possession", "possession_pct"),
    "fouls": ("fouls", "fouls_committed"),
    "offsides": ("offsides",),
}


def _lookup(container: Any, *candidates: str) -> Optional[Any]:
    """Procura o primeiro de `candidates` num dict, ignorando maiúsculas/minúsculas."""
    if not isinstance(container, dict):
        return None
    for candidate in candidates:
        if candidate in container and container[candidate] is not None:
            return container[candidate]
    lowered = {str(k).lower(): v for k, v in container.items()}
    for candidate in candidates:
        value = lowered.get(candidate.lower())
        if value is not None:
            return value
    return None


def _extract_1x2(odds: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not isinstance(odds, dict):
        return None, None, None
    container = odds.get("1x2") or odds.get("1X2") or odds.get("match_winner") or odds
    home = _lookup(container, "home", "HOME", "1")
    draw = _lookup(container, "draw", "DRAW", "x", "X")
    away = _lookup(container, "away", "AWAY", "2")
    return home, draw, away


def _extract_over_under(odds: Any, threshold: str) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(odds, dict):
        return None, None

    suffix = threshold.replace(".", "_")
    container = odds.get("over_under") or odds.get(f"over_under_{suffix}")

    sub = None
    if isinstance(container, dict):
        sub = container.get(threshold) or container.get(f"over_under_{suffix}") or container.get(f"OU_{suffix}")

    if sub is None:
        over = _lookup(odds, f"over_{suffix}", f"OVER_{suffix}")
        under = _lookup(odds, f"under_{suffix}", f"UNDER_{suffix}")
        return over, under

    over = _lookup(sub, "over", "OVER")
    under = _lookup(sub, "under", "UNDER")
    return over, under


def _extract_btts(odds: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(odds, dict):
        return None, None
    container = odds.get("btts") or odds.get("BTTS") or odds
    yes = _lookup(container, "yes", "YES", "btts_yes")
    no = _lookup(container, "no", "NO", "btts_no")
    return yes, no


def _extract_bookmakers(comparison: Any) -> list:
    """Devolve os slugs de bookmaker vistos em qualquer mercado de `/odds/comparison/`."""
    if not isinstance(comparison, dict):
        return []
    markets = comparison.get("markets")
    if not isinstance(markets, dict):
        return []
    bookmakers = set()
    for market_data in markets.values():
        if isinstance(market_data, dict):
            bookmakers.update(market_data.keys())
    return sorted(bookmakers)


def _team_containers(stats: Any) -> Tuple[Optional[dict], Optional[dict]]:
    if not isinstance(stats, dict):
        return None, None
    if "home" in stats or "away" in stats:
        return stats.get("home"), stats.get("away")
    teams = stats.get("teams")
    if isinstance(teams, list):
        home = next((t for t in teams if t.get("is_home") is True or t.get("side") in ("home", "HOME")), None)
        away = next((t for t in teams if t.get("is_home") is False or t.get("side") in ("away", "AWAY")), None)
        return home, away
    return None, None


def _extract_team_stat(container: Optional[dict], stat_name: str) -> Optional[Any]:
    if not isinstance(container, dict):
        return None
    return _lookup(container, *STAT_ALIASES.get(stat_name, (stat_name,)))


def _leftover(container: Optional[dict], known_aliases: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    if not isinstance(container, dict):
        return {}
    known_keys = {alias.lower() for aliases in known_aliases.values() for alias in aliases}
    return {k: v for k, v in container.items() if str(k).lower() not in known_keys}


def _json_or_none(value: Any) -> Optional[str]:
    if not value:
        return None
    return json.dumps(value, default=str, ensure_ascii=False)


def normalize_event(
    event: Dict[str, Any],
    odds: Any = None,
    stats: Any = None,
    league: Optional[Dict[str, Any]] = None,
    season: Optional[Dict[str, Any]] = None,
    odds_comparison: Any = None,
) -> Dict[str, Any]:
    """
    Combina o detalhe de um jogo (`/events/{id}/`), as suas odds
    (`/events/{id}/odds/`), estatísticas (`/events/{id}/stats/`) e,
    opcionalmente, a comparação de bookmakers (`/events/{id}/odds/comparison/`)
    num único registo plano, pronto para exportação (ver `storage.py`).

    Todos os argumentos além de `event` são opcionais — quando ausentes ou
    indisponíveis (jogo sem odds publicadas, sem estatísticas, etc.), os
    campos correspondentes ficam a `None`, respeitando o requisito de
    "sempre que disponível" em vez de falhar a construção do dataset.
    """
    league = league or {}
    season = season or {}

    home_stats, away_stats = _team_containers(stats)

    home_odd, draw_odd, away_odd = _extract_1x2(odds)
    over_1_5, under_1_5 = _extract_over_under(odds, "1.5")
    over_2_5, under_2_5 = _extract_over_under(odds, "2.5")
    over_3_5, under_3_5 = _extract_over_under(odds, "3.5")
    btts_yes, btts_no = _extract_btts(odds)

    bookmakers_available = _extract_bookmakers(odds_comparison)

    match_level_extra = {}
    if isinstance(stats, dict):
        match_level_extra = {k: v for k, v in stats.items() if k not in ("home", "away", "teams")}

    return {
        "event_id": event.get("id"),
        "competition_id": event.get("league_id") if event.get("league_id") is not None else league.get("id"),
        "competition": league.get("name"),
        "season_id": event.get("season_id") if event.get("season_id") is not None else season.get("id"),
        "season": season.get("name"),
        "round_number": event.get("round_number"),
        "round_name": event.get("round_name"),
        "date": event.get("event_date"),
        "status": event.get("status"),
        "home_team_id": event.get("home_team_id"),
        "home_team": event.get("home_team"),
        "away_team_id": event.get("away_team_id"),
        "away_team": event.get("away_team"),
        "home_score": event.get("home_score"),
        "away_score": event.get("away_score"),
        "home_score_ht": event.get("home_score_ht"),
        "away_score_ht": event.get("away_score_ht"),
        "venue_id": event.get("venue_id"),
        "odds_home": home_odd,
        "odds_draw": draw_odd,
        "odds_away": away_odd,
        "odds_over_1_5": over_1_5,
        "odds_under_1_5": under_1_5,
        "odds_over_2_5": over_2_5,
        "odds_under_2_5": under_2_5,
        "odds_over_3_5": over_3_5,
        "odds_under_3_5": under_3_5,
        "odds_btts_yes": btts_yes,
        "odds_btts_no": btts_no,
        "bookmaker": "consensus" if odds else None,
        "bookmakers_available": ",".join(bookmakers_available) if bookmakers_available else None,
        "cards_home_yellow": _extract_team_stat(home_stats, "yellow_cards"),
        "cards_home_red": _extract_team_stat(home_stats, "red_cards"),
        "cards_away_yellow": _extract_team_stat(away_stats, "yellow_cards"),
        "cards_away_red": _extract_team_stat(away_stats, "red_cards"),
        "corners_home": _extract_team_stat(home_stats, "corners"),
        "corners_away": _extract_team_stat(away_stats, "corners"),
        "shots_home": _extract_team_stat(home_stats, "shots_total"),
        "shots_away": _extract_team_stat(away_stats, "shots_total"),
        "shots_on_target_home": _extract_team_stat(home_stats, "shots_on_target"),
        "shots_on_target_away": _extract_team_stat(away_stats, "shots_on_target"),
        "possession_home": _extract_team_stat(home_stats, "possession"),
        "possession_away": _extract_team_stat(away_stats, "possession"),
        "fouls_home": _extract_team_stat(home_stats, "fouls"),
        "fouls_away": _extract_team_stat(away_stats, "fouls"),
        "offsides_home": _extract_team_stat(home_stats, "offsides"),
        "offsides_away": _extract_team_stat(away_stats, "offsides"),
        "extra_stats_home": _json_or_none(_leftover(home_stats, STAT_ALIASES)),
        "extra_stats_away": _json_or_none(_leftover(away_stats, STAT_ALIASES)),
        "extra_match_stats": _json_or_none(match_level_extra),
        "extra_odds": _json_or_none(odds if not isinstance(odds, dict) else {
            k: v for k, v in odds.items()
            if k not in ("1x2", "1X2", "match_winner", "over_under", "btts", "BTTS",
                         "home", "HOME", "draw", "DRAW", "away", "AWAY")
        }),
    }
