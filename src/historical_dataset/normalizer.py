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


def _unwrap_resource(payload: Any, key: str) -> Any:
    """
    Desembrulha `{key: {...}}` quando presente, devolvendo o dict interno.

    A BSD API real embrulha respostas de sub-recursos de evento sob uma
    chave com o nome do próprio recurso — padrão já reconhecido para
    `/leagues/{id}/seasons/` -> `{"seasons": [...]}` (ver
    `builder._extract_seasons`) e para `/events/{id}/player-stats/` ->
    `{"player_stats": [...]}` (ver `research/pressure_shots/build_raw_table.py`).
    O mesmo padrão aplica-se a `/events/{id}/odds/` -> `{"odds": {...}}`,
    confirmado por código de produção ativo (`main.py live`):
    `analysis["odds"]["odds"]["over_15_goals"]` em `src/cli/live.py` e
    `scripts/live_scanner.py`, onde `analysis["odds"]` é o JSON devolvido
    sem alterações por `APIOddsProvider.get_live_odds()` — a mesma chamada
    a `GET /events/{id}/odds/` que este builder faz.

    Se `payload[key]` não for um dict (ou a chave não existir), devolve
    `payload` tal como veio — mantém compatibilidade com a forma já
    desembrulhada (datasets/testes anteriores a esta correção).
    """
    if isinstance(payload, dict) and isinstance(payload.get(key), dict):
        return payload[key]
    return payload


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
    # A BSD API real devolve `/events/{id}/odds/` como um dict plano com
    # `home_win`/`away_win` (não `home`/`away`, que era a forma assumida
    # antes de haver uma resposta real para confirmar o formato).
    home = _lookup(container, "home", "HOME", "home_win", "HOME_WIN", "1")
    draw = _lookup(container, "draw", "DRAW", "x", "X")
    away = _lookup(container, "away", "AWAY", "away_win", "AWAY_WIN", "2")
    return home, draw, away


def _extract_over_under(odds: Any, threshold: str) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(odds, dict):
        return None, None

    suffix = threshold.replace(".", "_")
    # Forma real da BSD API: dict plano com `over_25_goals`/`under_25_goals`
    # (sufixo "25", sem underscore entre os dígitos, seguido de "_goals").
    goals_suffix = threshold.replace(".", "")
    container = odds.get("over_under") or odds.get(f"over_under_{suffix}")

    sub = None
    if isinstance(container, dict):
        sub = container.get(threshold) or container.get(f"over_under_{suffix}") or container.get(f"OU_{suffix}")

    if sub is None:
        over = _lookup(
            odds,
            f"over_{suffix}", f"OVER_{suffix}",
            f"over_{goals_suffix}_goals", f"OVER_{goals_suffix}_GOALS",
        )
        under = _lookup(
            odds,
            f"under_{suffix}", f"UNDER_{suffix}",
            f"under_{goals_suffix}_goals", f"UNDER_{goals_suffix}_GOALS",
        )
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
    # Forma real da BSD API: `/events/{id}/stats/` devolve o payload
    # embrulhado sob a chave "stats" (ver `_unwrap_resource`), confirmado
    # por `research/pressure_shots/build_raw_table.py`:
    # `stats.get("stats").get("home")`, validado contra respostas reais da
    # API. A forma já desembrulhada (`{"home": ..., "away": ...}`
    # diretamente no topo) continua suportada para compatibilidade.
    container = _unwrap_resource(stats, "stats")
    if not isinstance(container, dict):
        return None, None
    if "home" in container or "away" in container:
        return container.get("home"), container.get("away")
    teams = container.get("teams")
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

    # Forma real da BSD API: `/events/{id}/odds/` devolve o payload
    # embrulhado sob a chave "odds" (ver `_unwrap_resource`), confirmado
    # por código de produção ativo (`main.py live`):
    # `analysis["odds"]["odds"]["over_15_goals"]` em `src/cli/live.py` e
    # `scripts/live_scanner.py`. A forma já desembrulhada continua
    # suportada para compatibilidade com datasets/testes anteriores.
    odds_payload = _unwrap_resource(odds, "odds")

    home_odd, draw_odd, away_odd = _extract_1x2(odds_payload)
    over_1_5, under_1_5 = _extract_over_under(odds_payload, "1.5")
    over_2_5, under_2_5 = _extract_over_under(odds_payload, "2.5")
    over_3_5, under_3_5 = _extract_over_under(odds_payload, "3.5")
    btts_yes, btts_no = _extract_btts(odds_payload)

    bookmakers_available = _extract_bookmakers(odds_comparison)

    match_level_extra = {}
    if isinstance(stats, dict):
        exclude_keys = {"home", "away", "teams"}
        if isinstance(stats.get("stats"), dict):
            # Payload embrulhado: "stats" já foi consumido por
            # `_team_containers` acima — excluir para não duplicar
            # home/away dentro de `extra_match_stats`.
            exclude_keys.add("stats")
        match_level_extra = {k: v for k, v in stats.items() if k not in exclude_keys}

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
        "bookmaker": "consensus" if odds_payload else None,
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
        # `extra_odds` guarda o payload bruto de `/events/{id}/odds/` tal
        # como veio da API, sem filtrar as chaves já mapeadas para as
        # colunas normalizadas acima — serve de auditoria/fallback para
        # mercados ainda não suportados, não deve divergir do bruto.
        "extra_odds": _json_or_none(odds),
    }
