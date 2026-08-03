"""Orquestrador do Historical Dataset Builder.

Percorre: competições -> épocas (`season_id`) -> jogos terminados
(`status=finished`) -> odds disponíveis -> estatísticas disponíveis,
normalizando cada jogo num único registo plano (ver `normalizer.py`).

Não calcula nem altera nenhuma probabilidade, edge, EV, Kelly ou output
de modelo — é um pipeline de extração/normalização de dados brutos da
BSD API, para consumo posterior pelo Backtesting Framework já existente
(ver `backtest_bridge.py`).
"""

import logging
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Union

from src.historical_dataset.checkpoint import NullCheckpoint
from src.historical_dataset.client import BSDAPIError, BSDHistoricalClient
from src.historical_dataset.dedup import Deduplicator
from src.historical_dataset.normalizer import normalize_event
from src.historical_dataset.paginator import DEFAULT_PAGE_SIZE, iter_endpoint

logger = logging.getLogger(__name__)

LeagueLike = Union[int, Dict[str, Any]]


def _default_on_error(stage: str, event_id: Optional[int], exc: Exception) -> None:
    logger.warning("Falha ao obter %s do evento %s: %s", stage, event_id, exc)


def _extract_seasons(payload: Any) -> list:
    """
    Extrai a lista de épocas da resposta de `/api/v2/leagues/{id}/seasons/`.

    `schema.yaml` marca este endpoint como "No response body" (forma não
    documentada). Na prática, a API devolve um envelope próprio deste
    endpoint — `{"league_id": ..., "count": ..., "seasons": [...]}` —,
    distinto do envelope genérico de paginação `{"results": [...]}` usado
    por `paginator.extract_items` para `/leagues/` e `/events/` (que não é
    tocado por esta função). Aceita, por ordem de preferência:

    - array JSON simples (`[...]`);
    - `{"seasons": [...]}` (forma real observada nos logs de produção);
    - `{"results": [...]}` (compatibilidade com o envelope genérico, caso
      este endpoint alguma vez responda nessa forma).

    Qualquer outra forma (payload `None`, dict sem nenhuma destas chaves,
    ou tipo inesperado) devolve lista vazia em vez de falhar.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("seasons", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []
    return []


class HistoricalDatasetBuilder:
    """Constrói o dataset histórico, jogo a jogo, com checkpoint/resume e deduplicação."""

    def __init__(
        self,
        client: Optional[BSDHistoricalClient] = None,
        checkpoint=None,
        page_size: int = DEFAULT_PAGE_SIZE,
        include_odds: bool = True,
        include_stats: bool = True,
        include_odds_comparison: bool = False,
        on_error: Optional[Callable[[str, Optional[int], Exception], None]] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.client = client or BSDHistoricalClient()
        self.checkpoint = checkpoint or NullCheckpoint()
        self.page_size = page_size
        self.include_odds = include_odds
        self.include_stats = include_stats
        self.include_odds_comparison = include_odds_comparison
        self.dedup = Deduplicator()
        self._on_error = on_error or _default_on_error
        self._progress = progress_callback or (lambda event_type, data: None)

    def iter_competitions(
        self,
        country: Optional[str] = None,
        is_women: Optional[bool] = None,
        include_inactive: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Itera `/api/v2/leagues/`, paginado."""
        params: Dict[str, Any] = {}
        if country is not None:
            params["country"] = country
        if is_women is not None:
            params["is_women"] = is_women
        if include_inactive:
            params["include_inactive"] = True
        yield from iter_endpoint(self.client, "leagues/", params=params, page_size=self.page_size)

    def iter_seasons(self, league_id: int) -> list:
        """Devolve todas as épocas de uma liga (`/api/v2/leagues/{id}/seasons/`)."""
        payload = self.client.get(f"leagues/{league_id}/seasons/")
        return _extract_seasons(payload)

    def iter_finished_events(
        self,
        league_id: int,
        season_id: Optional[int],
        page_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Itera jogos com `status=finished` de uma liga/época (`/api/v2/events/`)."""
        params = {"league_id": league_id, "status": "finished"}
        if season_id is not None:
            params["season_id"] = season_id
        yield from iter_endpoint(
            self.client, "events/", params=params, page_size=self.page_size, page_callback=page_callback
        )

    def _safe_get(self, endpoint: str, stage: str, event_id: Optional[int]) -> Any:
        try:
            return self.client.get(endpoint)
        except BSDAPIError as exc:
            if exc.status_code == 404:
                return None
            self._on_error(stage, event_id, exc)
            return None
        except Exception as exc:  # falhas de rede não recuperadas por get_with_retry
            self._on_error(stage, event_id, exc)
            return None

    def fetch_odds(self, event_id: int) -> Any:
        return self._safe_get(f"events/{event_id}/odds/", "odds", event_id)

    def fetch_stats(self, event_id: int) -> Any:
        return self._safe_get(f"events/{event_id}/stats/", "stats", event_id)

    def fetch_odds_comparison(self, event_id: int) -> Any:
        return self._safe_get(f"events/{event_id}/odds/comparison/", "odds_comparison", event_id)

    def build(
        self,
        leagues: Optional[Iterable[LeagueLike]] = None,
        max_events: Optional[int] = None,
        season_ids: Optional[Iterable[int]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Gera registos normalizados (um dict por jogo terminado), streaming.

        `leagues`: subconjunto opcional de competições (dicts já obtidos de
        `iter_competitions`, ou apenas IDs) — por omissão percorre todas as
        competições ativas devolvidas por `iter_competitions()`.

        `max_events`: limite opcional do total de jogos emitidos nesta
        chamada (útil para execuções parciais/testes); jogos e épocas já
        concluídos ficam registados no checkpoint tal como numa execução
        completa.

        `season_ids`: subconjunto opcional de épocas (por `id`) — por
        omissão percorre todas as épocas de cada liga. Épocas fora deste
        conjunto são ignoradas antes de qualquer pedido a `/events/`.

        Se `progress_callback` tiver sido passado ao construtor, é invocado
        com `(event_type, data)` em pontos-chave (`competition_start`,
        `season_start`, `page`, `event`, `season_done`) — usado apenas para
        reportar progresso (ex. CLI/logging), não afeta o resultado.
        """
        emitted = 0
        odds_processed = 0
        season_id_filter = set(season_ids) if season_ids is not None else None
        source_leagues = leagues if leagues is not None else self.iter_competitions()

        for league in source_leagues:
            league_dict = league if isinstance(league, dict) else {"id": league}
            league_id = league_dict["id"]
            self._progress("competition_start", {"league_id": league_id, "league_name": league_dict.get("name")})

            for season in self.iter_seasons(league_id):
                season_id = season.get("id") if isinstance(season, dict) else season

                if season_id_filter is not None and season_id not in season_id_filter:
                    continue

                if self.checkpoint.is_season_done(league_id, season_id):
                    continue

                season_dict = season if isinstance(season, dict) else {"id": season_id}
                self._progress(
                    "season_start",
                    {"league_id": league_id, "season_id": season_id, "season_name": season_dict.get("name")},
                )

                def _on_page(page_number: int, items_count: int, _league_id=league_id, _season_id=season_id) -> None:
                    self._progress(
                        "page",
                        {
                            "league_id": _league_id,
                            "season_id": _season_id,
                            "page_number": page_number,
                            "items_count": items_count,
                        },
                    )

                for event in self.iter_finished_events(league_id, season_id, page_callback=_on_page):
                    event_id = event.get("id")

                    if self.dedup.is_duplicate(event_id) or self.checkpoint.is_event_done(event_id):
                        continue

                    odds = self.fetch_odds(event_id) if self.include_odds else None
                    stats = self.fetch_stats(event_id) if self.include_stats else None
                    comparison = (
                        self.fetch_odds_comparison(event_id) if self.include_odds_comparison else None
                    )
                    if odds is not None:
                        odds_processed += 1

                    record = normalize_event(
                        event,
                        odds=odds,
                        stats=stats,
                        league=league_dict,
                        season=season_dict,
                        odds_comparison=comparison,
                    )

                    self.dedup.add(event_id)
                    self.checkpoint.mark_event_done(event_id)

                    emitted += 1
                    self._progress(
                        "event",
                        {
                            "league_id": league_id,
                            "season_id": season_id,
                            "event_id": event_id,
                            "games_processed": emitted,
                            "odds_processed": odds_processed,
                        },
                    )

                    yield record

                    if max_events is not None and emitted >= max_events:
                        return

                self.checkpoint.mark_season_done(league_id, season_id)
                self._progress("season_done", {"league_id": league_id, "season_id": season_id})

    def build_to_list(self, **kwargs) -> list:
        """Conveniência: materializa `build(...)` numa lista (execuções pequenas/testes)."""
        return list(self.build(**kwargs))
