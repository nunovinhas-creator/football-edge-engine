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
from src.historical_dataset.paginator import DEFAULT_PAGE_SIZE, extract_items, iter_endpoint

logger = logging.getLogger(__name__)

LeagueLike = Union[int, Dict[str, Any]]


def _default_on_error(stage: str, event_id: Optional[int], exc: Exception) -> None:
    logger.warning("Falha ao obter %s do evento %s: %s", stage, event_id, exc)


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
    ):
        self.client = client or BSDHistoricalClient()
        self.checkpoint = checkpoint or NullCheckpoint()
        self.page_size = page_size
        self.include_odds = include_odds
        self.include_stats = include_stats
        self.include_odds_comparison = include_odds_comparison
        self.dedup = Deduplicator()
        self._on_error = on_error or _default_on_error

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
        return extract_items(payload) if isinstance(payload, dict) else (payload or [])

    def iter_finished_events(self, league_id: int, season_id: Optional[int]) -> Iterator[Dict[str, Any]]:
        """Itera jogos com `status=finished` de uma liga/época (`/api/v2/events/`)."""
        params = {"league_id": league_id, "status": "finished"}
        if season_id is not None:
            params["season_id"] = season_id
        yield from iter_endpoint(self.client, "events/", params=params, page_size=self.page_size)

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
        """
        emitted = 0
        source_leagues = leagues if leagues is not None else self.iter_competitions()

        for league in source_leagues:
            league_dict = league if isinstance(league, dict) else {"id": league}
            league_id = league_dict["id"]

            for season in self.iter_seasons(league_id):
                season_id = season.get("id") if isinstance(season, dict) else season

                if self.checkpoint.is_season_done(league_id, season_id):
                    continue

                for event in self.iter_finished_events(league_id, season_id):
                    event_id = event.get("id")

                    if self.dedup.is_duplicate(event_id) or self.checkpoint.is_event_done(event_id):
                        continue

                    odds = self.fetch_odds(event_id) if self.include_odds else None
                    stats = self.fetch_stats(event_id) if self.include_stats else None
                    comparison = (
                        self.fetch_odds_comparison(event_id) if self.include_odds_comparison else None
                    )

                    record = normalize_event(
                        event,
                        odds=odds,
                        stats=stats,
                        league=league_dict,
                        season=season if isinstance(season, dict) else {"id": season_id},
                        odds_comparison=comparison,
                    )

                    self.dedup.add(event_id)
                    self.checkpoint.mark_event_done(event_id)

                    yield record
                    emitted += 1

                    if max_events is not None and emitted >= max_events:
                        return

                self.checkpoint.mark_season_done(league_id, season_id)

    def build_to_list(self, **kwargs) -> list:
        """Conveniência: materializa `build(...)` numa lista (execuções pequenas/testes)."""
        return list(self.build(**kwargs))
