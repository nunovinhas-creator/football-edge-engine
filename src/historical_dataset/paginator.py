"""Paginação genérica sobre endpoints de lista da BSD API.

`schema.yaml` documenta `/api/v2/events/` e `/api/v2/leagues/` como
devolvendo um array JSON simples (`type: array`), mas paginado por
`limit`/`offset` nos parâmetros do pedido. Código de investigação
anterior (`research/pressure_shots/api.py`) observou que, na prática,
alguns endpoints podem devolver o array diretamente ou envolvido em
`{"results": [...]}` (paginação estilo Django REST Framework) — este
módulo aceita ambas as formas defensivamente.
"""

from typing import Any, Callable, Dict, Iterator, Optional

DEFAULT_PAGE_SIZE = 100


def extract_items(payload: Any) -> list:
    """Extrai a lista de itens de uma resposta paginada ou array simples."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        results = payload.get("results")
        if results is not None:
            return results
        return []
    return []


def iter_endpoint(
    client,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: Optional[int] = None,
    page_callback: Optional[Callable[[int, int], None]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Itera todos os itens de um endpoint paginado, avançando `offset` em
    incrementos de `page_size` até uma página devolver menos itens do que
    `page_size` (fim dos dados) ou uma lista vazia.

    `max_pages`, se fornecido, limita o número de páginas pedidas (útil em
    testes ou para limitar o âmbito de uma execução manual).

    `page_callback`, se fornecido, é invocado após cada página obtida com
    `(page_number, items_count)` (1-indexado) — usado apenas para reportar
    progresso (ex. CLI); não afeta a paginação em si.
    """
    base_params = dict(params or {})
    offset = 0
    pages_fetched = 0

    while True:
        page_params = dict(base_params)
        page_params["limit"] = page_size
        page_params["offset"] = offset

        payload = client.get(endpoint, params=page_params)
        items = extract_items(payload)

        if not items:
            return

        pages_fetched += 1
        if page_callback is not None:
            page_callback(pages_fetched, len(items))

        for item in items:
            yield item

        if max_pages is not None and pages_fetched >= max_pages:
            return

        if len(items) < page_size:
            return

        offset += page_size
