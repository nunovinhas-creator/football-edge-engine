"""Paginação genérica sobre endpoints de lista da BSD API.

`schema.yaml` documenta `/api/v2/events/` e `/api/v2/leagues/` como
devolvendo um array JSON simples (`type: array`), mas paginado por
`limit`/`offset` nos parâmetros do pedido. Código de investigação
anterior (`research/pressure_shots/api.py`) observou que, na prática,
alguns endpoints podem devolver o array diretamente ou envolvido em
`{"results": [...]}` (paginação estilo Django REST Framework) — este
módulo aceita ambas as formas defensivamente.
"""

from typing import Any, Dict, Iterator, Optional

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
) -> Iterator[Dict[str, Any]]:
    """
    Itera todos os itens de um endpoint paginado, avançando `offset` em
    incrementos de `page_size` até uma página devolver menos itens do que
    `page_size` (fim dos dados) ou uma lista vazia.

    `max_pages`, se fornecido, limita o número de páginas pedidas (útil em
    testes ou para limitar o âmbito de uma execução manual).
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

        for item in items:
            yield item

        pages_fetched += 1
        if max_pages is not None and pages_fetched >= max_pages:
            return

        if len(items) < page_size:
            return

        offset += page_size
