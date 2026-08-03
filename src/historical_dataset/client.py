"""Cliente HTTP para a BSD API dedicado ao Historical Dataset Builder.

Isolado dos clientes já existentes (`src.api.client.BzzoiroClient`,
`src.live.providers.*`) para não lhes tocar, mas reutiliza a mesma infra
partilhada de sempre:

- `src.config.settings` para a chave de API e `BASE_URL`.
- `src.api.http_retry.get_with_retry` para retries com backoff exponencial
  em falhas transitórias (timeout, connection error, 429/5xx) — o mesmo
  helper usado por todos os outros clientes do projeto.

Envia `Authorization: Token <key>`, o esquema `tokenAuth` definido em
`schema.yaml` (ver `docs/AUDIT_BSD_401.md`), e aplica um rate limiter
local antes de cada pedido (ver `rate_limiter.py`).
"""

import re
from typing import Any, Dict, Optional

from src.api.http_retry import get_with_retry
from src.config.settings import BASE_URL, require_api_key
from src.historical_dataset.rate_limiter import RateLimiter

DEFAULT_TIMEOUT_SECONDS = 30

# --- INÍCIO logging temporário de diagnóstico -------------------------------
# Auditoria exclusiva de GET /api/v2/leagues/{league_id}/seasons/ (ver PR).
# Não altera nenhum valor devolvido, nem a lógica de paginação/filtros —
# apenas imprime o que já foi obtido. Remover após o diagnóstico.
_SEASONS_ENDPOINT_RE = re.compile(r"^leagues/\d+/seasons/$")


def _diag_log_seasons_response(response) -> None:
    print(f"[DIAG seasons] URL completa: {response.url}")
    print(f"[DIAG seasons] Status HTTP: {response.status_code}")
    print(f"[DIAG seasons] Headers da resposta: {dict(response.headers)}")
    request_headers = dict(getattr(response.request, "headers", {}) or {})
    if "Authorization" in request_headers:
        request_headers["Authorization"] = "***REDACTED***"
    print(f"[DIAG seasons] Headers do pedido (Authorization redigido): {request_headers}")
    body_preview = response.text[:2000] if response.content else "(corpo vazio)"
    print(f"[DIAG seasons] JSON bruto (primeiros 2000 chars): {body_preview}")


def _diag_log_seasons_payload(data: Any) -> None:
    print(f"[DIAG seasons] Tipo do payload: {type(data).__name__}")
    if isinstance(data, dict):
        print(f"[DIAG seasons] Chaves do dict: {list(data.keys())}")
        list_key = next((k for k, v in data.items() if isinstance(v, list)), None)
        if list_key is not None:
            print(
                f"[DIAG seasons] Lista de épocas encontrada na chave '{list_key}' "
                f"({len(data[list_key])} itens)"
            )
        else:
            print("[DIAG seasons] Nenhuma chave do dict contém uma lista.")
    elif isinstance(data, list):
        print(f"[DIAG seasons] Lista direta (array JSON simples) com {len(data)} itens")
    else:
        print(f"[DIAG seasons] Payload não é list nem dict: {data!r}")
# --- FIM logging temporário de diagnóstico ----------------------------------


class BSDAPIError(Exception):
    """Erro não-transitório (4xx que não seja 429) devolvido pela BSD API."""

    def __init__(self, status_code: int, url: str, body: str = ""):
        self.status_code = status_code
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status_code} em {url}: {body[:300]}")


class BSDHistoricalClient:
    """Cliente fino e dedicado, com rate limiting local, para o Historical Dataset Builder."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.api_key = (api_key or require_api_key()).rstrip("/")
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.rate_limiter = rate_limiter or RateLimiter(max_calls=5, period_seconds=1.0)
        self.timeout = timeout
        self.request_count = 0

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json",
        }

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        GET a `{base_url}/{endpoint}`, com rate limiting local + retries
        (via `get_with_retry`). Devolve o JSON decodificado (lista ou
        dict, consoante o endpoint — ver `paginator.py`).

        Levanta `BSDAPIError` para respostas 4xx não-transitórias (exceto
        429, que já é tratado como retry transitório por `get_with_retry`).
        """
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/{endpoint}"

        self.rate_limiter.acquire()

        response = get_with_retry(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        self.request_count += 1

        _diag_seasons = bool(_SEASONS_ENDPOINT_RE.match(endpoint))  # TEMPORÁRIO — ver PR
        if _diag_seasons:
            _diag_log_seasons_response(response)

        if response.status_code >= 400:
            raise BSDAPIError(response.status_code, url, response.text)

        if not response.content:
            return None

        data = response.json()

        if _diag_seasons:  # TEMPORÁRIO — ver PR
            _diag_log_seasons_payload(data)

        return data
