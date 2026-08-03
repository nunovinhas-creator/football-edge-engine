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

from typing import Any, Dict, Optional

from src.api.http_retry import get_with_retry
from src.config.settings import BASE_URL, require_api_key
from src.historical_dataset.rate_limiter import RateLimiter

DEFAULT_TIMEOUT_SECONDS = 30


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

        if response.status_code >= 400:
            raise BSDAPIError(response.status_code, url, response.text)

        if not response.content:
            return None

        return response.json()
