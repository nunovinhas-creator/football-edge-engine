"""Helper centralizado de retries + exponential backoff para chamadas HTTP (requests).

Reexecuta o pedido apenas para falhas transitórias:
- requests.Timeout
- requests.ConnectionError
- respostas HTTP 429, 500, 502, 503, 504

Qualquer outro resultado (sucesso, ou erro não-transitório como 4xx de cliente)
é devolvido/propagado tal como aconteceria com uma chamada direta a
`requests.get`/`requests.post`, para que o código chamador continue a poder
usar `response.raise_for_status()` normalmente.
"""

import logging
import random
import time

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_BACKOFF_MAX_SECONDS = 30.0


def _sleep_with_backoff(attempt, backoff_base, backoff_max):
    delay = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
    delay += random.uniform(0, delay * 0.25)  # jitter
    logger.warning("Sleeping %.1f seconds before retry", delay)
    time.sleep(delay)


def request_with_retry(
    method,
    url,
    *,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
    backoff_base=DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_max=DEFAULT_BACKOFF_MAX_SECONDS,
    **kwargs,
):
    """Como requests.request, mas com retries automáticos para erros transitórios."""
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(method, url, **kwargs)
        except requests.Timeout:
            if attempt == max_attempts:
                raise
            logger.warning("Retry %d/%d after timeout", attempt + 1, max_attempts)
            _sleep_with_backoff(attempt, backoff_base, backoff_max)
            continue
        except requests.ConnectionError:
            if attempt == max_attempts:
                raise
            logger.warning("Retry %d/%d after connection error", attempt + 1, max_attempts)
            _sleep_with_backoff(attempt, backoff_base, backoff_max)
            continue

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts:
            logger.warning(
                "Retry %d/%d after HTTP %d", attempt + 1, max_attempts, response.status_code
            )
            _sleep_with_backoff(attempt, backoff_base, backoff_max)
            continue

        return response


def get_with_retry(url, **kwargs):
    return request_with_retry("GET", url, **kwargs)


def post_with_retry(url, **kwargs):
    return request_with_retry("POST", url, **kwargs)
