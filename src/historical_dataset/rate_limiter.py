"""Rate limiter local para pedidos ao Historical Dataset Builder.

A BSD API não documenta um limite de pedidos por minuto/segundo em
`schema.yaml` (ver `docs/07_historical_dataset_builder.md`, secção
"Limitações"), por isso este limiter é uma salvaguarda proativa do lado
do cliente — não uma implementação do limite real da API — para evitar
sobrecarregar a API ao percorrer milhares de jogos/épocas.

Implementação: janela deslizante simples (sliding window), sem
dependências externas. Thread-safe.
"""

import threading
import time
from collections import deque


class RateLimiter:
    """Permite no máximo `max_calls` chamadas por cada `period_seconds`."""

    def __init__(self, max_calls: int = 5, period_seconds: float = 1.0, sleep_func=time.sleep, time_func=time.monotonic):
        if max_calls <= 0:
            raise ValueError("max_calls tem de ser positivo")
        if period_seconds <= 0:
            raise ValueError("period_seconds tem de ser positivo")

        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._sleep = sleep_func
        self._time = time_func
        self._lock = threading.Lock()
        self._calls = deque()

    def acquire(self) -> float:
        """
        Bloqueia (via `sleep_func`) o tempo necessário para respeitar o
        limite configurado, e regista a chamada atual. Devolve o número de
        segundos efetivamente esperados (0.0 se não foi preciso esperar).
        """
        waited = 0.0
        with self._lock:
            now = self._time()
            self._evict_expired(now)

            if len(self._calls) >= self.max_calls:
                oldest = self._calls[0]
                wait_for = self.period_seconds - (now - oldest)
                if wait_for > 0:
                    waited = wait_for

        if waited > 0:
            self._sleep(waited)

        with self._lock:
            now = self._time()
            self._evict_expired(now)
            self._calls.append(now)

        return waited

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self.period_seconds
        while self._calls and self._calls[0] <= cutoff:
            self._calls.popleft()

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
