"""Deduplicação de registos do Historical Dataset Builder.

Duas fontes possíveis de duplicados:

1. Dentro de uma única execução: sobreposição de páginas (ex. um jogo
   entra numa liga por `league_id` e novamente por fazer parte doutra
   pesquisa), ou o mesmo jogo aparecer em mais do que uma época por
   inconsistência de dados da API.
2. Entre execuções (checkpoint/resume): o `Checkpoint` já evita reprocessar
   jogos marcados como concluídos, mas esta classe garante que, mesmo sem
   checkpoint (execução única em memória), o dataset final nunca tem duas
   linhas para o mesmo jogo.
"""

from typing import Any, Hashable, Iterable, Iterator, Set


class Deduplicator:
    """Rastreia chaves já vistas nesta execução (em memória)."""

    def __init__(self):
        self._seen: Set[Hashable] = set()
        self._duplicate_count = 0

    def is_duplicate(self, key: Hashable) -> bool:
        found = key in self._seen
        if found:
            self._duplicate_count += 1
        return found

    def add(self, key: Hashable) -> None:
        self._seen.add(key)

    def seen_count(self) -> int:
        return len(self._seen)

    def duplicate_count(self) -> int:
        """Nº de vezes que `is_duplicate` detetou uma chave já vista (para relatórios de qualidade)."""
        return self._duplicate_count


def dedupe_records(records: Iterable[dict], key_fn=lambda r: r.get("event_id")) -> Iterator[dict]:
    """
    Filtra um iterável de registos normalizados, mantendo apenas a
    primeira ocorrência de cada chave (por omissão, `event_id`). Útil para
    limpar um dataset já materializado (ex. resultado de juntar exports de
    execuções diferentes) sem repetir a lógica de deduplicação em cada
    chamador.
    """
    dedup = Deduplicator()
    for record in records:
        key = key_fn(record)
        if dedup.is_duplicate(key):
            continue
        dedup.add(key)
        yield record
