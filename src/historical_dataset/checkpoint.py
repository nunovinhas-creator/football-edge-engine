"""Checkpoint/resume para o Historical Dataset Builder.

Uma construção completa do dataset pode envolver milhares de pedidos HTTP
(competições x épocas x jogos x odds x estatísticas) e demorar horas; este
módulo persiste em disco o progresso já feito para que uma execução
interrompida (erro de rede, rate limit, kill do processo) possa ser
retomada sem repetir trabalho já concluído.

Dois níveis de granularidade, escolhidos por serem baratos de persistir
com a frequência certa:

- `completed_seasons`: pares (league_id, season_id) já totalmente
  processados. Persistido como JSON (lista pequena, reescrita completa é
  barata) sempre que uma época termina.
- `processed_events`: IDs de jogos já processados (odds + stats + registo
  normalizado emitido). Persistido como ficheiro append-only (uma linha
  por ID, `flush()` imediato) para não pagar o custo de reescrever tudo a
  cada jogo — importante porque isto é escrito uma vez por jogo, e uma
  época pode ter centenas.
"""

import json
import os
from pathlib import Path
from typing import Optional, Set, Tuple


class Checkpoint:
    """Checkpoint persistido em disco, com resume automático a partir do estado existente."""

    def __init__(self, path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

        self._seasons_path = self.dir / "completed_seasons.json"
        self._events_path = self.dir / "processed_events.log"

        self._completed_seasons: Set[Tuple[int, int]] = self._load_seasons()
        self._processed_events: Set[int] = self._load_events()
        self._events_file = open(self._events_path, "a", encoding="utf-8")

    def _load_seasons(self) -> Set[Tuple[int, int]]:
        if not self._seasons_path.exists():
            return set()
        data = json.loads(self._seasons_path.read_text(encoding="utf-8"))
        return {tuple(pair) for pair in data}

    def _load_events(self) -> Set[int]:
        if not self._events_path.exists():
            return set()
        ids = set()
        with open(self._events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(int(line))
        return ids

    def _save_seasons(self) -> None:
        tmp_path = self._seasons_path.with_suffix(".json.tmp")
        payload = [list(pair) for pair in sorted(self._completed_seasons)]
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, self._seasons_path)

    def is_season_done(self, league_id: int, season_id: Optional[int]) -> bool:
        return (league_id, season_id) in self._completed_seasons

    def mark_season_done(self, league_id: int, season_id: Optional[int]) -> None:
        self._completed_seasons.add((league_id, season_id))
        self._save_seasons()

    def is_event_done(self, event_id: int) -> bool:
        return event_id in self._processed_events

    def mark_event_done(self, event_id: int) -> None:
        if event_id in self._processed_events:
            return
        self._processed_events.add(event_id)
        self._events_file.write(f"{event_id}\n")
        self._events_file.flush()

    def close(self) -> None:
        if not self._events_file.closed:
            self._events_file.close()

    def __enter__(self) -> "Checkpoint":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class NullCheckpoint:
    """No-op: usado quando o utilizador não pede checkpoint/resume (execução única)."""

    def is_season_done(self, league_id, season_id) -> bool:
        return False

    def mark_season_done(self, league_id, season_id) -> None:
        return None

    def is_event_done(self, event_id) -> bool:
        return False

    def mark_event_done(self, event_id) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "NullCheckpoint":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None
