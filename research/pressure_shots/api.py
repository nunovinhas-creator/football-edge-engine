"""Cliente HTTP fino para a API Bzzoiro, com cache em disco e paginação.

Isolado de src/ propositadamente: este é código de investigação, não de produção.
"""
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://sports.bzzoiro.com/api/v2"
API_KEY = os.getenv("BZZ_API_KEY")

CACHE_ROOT = os.path.join("data", "cache", "pressure_shots")

# Ritmo comedido entre pedidos que vão à rede (cache hits não esperam).
REQUEST_DELAY_SECONDS = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


class BzzoiroError(Exception):
    pass


def _headers():
    return {"Authorization": f"Token {API_KEY}"}


def _cache_path(subdir: str, key: str) -> str:
    d = os.path.join(CACHE_ROOT, subdir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.json")


def _request(path: str, params: dict | None = None):
    url = f"{BASE_URL}/{path}"
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        except requests.RequestException as e:
            last_err = e
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", RETRY_BACKOFF_SECONDS * attempt))
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            last_err = BzzoiroError(f"HTTP {resp.status_code} em {url}")
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code >= 400:
            raise BzzoiroError(f"HTTP {resp.status_code} em {url}: {resp.text[:300]}")

        time.sleep(REQUEST_DELAY_SECONDS)
        return resp.json()

    raise BzzoiroError(f"Falhou após {MAX_RETRIES} tentativas em {url}: {last_err}")


def get_cached(subdir: str, key: str, path: str, params: dict | None = None):
    """Devolve o JSON em cache para (subdir, key); caso contrário pede à API e guarda."""
    cache_file = _cache_path(subdir, key)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    data = _request(path, params=params)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def iter_events_finished(league_id: int, season_id: int, page_size: int = 200):
    """Itera todos os eventos finished de uma liga/época, paginando."""
    offset = 0
    while True:
        params = {
            "league_id": league_id,
            "season_id": season_id,
            "status": "finished",
            "limit": page_size,
            "offset": offset,
        }
        data = _request("events/", params=params)

        # A API pode devolver lista simples ou objeto paginado {results: [...]}.
        if isinstance(data, dict):
            results = data.get("results", [])
        else:
            results = data

        if not results:
            break

        for ev in results:
            yield ev

        if len(results) < page_size:
            break
        offset += page_size


def get_event_stats(event_id: int):
    return get_cached("stats", str(event_id), f"events/{event_id}/stats/")


def get_event_player_stats(event_id: int):
    return get_cached("player_stats", str(event_id), f"events/{event_id}/player-stats/")


def get_league_seasons(league_id: int):
    return _request(f"leagues/{league_id}/seasons/")
