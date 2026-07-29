"""FASE 1 — recolha bruta e tabela por-jogo-por-equipa.

Para uma ou mais combinacoes (league_id, season_id), percorre os jogos finished,
cacheia stats/ e player-stats/, e monta uma linha por (jogo, equipa) com as
metricas de pressao do proprio jogo e o total de remates somado dos jogadores.

Jogos sem dangerous_attack/attack/ball_safe em stats.home e stats.away sao
excluidos e contados.

Uso:
    python -m research.pressure_shots.build_raw_table
"""
import json
import os
import sys

import pandas as pd

from research.pressure_shots import api

OUT_DIR = os.path.join("data", "processed", "pressure_shots")
OUT_PICKLE = os.path.join(OUT_DIR, "raw_team_match.pkl")
OUT_CSV = os.path.join(OUT_DIR, "raw_team_match.csv")

PRESSURE_FIELDS = [
    "attack", "attack_pct",
    "dangerous_attack", "dangerous_attack_pct",
    "ball_safe", "ball_safe_pct",
]


def _sum_player_shots(player_stats_payload, team_id):
    total_shots = 0
    shots_on_target = 0
    expected_goals = 0.0
    found_any = False
    for p in player_stats_payload.get("player_stats", []):
        if p.get("team_id") != team_id:
            continue
        found_any = True
        total_shots += p.get("total_shots") or 0
        shots_on_target += p.get("shots_on_target") or 0
        expected_goals += p.get("expected_goals") or 0.0
    return found_any, total_shots, shots_on_target, expected_goals


def collect(league_season_pairs, verbose=True):
    """league_season_pairs: lista de (league_id, season_id, label)."""
    rows = []
    n_games_seen = 0
    n_excluded_no_pressure_stats = 0
    n_excluded_no_player_stats = 0

    for league_id, season_id, label in league_season_pairs:
        for ev in api.iter_events_finished(league_id, season_id):
            n_games_seen += 1
            event_id = ev["id"]
            event_date = ev["event_date"]

            try:
                stats = api.get_event_stats(event_id)
            except api.BzzoiroError as e:
                if verbose:
                    print(f"  [aviso] stats falhou para {event_id}: {e}", file=sys.stderr)
                n_excluded_no_pressure_stats += 1
                continue

            team_stats = stats.get("stats") or {}
            home_stats = team_stats.get("home") or {}
            away_stats = team_stats.get("away") or {}

            if "dangerous_attack" not in home_stats or "dangerous_attack" not in away_stats:
                n_excluded_no_pressure_stats += 1
                continue

            try:
                player_stats = api.get_event_player_stats(event_id)
            except api.BzzoiroError as e:
                if verbose:
                    print(f"  [aviso] player-stats falhou para {event_id}: {e}", file=sys.stderr)
                n_excluded_no_player_stats += 1
                continue

            for is_home, team_id, opp_stats, own_stats in (
                (True, ev["home_team_id"], away_stats, home_stats),
                (False, ev["away_team_id"], home_stats, away_stats),
            ):
                found_any, total_shots, shots_on_target, xg_sum = _sum_player_shots(
                    player_stats, team_id
                )
                if not found_any:
                    n_excluded_no_player_stats += 1
                    continue

                row = {
                    "event_id": event_id,
                    "league_id": league_id,
                    "season_id": season_id,
                    "season_label": label,
                    "event_date": event_date,
                    "team_id": team_id,
                    "opponent_team_id": ev["away_team_id"] if is_home else ev["home_team_id"],
                    "is_home": is_home,
                    "total_shots": total_shots,
                    "shots_on_target": shots_on_target,
                    "expected_goals_players": xg_sum,
                }
                for f in PRESSURE_FIELDS:
                    row[f] = own_stats.get(f)
                    row[f"opp_{f}"] = opp_stats.get(f)
                rows.append(row)

        if verbose:
            print(f"[{label}] acumulado: {len(rows)} linhas ate agora")

    df = pd.DataFrame(rows)
    report = {
        "n_games_seen": n_games_seen,
        "n_rows": len(df),
        "n_excluded_no_pressure_stats": n_excluded_no_pressure_stats,
        "n_excluded_no_player_stats": n_excluded_no_player_stats,
    }
    return df, report


def main():
    # Allsvenskan 2026 (league_id=26, season_id=9): epoca em curso, mas o
    # inicio da epoca (2026-04-04) coincide quase com o arranque da cobertura
    # de dangerous_attack/attack/ball_safe (~2026-04-24) — ver README.md.
    pairs = [(26, 9, "Allsvenskan 2026")]

    df, report = collect(pairs)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_pickle(OUT_PICKLE)
    df.to_csv(OUT_CSV, index=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print()
    print(df["total_shots"].describe())


if __name__ == "__main__":
    main()
