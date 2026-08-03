"""
Implementação única e oficial do label `goal_in_next_15m`.

Antes desta consolidação existiam três implementações divergentes da
mesma label (ver `docs/AUDIT_MATEMATICA.md`, secção 10.2):

    (a) `src/backtest/logger.py::update_outcomes`   — janela de tolerância
        `[minuto-18, minuto-12]` sobre `current_minute`, aplicada de forma
        incremental a cada evento ao vivo, só quando o label ainda era
        `NULL`.
    (b) `src/training/create_labels.py`             — janela `(minuto,
        minuto+15]` sobre `current_minute`, recalculada para a tabela
        inteira, sobrescrevendo sempre.
    (c) `src/backtest/labeler.py`                   — janela `(timestamp,
        timestamp+15min]` sobre `timestamp` (relógio), só escrevia `1`
        explicitamente.

Como (b) corria sempre por último no workflow `live_logger.yml`, era o
valor efetivamente persistido em produção — por isso é essa definição que
se torna a implementação oficial, agora centralizada aqui. Todos os
restantes módulos que precisem de calcular ou recalcular
`goal_in_next_15m` devem reutilizar `recompute_goal_in_next_15m` (ou
`recompute_goal_in_next_15m_for_db`) em vez de reimplementar o SQL.

Definição matemática
--------------------
Para um snapshot `s` de um jogo (`match_id`) registado no minuto `m`, com
golos totais `g(s) = home_score(s) + away_score(s)`:

    goal_in_next_15m(s) = 1  se existe um snapshot posterior `s'` do
                              mesmo jogo, em minuto `m'`, tal que
                                  m < m' <= m + 15
                              e   g(s') > g(s)
                          = 0  caso contrário (inclui golos que só
                              aparecem depois de m+15, ou nenhum golo).

A janela é fechada à direita (`m+15` inclusive) e aberta à esquerda
(`m` exclusive) — um snapshot registado exatamente no minuto `m+15`
conta como "dentro da janela"; um golo já refletido no próprio snapshot
`s` (mesmo minuto) não conta, porque exige `m' > m`.

Só são recalculadas linhas com `current_minute IS NOT NULL`.
"""

import sqlite3

LABEL_COLUMN = "goal_in_next_15m"

_RECOMPUTE_SQL = """
UPDATE match_snapshots
SET goal_in_next_15m = (
    SELECT CASE
        WHEN EXISTS (
            SELECT 1
            FROM match_snapshots b
            WHERE b.match_id = match_snapshots.match_id
              AND b.home_score + b.away_score >
                  match_snapshots.home_score + match_snapshots.away_score
              AND b.current_minute > match_snapshots.current_minute
              AND b.current_minute <= match_snapshots.current_minute + 15
        ) THEN 1
        ELSE 0
    END
)
WHERE current_minute IS NOT NULL
"""


def recompute_goal_in_next_15m(conn: sqlite3.Connection) -> int:
    """Recalcula `goal_in_next_15m` para todos os snapshots elegíveis.

    Usa uma ligação sqlite3 já aberta; não faz commit nem fecha a
    ligação — isso fica a cargo de quem chama. Devolve o número de
    linhas afetadas (`cursor.rowcount`).
    """
    cur = conn.cursor()
    cur.execute(_RECOMPUTE_SQL)
    return cur.rowcount


def recompute_goal_in_next_15m_for_db(db_path: str) -> int:
    """Abre `db_path`, recalcula o label, faz commit e fecha a ligação.

    Conveniência para scripts/CLI que só precisam de recalcular a label
    numa base de dados sqlite em disco.
    """
    conn = sqlite3.connect(db_path)
    try:
        rowcount = recompute_goal_in_next_15m(conn)
        conn.commit()
    finally:
        conn.close()
    return rowcount
