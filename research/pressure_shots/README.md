# Dataset de treino: pressão histórica → remates por equipa

Módulo de investigação, isolado de `src/`. Nada aqui é usado em produção.

## Achado crítico da FASE 0 (seleção de liga/época)

A auditoria original assumia que `dangerous_attack`/`attack`/`ball_safe`
estavam disponíveis por liga+época (ex. league_id 83 ou 84, "época mais
recente completa"). Não é o caso: é uma **janela temporal global**, não uma
propriedade da liga. Testei 74 ligas e várias épocas — nenhuma época já
terminada tem estes campos preenchidos; só existem para jogos com
`event_date >= ~2026-04-24`, em qualquer liga. Ou seja, o provider ligou esta
métrica nessa data, e não há histórico retroativo.

Decisão (aprovada pelo utilizador): usar **Allsvenskan 2026**
(`league_id=26`, `season_id=9`). É uma liga de calendário anual
(abril–novembro) cuja época começou em 2026-04-04, quase coincidindo com o
arranque da cobertura — só perdemos as primeiras ~3 semanas da época. A
partir de 2026-04-25 a cobertura é 100% (81/81 jogos terminados até
2026-07-27). 16 equipas, 10–11 jogos cada já disponíveis para histórico.

Consequência para as fases seguintes: esta é uma **época em curso**, não
completa. As features de "época até à véspera" da FASE 2 vão naturalmente
ter menos jogos de histórico no início da janela (a partir de 2026-04-04) do
que teriam numa época completa — isso é esperado, não um bug.

`expected_goals` em player-stats vem sempre `null` nesta liga (mesmo padrão
do `xg.actual` em stats, que a auditoria já tinha identificado como null) —
ignorado, tal como o xg.

## Ficheiros

- `api.py` — cliente HTTP com cache em disco (`data/cache/pressure_shots/`)
  e paginação. `data/` está no `.gitignore`, nada aqui é commitado.
- `build_raw_table.py` — FASE 1: tabela por-jogo-por-equipa
  (`data/processed/pressure_shots/raw_team_match.{csv,pkl}`).
