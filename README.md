# ⚽ Football Edge Engine v4

## Quantitative Football Betting Research Platform

O **Football Edge Engine v4** é uma plataforma quantitativa para análise de futebol, modelação probabilística e deteção de oportunidades de valor em mercados pré-jogo e live.

O sistema combina:

- Modelos estatísticos (Poisson)
- Simulação Monte Carlo
- Machine Learning
- Engenharia de features
- Dados live em tempo real
- Análise de odds
- Cálculo de Value Edge
- Gestão de risco via Kelly Criterion


---

# 🚀 Arquitetura Geral

```text
                    FOOTBALL EDGE ENGINE v4


                         DATA SOURCES

        ┌─────────────────────────────────────┐
        │                                     │
        │  BSD Sports Data API                │
        │                                     │
        │  - Live Events                      │
        │  - Live Odds                        │
        │  - Match Details                    │
        │  - Incidents                        │
        │  - Statistics                       │
        │                                     │
        └─────────────────────────────────────┘

                         │

                         ▼


              LIVE DATA PROCESSING LAYER


        ┌─────────────────────────────────────┐
        │                                     │
        │ APIMatchProvider                    │
        │ APIOddsProvider                     │
        │ StatsProvider                       │
        │ IncidentsProvider                   │
        │ BSDFeatureAdapter                   │
        │                                     │
        └─────────────────────────────────────┘


                         │

                         ▼


                    LIVE MATCH STATE


        ┌─────────────────────────────────────┐
        │                                     │
        │ LiveMatchState                      │
        │                                     │
        │ minute                              │
        │ score                               │
        │ xG                                  │
        │ incidents                           │
        │ pressure                            │
        │ game state                          │
        │                                     │
        └─────────────────────────────────────┘


                         │

                         ▼


                 ANALYTICS ENGINE


        ┌─────────────────────────────────────┐
        │                                     │
        │ LiveGoalEngine                     │
        │                                     │
        │ Monte Carlo Simulator              │
        │                                     │
        │ Probability Engine                 │
        │                                     │
        │ Value Edge Engine                  │
        │                                     │
        └─────────────────────────────────────┘


                         │

                         ▼


                  BETTING DECISION LAYER


        Markets:

        - Next Goal
        - Over 1.5
        - Over 2.5
        - BTTS
        - Match Winner
        - Handicap
