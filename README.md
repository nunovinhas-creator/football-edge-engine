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

        Estrutura do Projeto
        src/

├── api/
│
├── engine/
│   ├── simulation.py
│   ├── live_pipeline.py
│   ├── poisson.py
│   ├── kelly.py
│   └── edge.py
│
├── live/
│   │
│   ├── engine.py
│   │
│   └── providers/
│       │
│       ├── api_match_provider.py
│       ├── api_odds_provider.py
│       ├── stats_provider.py
│       ├── incidents_provider.py
│       ├── bsd_feature_adapter.py
│       ├── mock_match_provider.py
│       └── mock_odds_provider.py
│
├── models/
│   └── live_state.py
│
├── backtest/
│
├── tools/
│
└── utils/

🔴 Live Core v4
O módulo live permite analisar jogos em andamento utilizando dados reais.

Providers
APIMatchProvider

Responsável por:

Jogos live
Estado atual do jogo
Resultado
Minuto
Equipas

Fonte:GET /api/v2/events/live/
GET /api/v2/events/{id}/?full=true

APIOddsProvider

Responsável por:

Odds live
Mercados disponíveis
Comparação preço/probabilidade

Mercados:

Over
Under
BTTS
Resultado
Next Goal

Endpoints:GET /api/v2/events/{id}/odds/
GET /api/v2/odds/

StatsProvider

Recolhe:

xG
Estatísticas do jogo
Dados adicionais

Endpoint:GET /api/v2/events/{id}/stats/

IncidentsProvider

Transforma acontecimentos reais em features:

Exemplos:

Golos
Períodos
Cartões
Eventos importantes

Endpoint:GET /api/v2/events/{id}/incidents/

BSDFeatureAdapter

Converte dados BSD em variáveis usadas pelos modelos.

Exemplo:{
"goals_last_15": 1,
"last_goal_minute": 76,
"red_cards": 0,
"game_state": "second_half"
}


🧠 LiveGoalEngine
Motor responsável por calcular:

Pressão ofensiva
Probabilidade do próximo golo
Dominância
xG estimado

Output: {
"pressure":54.9,
"dominance_index":81,
"estimated_xg_10m":1.75,
"next_goal_probability":95
}


🎲 Monte Carlo Simulation Engine

Simulação probabilística de milhares de cenários.

Calcula:

Over 1.5
Over 2.5
BTTS
Golos esperados

Exemplo:{
"over_15":0.85,
"over_25":0.62,
"btts":0.55
}

📊 Value Edge Engine

Compara:
Probabilidade Modelo
          |
          |
          ▼
Probabilidade Implícita Odds
          |
          |
          ▼
Expected Value

💰 Kelly Criterion

Gestão matemática de banca.

Calcula:

Stake ideal
Exposição
Risco
🤖 Machine Learning

Pipeline preparado para:

Feature Engineering
Random Forest
Modelos de pressão ofensiva
Previsão de golos

Features:

Shots
Dangerous attacks
Possession
xG
Momentum
Match state
📡 BSD API Integration

Configuração:export BSD_API_KEY="your_key"
Headers:Authorization: Token BSD_API_KEY

🧪 Testes

Executar:PYTHONPATH=. python - << EOF

from src.live.providers.api_match_provider import APIMatchProvider

p = APIMatchProvider()

print(
    p.get_live_matches()
)

EOF

🛠 Desenvolvimento

Criar ambiente:python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

🏗 Roadmap
v4.0

✅ Live Providers
✅ BSD API Integration
✅ Live Match State
✅ Odds Integration
✅ Incidents Processing
✅ Monte Carlo Live Simulation

Próximas fases
v4.1
Websocket live feed
Momentum real-time
Shotmap integration
Possession tracking
v4.2
Automated betting alerts
Telegram Bot
Live opportunity scanner
v5
Full ML prediction engine
Reinforcement learning
Portfolio betting management
⚠️ Disclaimer

Este projeto é uma plataforma de investigação quantitativa.

Não representa garantia de resultados em apostas.

O objetivo é:

Modelação estatística
Investigação
Deteção de probabilidades incorretas
Gestão racional de risco
👨‍💻 Football Edge Engine

Built for quantitative football analysis.
EOF
Depois verifica:

```bash
wc -l README.md
e:
git diff -- README.md | head -100
git add README.md
git commit -m "update complete project documentation and architecture"
git push origin feature/live-core-v4
