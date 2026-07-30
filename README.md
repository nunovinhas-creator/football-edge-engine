⚽ Football Edge Engine v4
Quantitative Football Betting Research Platform

O Football Edge Engine v4 é uma plataforma quantitativa para análise de futebol, modelação probabilística e deteção de oportunidades de valor em mercados pré-jogo e live.

O sistema combina:
- Modelos estatísticos (Poisson)
- Simulação Monte Carlo
- Machine Learning
- Engenharia de features
- Dados live em tempo real
- Análise de odds
- Cálculo de Value Edge
- Gestão de risco via Kelly Criterion

🚀 Arquitetura Geral
FOOTBALL EDGE ENGINE v4 DATA SOURCES
┌─────────────────────────────────────┐
│                                     │
│        BSD Sports Data API          │
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
│         APIMatchProvider            │
│         APIOddsProvider             │
│          StatsProvider              │
│        IncidentsProvider            │
│        BSDFeatureAdapter            │
│                                     │
└─────────────────────────────────────┘
                   │
                   ▼
            LIVE MATCH STATE
┌─────────────────────────────────────┐
│                                     │
│          LiveMatchState             │
│                                     │
│  minute                             │
│  score                              │
│  xG                                 │
│  incidents                          │
│  pressure                           │
│  game state                         │
│                                     │
└─────────────────────────────────────┘
                   │
                   ▼
            ANALYTICS ENGINE
┌─────────────────────────────────────┐
│                                     │
│          LiveGoalEngine             │
│        Monte Carlo Simulator        │
│          Probability Engine         │
│          Value Edge Engine          │
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
│   ├── engine/
│   │   ├── simulation.py
│   │   ├── live_pipeline.py
│   │   ├── poisson.py
│   │   ├── kelly.py
│   │   └── edge.py
│   ├── live/
│   │   ├── engine.py
│   │   └── providers/
│   │       ├── api_match_provider.py
│   │       ├── api_odds_provider.py
│   │       ├── stats_provider.py
│   │       ├── incidents_provider.py
│   │       ├── bsd_feature_adapter.py
│   │       ├── mock_match_provider.py
│   │       └── mock_odds_provider.py
│   ├── models/
│   │   └── live_state.py
│   ├── backtest/
│   ├── tools/
│   └── utils/

🔴 Live Core v4
O módulo live permite analisar jogos em andamento utilizando dados reais.

Providers:
- APIMatchProvider
  Fonte: GET /api/v2/events/live/ | GET /api/v2/events/{id}/?full=true
- APIOddsProvider
  Endpoints: GET /api/v2/events/{id}/odds/ | GET /api/v2/odds/
- StatsProvider
  Endpoint: GET /api/v2/events/{id}/stats/
- IncidentsProvider
  Endpoint: GET /api/v2/events/{id}/incidents/
- BSDFeatureAdapter
  Converte dados BSD em variáveis usadas pelos modelos.

🧠 LiveGoalEngine
Calcula pressão ofensiva, probabilidade do próximo golo, dominância e xG estimado.

🎲 Monte Carlo Simulation Engine
Simulação probabilística de milhares de cenários (Over 1.5, Over 2.5, BTTS).

📊 Value Edge Engine
Compara probabilidade do modelo vs. probabilidade implícita das odds para calcular Expected Value.

💰 Kelly Criterion
Gestão matemática de banca (Stake ideal, exposição, risco).

🤖 Machine Learning
Pipeline para Feature Engineering (Shots, Dangerous attacks, Possession, xG, Momentum, Match state) e Random Forest.

📡 BSD API Integration
Configuração: export BSD_API_KEY="your_key"
Headers: Authorization: Token BSD_API_KEY

🧪 Testes
PYTHONPATH=. python - << 'TEST_EOF'
from src.live.providers.api_match_provider import APIMatchProvider
p = APIMatchProvider()
print(p.get_live_matches())
TEST_EOF

🛠 Desenvolvimento
python -m venv .venv
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

Próximas fases:
v4.1: Websocket live feed, Momentum real-time, Shotmap integration, Possession tracking
v4.2: Automated betting alerts, Telegram Bot, Live opportunity scanner
v5.0: Full ML prediction engine, Reinforcement learning, Portfolio betting management

⚠️ Disclaimer
Este projeto é uma plataforma de investigação quantitativa. Não representa garantia de resultados em apostas.

👨‍💻 Football Edge Engine
Built for quantitative football analysis.
EOF
