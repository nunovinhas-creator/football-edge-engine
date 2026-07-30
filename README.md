# Football Edge Engine

Plataforma de análise estocástica, modelação probabilística (Poisson / Machine Learning) e monitorização de odds de futebol em tempo real.

## 🚀 Funcionalidades Principais

- **Data Collector & Odds API Integration:** Recolha automatizada de dados e odds de casas de apostas.
- **Value Edge & Kelly Criterion Engine:** Cálculo de valor esperado (EV) e gestão de banca com dimensionamento de stake via Kelly Criterion.
- **Poisson & Pressure Shots ML Pipeline:** Modelação de golos esperados e modelo de *Random Forest* para momentos de pressão de jogo.
- **Live Monitor & Telegram Bot:** Alertas em tempo real para oportunidades detetadas em jogos ao vivo.
- **Backtesting Suite:** Simulação histórica de estratégias e relatórios de rentabilidade/Drawdown.

## 📁 Estrutura do Projeto

```text
.
├── src/
│   ├── api/          # Clientes de API e fetchers de odds
│   ├── engine/       # Modelação (Poisson, Kelly, Edge, Filtros)
│   ├── live/         # Processamento de jogos ao vivo
│   ├── backtest/     # Motor de simulação histórica
│   ├── tools/        # Utilitários de diagnóstico e testes
│   └── utils/        # Notificações (Telegram, Logging)
├── research/
│   └── pressure_shots/ # Pipeline de ML (Feature Engineering & Random Forest)
├── docs/             # Arquitetura, escopo e dicionário de APIs
├── app.py            # Entry point principal
├── predict_today.py  # Gerador de previsões diárias
└── live_monitor.py   # Script de monitorização live