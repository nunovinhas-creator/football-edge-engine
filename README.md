# ⚽ Football Edge Engine

O **Football Edge Engine** é um sistema automatizado em Python concebido para identificar apostas com valor esperado positivo (EV+) no mercado de **remates** (*Over Remates*), utilizando dados em tempo real da **BSD API** e alertas automatizados via **Telegram**.

---

## 🏗️ Arquitetura do Repositório

```
football-edge-engine/
│
├── .github/workflows/      # Automação via GitHub Actions (Daily Run)
├── data/                   # Armazenamento de dados temporários / cache
├── docs/                   # Documentação adicional e esquemas
├── notebooks/              # Análises exploratórias (Jupyter)
├── research/               # Datasets históricos e prototipagem de modelos
│   └── pressure_shots/     # Features e Random Forest
├── scripts/                # Utility scripts de suporte
├── src/
│   ├── api/                # Cliente BSD API (Bzzoiro)
│   ├── engine/             # Pipeline de decisão (Kelly, EV+, Risk)
│   └── utils/              # Telegram Notifier, Logging
│
├── predict_today.py        # Script principal de execução diária
├── requirements.txt        # Dependências do projeto
└── schema.yaml             # Definição de dados/schemas
```

---

## 🚀 Funcionalidades Principais

* **Integração BSD API:** Extração em tempo real de eventos e odds.
* **Filtro Temporal de Janela Móvel:** Validação estrita em UTC para analisar exclusivamente jogos agendados para os **próximos 3 dias**.
* **Modelo Predictivo de Remates:** Estimativa de probabilidade para a linha *Over 12.5 Remates* via Random Forest.
* **Motor EV+ & Sizing:** Cálculo de Value Bet e recomendações de stake com gestão de banca.
* **Alertas no Telegram:** Notificação automática via Bot com as top oportunidades do dia.

---

## ⚙️ Instalação e Execução

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar script de previsões diárias
python predict_today.py
```
