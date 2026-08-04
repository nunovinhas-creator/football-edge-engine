"""
Football Edge Engine — Dashboard Pro.

Camada de INTERFACE apenas. Todo o conteúdo mostrado aqui vem de
`src.report.dashboard_data` (que por sua vez só invoca os módulos
oficiais do motor, inalterados: Goal Engine, Monte Carlo, Dixon-Coles,
Machine Learning, Edge, EV, Kelly, Decision Engine, Backtesting/
Evaluation Framework). Este ficheiro não calcula nenhuma probabilidade,
edge, EV, Kelly ou lambda — apenas formata, organiza e apresenta.
"""

import sys
from datetime import datetime
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.api.live_fetcher import BSDLiveFetcher
from src.live.engine import LiveGoalEngine
from src.model.ml_predictor import LiveMLPredictor
from src.backtest.historical.metrics import equity_curve

from src.report.dashboard_data import (
    DEMO_EVENT,
    DEMO_MATCH_DATA,
    build_match_snapshot,
    extract_competition,
    extract_status_label,
    get_bsd_status,
    get_ml_status,
    get_telegram_status,
    load_live_history,
    load_value_alerts,
    run_demo_backtest,
)
from src.report.explainability import generate_explanation

DASHBOARD_VERSION = "Pro v1.0"

# ---------------------------------------------------------------------------
# Configuração da página + estilos
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Football Edge Engine | Dashboard Pro",
    page_icon="⚽",
    layout="wide",
)

_BADGE_COLORS = {
    "ok": "#0f5132",
    "warn": "#664d03",
    "off": "#58151c",
}
_BADGE_BORDERS = {
    "ok": "#1DB954",
    "warn": "#e6b800",
    "off": "#e5484d",
}

st.markdown(
    """
    <style>
    .fee-pill {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-right: 8px; margin-bottom: 6px;
        border: 1px solid rgba(255,255,255,0.15); color: #ffffff !important;
    }
    .fee-pill * { color: #ffffff !important; }
    .fee-card {
        background-color: rgba(127,127,127,0.08); border-radius: 12px;
        padding: 16px 18px; border: 1px solid rgba(127,127,127,0.18);
        margin-bottom: 12px; height: 100%;
    }
    .fee-decision-box {
        border-radius: 16px; padding: 28px 24px; text-align: center;
        border: 2px solid rgba(255,255,255,0.15); margin-bottom: 10px;
    }
    .fee-decision-label { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
    .fee-decision-reason { font-size: 0.95rem; opacity: 0.85; margin-top: 6px; }
    .fee-section-title {
        font-size: 1.05rem; font-weight: 700; margin: 22px 0 8px 0;
        border-left: 4px solid #1DB954; padding-left: 10px;
    }
    .fee-explain li { margin-bottom: 6px; }
    .fee-why-col { padding: 10px 4px; }
    .fee-why-title { font-weight: 700; margin-bottom: 6px; }
    .fee-why-col ul { margin: 0; padding-left: 18px; }
    .fee-why-col li { margin-bottom: 6px; }
    .fee-why-summary {
        margin-top: 6px; padding-top: 10px; border-top: 1px solid rgba(127,127,127,0.25);
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pill(label: str, color_key: str) -> str:
    bg = _BADGE_COLORS.get(color_key, "#333")
    border = _BADGE_BORDERS.get(color_key, "#888")
    return f'<span class="fee-pill" style="background:{bg};border-color:{border};">{label}</span>'


def section_title(text: str) -> None:
    st.markdown(f'<div class="fee-section-title">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Recursos "pesados" — carregados uma única vez por sessão (item 15:
# reutilizar resultados já calculados, não recarregar/retreinar modelos)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_goal_engine() -> LiveGoalEngine:
    return LiveGoalEngine()


@st.cache_resource(show_spinner=False)
def _load_ml_predictor() -> LiveMLPredictor:
    return LiveMLPredictor()


@st.cache_data(ttl=60, show_spinner=False)
def _load_backtest_report():
    return run_demo_backtest()


@st.cache_data(ttl=15, show_spinner=False)
def _load_history_df():
    return load_live_history()


@st.cache_data(ttl=15, show_spinner=False)
def _load_alerts_df():
    return load_value_alerts()


goal_engine = _load_goal_engine()
ml_predictor = _load_ml_predictor()

# ---------------------------------------------------------------------------
# 1. Cabeçalho
# ---------------------------------------------------------------------------

try:
    fetcher = BSDLiveFetcher()
    live_events = fetcher.get_live_events()
    fetch_error = None
except Exception as exc:
    fetcher = None
    live_events = []
    fetch_error = str(exc)

using_demo = not live_events

bsd_label, bsd_color = get_bsd_status()
telegram_label, telegram_color = get_telegram_status()
ml_label, ml_color = get_ml_status(ml_predictor)
system_label, system_color = ("🟢 Operacional", "ok")

header_left, header_right = st.columns([3, 1])
with header_left:
    st.title("⚽ Football Edge Engine")
    st.caption(f"Dashboard Pro — {DASHBOARD_VERSION} · Decisão do motor em destaque, sem ruído.")
with header_right:
    if st.button("🔄 Atualizar Dados", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown(
    pill(f"Sistema: {system_label}", system_color)
    + pill(f"API BSD: {bsd_label}", bsd_color)
    + pill(f"Telegram: {telegram_label}", telegram_color)
    + pill(f"Machine Learning: {ml_label}", ml_color)
    + pill(f"Jogos ao vivo: {len(live_events)}", "ok" if live_events else "warn")
    + pill(f"Última atualização: {datetime.now().strftime('%H:%M:%S')}", "ok"),
    unsafe_allow_html=True,
)

if using_demo:
    st.info(
        "ℹ️ Sem jogos em direto disponíveis na BSD API neste momento "
        + (f"({fetch_error})." if fetch_error else ".")
        + " A mostrar um jogo de demonstração para validação do layout."
    )

st.divider()

# ---------------------------------------------------------------------------
# Navegação principal
# ---------------------------------------------------------------------------

tab_live, tab_backtest, tab_history = st.tabs(
    ["🔥 Monitor ao Vivo", "📊 Backtest", "🗂️ Histórico & Logs"]
)


# ---------------------------------------------------------------------------
# Painéis auxiliares (reutilizados para cada jogo)
# ---------------------------------------------------------------------------

def render_decision_panel(snap: dict) -> None:
    section_title("🎯 Decisão do Motor")
    d = snap["decision"]
    col_decision, col_confidence, col_score = st.columns([2, 1, 1])

    with col_decision:
        st.markdown(
            f"""
            <div class="fee-decision-box" style="background:{_BADGE_COLORS[d['color']]};
                 border-color:{_BADGE_BORDERS[d['color']]};">
                <div class="fee-decision-label">{d['label']}</div>
                <div class="fee-decision-reason">{d['reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_confidence:
        st.markdown(
            f"""
            <div class="fee-card" style="text-align:center;">
                <div style="opacity:0.75;font-size:0.85rem;">Confiança</div>
                <div style="font-size:1.6rem;font-weight:800;">{d['confidence_label']}</div>
                <div style="opacity:0.7;font-size:0.85rem;">{d['confidence_score']:.1f}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(d["confidence_score"] / 100.0, 0.0), 1.0))

    with col_score:
        es = snap["engine_score"]
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=es["score"],
                number={"suffix": "", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": _BADGE_BORDERS[es["color"]]},
                    "steps": [
                        {"range": [0, 35], "color": "rgba(229,72,77,0.25)"},
                        {"range": [35, 55], "color": "rgba(230,184,0,0.25)"},
                        {"range": [55, 75], "color": "rgba(29,185,84,0.15)"},
                        {"range": [75, 100], "color": "rgba(29,185,84,0.30)"},
                    ],
                },
                title={"text": f"Engine Score — {es['label']}", "font": {"size": 13}},
            )
        )
        fig.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_models_panel(snap: dict) -> None:
    section_title("🧩 Painel dos Modelos")
    m = snap["models"]
    cols = st.columns(4)

    cards = [
        ("⚙️ Goal Engine", m["goal_engine"]["probability"], m["goal_engine"]["market"], m["goal_engine"]["status"]),
        ("🤖 Machine Learning", m["machine_learning"]["probability"], m["machine_learning"]["market"], f"Conf.: {m['machine_learning']['confidence']:.0f}/100"),
        ("🎲 Monte Carlo", m["monte_carlo"]["over_15"], m["monte_carlo"]["market"], f"Over 2.5: {m['monte_carlo']['over_25']}% · BTTS: {m['monte_carlo']['btts']}%"),
        ("📐 Dixon-Coles", max(m["dixon_coles"]["home"], m["dixon_coles"]["draw"], m["dixon_coles"]["away"]), m["dixon_coles"]["market"], f"1:{m['dixon_coles']['home']}% X:{m['dixon_coles']['draw']}% 2:{m['dixon_coles']['away']}%"),
    ]

    for col, (name, prob, market, status) in zip(cols, cards):
        color = "ok" if prob >= 60 else ("warn" if prob >= 35 else "off")
        with col:
            st.markdown(
                f"""
                <div class="fee-card">
                    <div style="font-weight:700;">{name}</div>
                    <div style="opacity:0.7;font-size:0.78rem;margin-bottom:6px;">{market}</div>
                    <div style="font-size:1.8rem;font-weight:800;color:{_BADGE_BORDERS[color]};">{prob:.1f}%</div>
                    <div style="opacity:0.75;font-size:0.78rem;">{status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(prob / 100.0, 0.0), 1.0))

    c = snap["consensus"]
    st.markdown(
        pill(f"Consenso entre modelos (Goal Engine × ML): {c['label']} — diferença {c['gap']:.1f} p.p.", c["color"]),
        unsafe_allow_html=True,
    )


def render_value_panel(snap: dict, bankroll: float) -> None:
    section_title("💰 Painel de Valor")
    v = snap["value"]
    stake_amount = round(bankroll * (v["kelly_pct"] / 100.0), 2)

    cols = st.columns(6)
    cols[0].metric("Odd Mercado", f"{v['bookie_odd']:.2f}")
    cols[1].metric("Odd Justa", f"{v['fair_odd']:.2f}" if v["fair_odd"] else "—")
    cols[2].metric("Edge", f"{v['edge_pct']:+.1f}%")
    cols[3].metric("EV", f"{v['ev_pct']:+.1f}%")
    cols[4].metric("Kelly", f"{v['kelly_pct']:.2f}%")
    cols[5].metric("Stake Recomendada", f"{stake_amount:.2f} €")

    st.caption(
        f"Mercado avaliado: **{v['market']}**. Mercado alternativo (Over 1.5, Monte Carlo): "
        f"edge {v['over15_edge_pct']:+.1f}% · EV {v['over15_ev_pct']:+.1f}% · Kelly {v['over15_kelly_pct']:.2f}% "
        f"→ {v['over15_action']}."
    )


def render_live_panel(snap: dict) -> None:
    section_title("📡 Painel Live")
    live = snap["live"]

    cols = st.columns(4)
    with cols[0]:
        st.markdown("**Pressão**")
        st.progress(min(max(live["pressure"] / 100.0, 0.0), 1.0), text=f"{live['pressure']:.1f}/100")
        st.markdown("**Dominância**")
        st.progress(min(max(live["dominance_index"] / 100.0, 0.0), 1.0), text=f"{live['dominance_index']:.1f}/100")
    with cols[1]:
        st.markdown("**Posse de Bola (casa)**")
        st.progress(min(max(live["possession"] / 100.0, 0.0), 1.0), text=f"{live['possession']:.0f}%")
        st.metric("xG (10 min)", f"{live['estimated_xg_10m']:.2f}")
    with cols[2]:
        st.metric("Ataques Perigosos (10m)", live["dangerous_attacks_10m"])
        st.metric("Remates (10m)", live["shots_10m"])
        st.metric("Remates Enquadrados (10m)", live["shots_on_target_10m"])
    with cols[3]:
        st.metric("Cantos (10m)", live["corners_10m"])
        st.metric("Cartões Vermelhos", live["red_cards"])
        momentum_color = {"SURGING": "ok", "RISING": "ok", "STABLE": "warn", "FALLING": "off", "COLLAPSING": "off"}.get(live["momentum"], "warn")
        st.markdown(pill(f"Momentum: {live['momentum']}", momentum_color), unsafe_allow_html=True)

    st.caption(f"🪟 Janela de golo prevista: **{live['goal_window']}** — {live['goal_window_intensity']}")


def render_strength_panel(snap: dict) -> None:
    section_title("🏋️ Strength")
    s = snap["strength"]
    cols = st.columns(4)
    cols[0].metric("Força Casa (λ dinâmico)", f"{s['home_lambda']:.2f} golos esp.")
    cols[1].metric("Força Visitante (λ dinâmico)", f"{s['away_lambda']:.2f} golos esp.")
    cols[2].metric("Tier", "N/D (live)")
    cols[3].metric("H2H Disponível", "Não" if not s["h2h_available"] else "Sim")
    st.caption(
        "Em modo ao vivo, a força das equipas é aproximada pelo λ dinâmico (pressão + xG ao vivo) "
        "já usado pelo Monte Carlo/Dixon-Coles — o Tier e a Effective Sample Size do Lambda Estimator "
        "só existem quando há dataset histórico H2H carregado (fluxo pré-jogo)."
    )


def render_explanation_panel(snap: dict) -> None:
    section_title("🧠 Explicação da Decisão")
    items = "".join(f"<li>{b}</li>" for b in snap["explanation"])
    st.markdown(f'<div class="fee-card fee-explain"><ul>{items}</ul></div>', unsafe_allow_html=True)


def render_why_this_decision_panel(snap: dict) -> None:
    """
    Melhoria #13 — Explainability Engine (`src.report.explainability`).
    Interpretação 100% determinística (sem IA/LLM) sobre os valores já
    presentes no snapshot: não recalcula nenhuma probabilidade, edge, EV,
    Kelly ou lambda, nem altera nenhuma decisão do motor.
    """
    section_title("🧠 Porque esta decisão?")
    explanation = generate_explanation(snap)

    def _list_html(items):
        if not items:
            return "<li><em>Nenhum.</em></li>"
        return "".join(f"<li>{i}</li>" for i in items)

    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f'<div class="fee-card fee-why-col">'
            f'<div class="fee-why-title">✔ Pontos positivos</div>'
            f'<ul>{_list_html(explanation.positives)}</ul></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="fee-card fee-why-col">'
            f'<div class="fee-why-title">⚠ Riscos</div>'
            f'<ul>{_list_html(explanation.negatives)}</ul></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="fee-card fee-why-col">'
            f'<div class="fee-why-title">🚨 Avisos</div>'
            f'<ul>{_list_html(explanation.warnings)}</ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="fee-card fee-why-summary">Resumo: {explanation.summary}</div>',
        unsafe_allow_html=True,
    )


def render_logs_panel(snap: dict) -> None:
    with st.expander("🧾 Logs — snapshot completo (todos os valores usados nesta análise)"):
        st.json(snap)


def render_match(snap: dict, bankroll: float) -> None:
    card = snap["card"]
    section_title("🏟️ Resumo do Jogo")
    st.subheader(f"{card['home_team']} {card['home_score']} - {card['away_score']} {card['away_team']}")
    meta_cols = st.columns(4)
    with meta_cols[0]:
        st.caption("Competição")
        st.markdown(f"**{card['competition']}**")
    meta_cols[1].metric("Minuto", card["elapsed"])
    meta_cols[2].metric("Tempo Decorrido", f"{card['minute']}'")
    meta_cols[3].metric("Estado", card["status"])

    render_decision_panel(snap)
    render_models_panel(snap)
    render_value_panel(snap, bankroll)
    render_live_panel(snap)
    render_strength_panel(snap)
    render_explanation_panel(snap)
    render_why_this_decision_panel(snap)
    render_logs_panel(snap)


# ---------------------------------------------------------------------------
# 🔥 Tab: Monitor ao Vivo
# ---------------------------------------------------------------------------

with tab_live:
    bankroll = st.number_input(
        "Banca de referência para cálculo da stake (€)", min_value=10.0, value=1000.0, step=50.0
    )

    if using_demo:
        events_to_render = [DEMO_EVENT]
    else:
        events_to_render = live_events

    for idx, event in enumerate(events_to_render):
        if using_demo:
            match_data = DEMO_MATCH_DATA
        else:
            try:
                match_data = fetcher.parse_live_metrics_for_engine(event)
            except Exception:
                match_data = {
                    "match_id": event.get("id", 0),
                    "home_team": event.get("home_team", "Casa"),
                    "away_team": event.get("away_team", "Fora"),
                    "current_minute": event.get("current_minute", 0),
                    "home_score": event.get("home_score", 0),
                    "away_score": event.get("away_score", 0),
                }

        competition = extract_competition(event)
        status_label = extract_status_label(event, match_data.get("current_minute", 0))

        snap = build_match_snapshot(
            match_data,
            competition=competition,
            status_label=status_label,
            ml_predictor=ml_predictor,
            goal_engine=goal_engine,
        )

        card = snap["card"]
        header = (
            f"⚽ {card['home_team']} {card['home_score']}-{card['away_score']} {card['away_team']} "
            f"({card['elapsed']}) — {snap['decision']['label']} · Engine Score {snap['engine_score']['score']:.0f}/100"
        )
        with st.expander(header, expanded=(idx == 0)):
            render_match(snap, bankroll)


# ---------------------------------------------------------------------------
# 📊 Tab: Backtest
# ---------------------------------------------------------------------------

with tab_backtest:
    st.subheader("📊 Backtesting Framework")
    st.caption(
        "Demonstração com o dataset histórico real incluído no repositório "
        "(`examples/backtest/sample_real_games.csv`) — mesmo `BacktestEngine` usado por `run_backtest.py --demo`, "
        "sem alterar nenhuma métrica nem recalcular nenhum modelo."
    )

    report = _load_backtest_report()
    g = report.global_metrics
    s = report.statistical_metrics

    row1 = st.columns(6)
    row1[0].metric("ROI", f"{g['roi_pct']:.1f}%")
    row1[1].metric("Yield", f"{g['yield_pct']:.1f}%")
    row1[2].metric("Hit Rate", f"{g['hit_rate_pct']:.1f}%")
    row1[3].metric("Nº Apostas", g["n_bets"])
    row1[4].metric("Lucro Líquido", f"{g['net_profit']:.2f} u")
    row1[5].metric("Drawdown Máx.", f"{g['max_drawdown_pct']:.1f}%")

    row2 = st.columns(6)
    row2[0].metric("Brier Score", f"{s['brier_score']:.4f}")
    row2[1].metric("Log Loss", f"{s['log_loss']:.4f}")
    row2[2].metric("ECE", f"{s['calibration_error']:.4f}")
    row2[3].metric("Profit Factor", f"{g['profit_factor']:.2f}" if g["profit_factor"] != float("inf") else "∞")
    row2[4].metric("Odd Média", f"{g['avg_odd']:.2f}")
    clv_cov = g.get("clv_coverage_pct", 0.0)
    row2[5].metric("Cobertura CLV", f"{clv_cov:.1f}%")

    col_equity, col_calib = st.columns(2)

    with col_equity:
        curve = equity_curve(report.placed_bets)
        fig = go.Figure()
        if not curve.empty:
            fig.add_trace(go.Scatter(x=list(range(1, len(curve) + 1)), y=curve.values, mode="lines+markers", line=dict(color="#1DB954")))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Evolução da Banca (lucro acumulado)", height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_calib:
        curve_df = report.calibration_curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Calibração perfeita"))
        if not curve_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=curve_df["predicted_mean"], y=curve_df["actual_frequency"],
                    mode="lines+markers", name="Modelo", line=dict(color="#e5484d"),
                )
            )
        fig.update_layout(
            title="Curva de Calibração", height=340, margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(range=[0, 1], title="Prob. Prevista"), yaxis=dict(range=[0, 1], title="Frequência Real"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Apostas avaliadas (detalhe)"):
        st.dataframe(report.all_bets, use_container_width=True)


# ---------------------------------------------------------------------------
# 🗂️ Tab: Histórico & Logs
# ---------------------------------------------------------------------------

with tab_history:
    st.subheader("🗂️ Histórico")

    history_df = _load_history_df()
    alerts_df = _load_alerts_df()

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Snapshots Gravados", len(history_df))
    col_s2.metric("Jogos Únicos Monitorizados", history_df["match_id"].nunique() if "match_id" in history_df.columns else 0)
    col_s3.metric("Alertas Telegram Enviados", len(alerts_df))

    st.markdown("**📈 Últimos Sinais — Pressão ao longo do tempo (por jogo, últimos snapshots)**")
    if not history_df.empty and {"match_id", "pressure"}.issubset(history_df.columns):
        plot_df = history_df.copy()
        plot_df["jogo"] = plot_df["home_team"].astype(str) + " vs " + plot_df["away_team"].astype(str)
        plot_df = plot_df.sort_values("id")
        fig = px.line(
            plot_df.tail(500), x="id", y="pressure", color="jogo",
            labels={"id": "Snapshot", "pressure": "Pressão"},
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ainda não há snapshots suficientes para desenhar o gráfico de pressão.")

    st.markdown("**⚽ Últimos Jogos Monitorizados**")
    if not history_df.empty:
        last_per_match = (
            history_df.sort_values("id")
            .groupby("match_id", as_index=False)
            .tail(1)
            .sort_values("id", ascending=False)
        )
        display_cols = [
            c for c in [
                "timestamp", "match_id", "home_team", "away_team", "current_minute",
                "home_score", "away_score", "pressure", "dominance_index", "estimated_xg_10m",
            ]
            if c in last_per_match.columns
        ]
        st.dataframe(last_per_match[display_cols].head(30), use_container_width=True)
    else:
        st.info("Sem jogos monitorizados ainda.")

    st.markdown("**🚨 Últimos Alertas Telegram (+EV)**")
    if not alerts_df.empty:
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.info("Ainda não foi enviado nenhum alerta Telegram de valor (+EV).")

    with st.expander("🧾 Logs — todos os snapshots gravados (data/live_history.db)"):
        st.dataframe(history_df, use_container_width=True)
