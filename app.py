import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from datetime import datetime

from src.api.live_fetcher import BSDLiveFetcher
from src.live.engine import LiveGoalEngine

# Configuração da página Web
st.set_page_config(
    page_title="Football Edge Engine | Live Dashboard",
    page_icon="⚽",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00d47e;
        margin-bottom: 15px;
    }
    .ev-badge {
        background-color: #00d47e;
        color: #000000;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Football Edge Engine — In-Play Goal Monitor")
st.markdown("Monitorização dinâmica de pressão em tempo real e deteção de **+EV (Expected Value)**.")

# Botão de Atualização no Topo
col_btn, col_time = st.columns([1, 4])
with col_btn:
    if st.button("🔄 Atualizar Dados"):
        st.rerun()
with col_time:
    st.caption(f"Última verificação: {datetime.now().strftime('%H:%M:%S')}")

tab1, tab2 = st.tabs(["🔥 Jogos em Direto (Live)", "📊 Backtest & Histórico Logger"])

with tab1:
    st.header("Radar de Jogos em Tempo Real")
    
    engine = LiveGoalEngine()
    
    try:
        fetcher = BSDLiveFetcher()
        events = fetcher.get_live_events()
    except Exception as e:
        events = []
        st.warning(f"ℹ️ Não foi possível carregar eventos da API em direto (ou sem chave ativa): {e}")

    if not events:
        st.info("ℹ️ Nenhum jogo a decorrer neste momento na BSD API. A mostrar simulação de teste para validação de layout:")
        # Fallback de demonstração caso não haja jogos a decorrer à hora do teste
        events = [{
            'id': 999, 'home_team': 'FC Porto', 'away_team': 'Sporting CP', 
            'current_minute': 64, 'home_score': 1, 'away_score': 1
        }]

    for event in events:
        try:
            match_data = fetcher.parse_live_metrics_for_engine(event)
        except Exception:
            # Fallback seguro para mock de teste
            match_data = {
                'match_id': event.get('id', 0),
                'home_team': event.get('home_team', 'Casa'),
                'away_team': event.get('away_team', 'Fora'),
                'current_minute': event.get('current_minute', 60),
                'home_score': event.get('home_score', 0),
                'away_score': event.get('away_score', 0),
                'home_xg_last5': 1.8, 'away_conceded_xg_last5': 1.4,
                'home_style': 'high_press', 'away_style': 'low_block_vulnerable',
                'dangerous_attacks_10m': 14, 'shots_on_target_10m': 3, 'corners_10m': 4,
                'live_odd_over': 2.10
            }

        p_goal_15m = engine.predict_next_goal_probability(match_data)
        fair_odd = 1.0 / p_goal_15m if p_goal_15m > 0 else 99.0
        bookie_odd = match_data.get('live_odd_over', 1.85)
        
        # Cálculo de EV%
        ev_percent = ((p_goal_15m * bookie_odd) - 1.0) * 100

        # Layout do Jogo
        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
            
            with c1:
                st.subheader(f"🏟️ {match_data['home_team']} {match_data['home_score']} - {match_data['away_score']} {match_data['away_team']}")
                st.caption(f"⏱️ Minuto: **{match_data['current_minute']}'** | Cantos (10m): {match_data['corners_10m']} | Remates Alvo (10m): {match_data['shots_on_target_10m']}")

            with c2:
                st.metric(label="P(Golo 15m)", value=f"{p_goal_15m*100:.1f}%")
                st.progress(p_goal_15m)

            with c3:
                st.metric(label="Odd Justa vs Casa", value=f"{fair_odd:.2f}", delta=f"Casa: {bookie_odd:.2f}")

            with c4:
                if p_goal_15m >= 0.65 and ev_percent > 0:
                    st.error(f"🚀 **+EV DETETADO (+{ev_percent:.1f}%)**")
                    st.markdown("<span class='ev-badge'>ENTRADA SUGERIDA: OVER 0.5 GOLOS (15M)</span>", unsafe_allow_html=True)
                elif p_goal_15m >= 0.60:
                    st.warning("⚠️ **PRESSÃO A AUMENTAR** — Aguardar subida da Odd.")
                else:
                    st.info("ℹ️ Ritmo Normal / Sem valor de aposta.")
        st.divider()

with tab2:
    st.header("Base de Dados do Logger (`live_history.db`)")
    
    db_path = "data/live_history.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM match_snapshots ORDER BY id DESC", conn)
        conn.close()
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Total Snapshots Gravados", len(df))
        col_s2.metric("Jogos Únicos Monitorizados", df['match_id'].nunique() if 'match_id' in df.columns else 0)
        
        if 'goal_in_next_15m' in df.columns:
            resolved = df[df['goal_in_next_15m'].notnull()]
            col_s3.metric("Snapshots Verificados (Ground Truth)", len(resolved))

        st.dataframe(df, use_container_width=True)
    else:
        st.warning("⚠️ A base de dados ainda não foi criada localmente. O GitHub Actions irá atualizá-la automaticamente.")
