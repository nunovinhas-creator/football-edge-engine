import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px
from src.live.engine import LiveGoalEngine

# Configuração da página Web
st.set_page_config(
    page_title="Football Edge Engine | Live Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Edge Engine — In-Play Goal Monitor")
st.markdown("Monitorização em tempo real de pressão, atrito tático e probabilidade de golo iminente.")

# Tabs do Dashboard
tab1, tab2 = st.tabs(["🔥 Jogos em Direto (Live)", "📊 Backtest & Histórico"])

with tab1:
    st.header("Jogos em Monitorização (Ao Vivo)")
    
    # Simulação/Leitura dos jogos atuais
    # (Podes conectar diretamente ao BSDLiveFetcher)
    engine = LiveGoalEngine()
    
    # Exemplo de visualização de jogo ao vivo
    st.subheader("🏟️ Liga Portugal / European Leagues")
    
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        st.metric(label="Jogo", value="FC Porto vs Sporting CP")
        st.caption("⏱️ Minuto: 62' | Placard: 1 - 1")
        
    with col2:
        # Calcular probabilidade
        sample_data = {
            'home_xg_last5': 1.8, 'away_conceded_xg_last5': 1.4,
            'home_style': 'high_press', 'away_style': 'low_block_vulnerable',
            'dangerous_attacks_10m': 14, 'shots_on_target_10m': 3, 'corners_10m': 4
        }
        prob = engine.predict_next_goal_probability(sample_data)
        
        st.metric(label="Probabilidade Golo (Próx. 15m)", value=f"{prob*100:.1f}%")
        st.progress(prob)

    with col3:
        if prob >= 0.65:
            st.error("🚨 **ALERTA DE PRESSÃO EXTREMA!** Elevada probabilidade de golo nos próximos minutos.")
        else:
            st.info("ℹ️ Ritmo de jogo dentro da média normal.")

with tab2:
    st.header("Análise de Dados Registados (Data Logger)")
    
    db_path = "data/live_history.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM match_snapshots ORDER BY id DESC", conn)
        conn.close()
        
        st.metric(label="Total de Snapshots Registados", value=len(df))
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("⚠️ Base de dados histórica ainda não encontrada. O GitHub Actions irá criá-la assim que houver jogos em direto.")

