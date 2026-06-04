import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.predict import predict_match, get_team_match_count

DATA_PATH = 'data/results.csv'
MODEL_PATH = 'model.pkl'

st.set_page_config(page_title="Predictor de Partidos", page_icon="⚽")
st.title("⚽ Predictor de Partidos de Fútbol")

if not os.path.exists(MODEL_PATH):
    st.error("Modelo no encontrado. Ejecuta primero: `python -m src.train`")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset no encontrado. Coloca `results.csv` en la carpeta `data/`.")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=['date'])


df = load_data()
teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))

default_home = teams.index('Mexico') if 'Mexico' in teams else 0
default_away = teams.index('Argentina') if 'Argentina' in teams else min(1, len(teams) - 1)

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Equipo Local", teams, index=default_home)
with col2:
    away_team = st.selectbox("Equipo Visitante", teams, index=default_away)

if st.button("Predecir", type="primary"):
    if home_team == away_team:
        st.error("Selecciona dos equipos diferentes.")
    else:
        for team in [home_team, away_team]:
            if get_team_match_count(df, team) < 5:
                st.warning(
                    f"{team} tiene menos de 5 partidos en el historial. "
                    "La predicción puede ser imprecisa."
                )

        try:
            prediction = predict_match(home_team, away_team, df, MODEL_PATH)
        except Exception as e:
            st.error(f"Error al generar la predicción: {e}")
            st.stop()

        st.subheader(f"Resultado más probable: **{prediction['result_label']}**")

        probs = prediction['probabilities']
        labels = [f"{home_team} gana", "Empate", f"{away_team} gana"]
        values = [probs.get('H', 0), probs.get('D', 0), probs.get('A', 0)]

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            text=[f"{v * 100:.1f}%" for v in values],
            textposition='auto',
            marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
        ))
        fig.update_layout(
            xaxis=dict(tickformat='.0%', range=[0, 1]),
            yaxis=dict(autorange='reversed'),
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
