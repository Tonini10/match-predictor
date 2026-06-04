import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.predict import predict_match, get_team_match_count
from src.stats import (
    get_team_overall_stats,
    get_radar_stats,
    get_head_to_head,
    get_recent_matches,
)

DATA_PATH = 'data/results.csv'
MODEL_PATH = 'model.pkl'

st.set_page_config(
    page_title="Predictor de Partidos ⚽",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Predictor de Partidos de Fútbol")
st.caption("Modelo Random Forest · Datos históricos internacionales 1872–2023")

if not os.path.exists(DATA_PATH):
    st.error("Dataset no encontrado. Coloca `results.csv` en la carpeta `data/`.")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=['date'])


df = load_data()

if not os.path.exists(MODEL_PATH):
    with st.spinner("Entrenando modelo por primera vez... (~30 segundos)"):
        from src.train import train
        train()

teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
default_home = teams.index('Mexico') if 'Mexico' in teams else 0
default_away = teams.index('Argentina') if 'Argentina' in teams else min(1, len(teams) - 1)

col1, col2, col3 = st.columns([5, 1, 5])
with col1:
    home_team = st.selectbox("🏠 Equipo Local", teams, index=default_home)
with col2:
    st.markdown("<div style='text-align:center;padding-top:28px;font-size:1.4rem;color:#666'>VS</div>",
                unsafe_allow_html=True)
with col3:
    away_team = st.selectbox("✈️ Equipo Visitante", teams, index=default_away)

predict_btn = st.button("🎯 Predecir Partido", type="primary", use_container_width=True)

# ── Shared chart layout ────────────────────────────────────────────────────
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1d23",
    plot_bgcolor="#1a1d23",
    font=dict(color="#fafafa", size=13),
    margin=dict(l=20, r=20, t=40, b=20),
)
_COLORS = {'H': '#00d4aa', 'D': '#f39c12', 'A': '#e74c3c'}
_RESULT_COLOR = {'W': '#00d4aa', 'D': '#f39c12', 'L': '#e74c3c'}


def _badge(result):
    c = _RESULT_COLOR.get(result, '#888')
    label = {'W': 'V', 'D': 'E', 'L': 'D'}.get(result, result)
    return (f'<span style="background:{c};color:#000;padding:3px 9px;'
            f'border-radius:5px;font-weight:bold;margin:2px;font-size:13px">{label}</span>')


if predict_btn:
    if home_team == away_team:
        st.error("Selecciona dos equipos diferentes.")
        st.stop()

    for team in [home_team, away_team]:
        if get_team_match_count(df, team) < 5:
            st.warning(f"{team} tiene menos de 5 partidos en el historial. La predicción puede ser imprecisa.")

    try:
        prediction = predict_match(home_team, away_team, df, MODEL_PATH)
    except Exception as e:
        st.error(f"Error al generar la predicción: {e}")
        st.stop()

    probs = prediction['probabilities']
    result_key = prediction['result']
    result_label = prediction['result_label']
    confidence = max(probs.values()) * 100
    result_color = _COLORS[result_key]

    home_stats = get_team_overall_stats(df, home_team)
    away_stats = get_team_overall_stats(df, away_team)
    radar = get_radar_stats(df, home_team, away_team)
    h2h = get_head_to_head(df, home_team, away_team)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Predicción", "📊 Estadísticas", "⚔️ Head-to-Head", "📋 Historial"]
    )

    # ── TAB 1: PREDICCIÓN ─────────────────────────────────────────────────
    with tab1:
        st.markdown(
            f'<div style="text-align:center;padding:28px;background:#1a1d23;'
            f'border-radius:14px;margin-bottom:20px;border:2px solid {result_color}">'
            f'<div style="font-size:2.6rem;font-weight:900;color:{result_color};'
            f'letter-spacing:1px">{result_label.upper()}</div>'
            f'<div style="color:#888;margin-top:10px;font-size:1rem">'
            f'Confianza del modelo: <b style="color:#fafafa">{confidence:.1f}%</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏠 {home_team} gana", f"{probs.get('H', 0)*100:.1f}%")
        m2.metric("🤝 Empate", f"{probs.get('D', 0)*100:.1f}%")
        m3.metric(f"✈️ {away_team} gana", f"{probs.get('A', 0)*100:.1f}%")

        fig_prob = go.Figure(go.Bar(
            x=[probs.get('H', 0), probs.get('D', 0), probs.get('A', 0)],
            y=[f"{home_team} gana", "Empate", f"{away_team} gana"],
            orientation='h',
            text=[f"{v*100:.1f}%" for v in [probs.get('H', 0), probs.get('D', 0), probs.get('A', 0)]],
            textposition='auto',
            marker_color=['#00d4aa', '#f39c12', '#e74c3c'],
        ))
        fig_prob.update_layout(
            **_LAYOUT,
            xaxis=dict(tickformat='.0%', range=[0, 1], showgrid=False),
            yaxis=dict(autorange='reversed'),
            height=210,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"### 🏠 {home_team}")
            st.metric("⚽ Goles anotados/PJ", home_stats['goals_scored_per_game'])
            st.metric("🛡️ Goles recibidos/PJ", home_stats['goals_conceded_per_game'])
            st.metric("🏆 % Victorias", f"{home_stats['win_rate']*100:.0f}%")
            badges = " ".join(_badge(r) for r in home_stats['form'][-5:])
            st.markdown(f"**Forma reciente:** {badges or 'Sin datos'}", unsafe_allow_html=True)

        with c2:
            st.markdown(f"### ✈️ {away_team}")
            st.metric("⚽ Goles anotados/PJ", away_stats['goals_scored_per_game'])
            st.metric("🛡️ Goles recibidos/PJ", away_stats['goals_conceded_per_game'])
            st.metric("🏆 % Victorias", f"{away_stats['win_rate']*100:.0f}%")
            badges = " ".join(_badge(r) for r in away_stats['form'][-5:])
            st.markdown(f"**Forma reciente:** {badges or 'Sin datos'}", unsafe_allow_html=True)

    # ── TAB 2: ESTADÍSTICAS ───────────────────────────────────────────────
    with tab2:
        st.markdown("### 🕸️ Radar de rendimiento")
        dims = radar['dimensions']
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar['home'] + [radar['home'][0]],
            theta=dims + [dims[0]],
            fill='toself', name=home_team,
            line=dict(color='#00d4aa', width=2),
            fillcolor='rgba(0,212,170,0.15)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar['away'] + [radar['away'][0]],
            theta=dims + [dims[0]],
            fill='toself', name=away_team,
            line=dict(color='#e74c3c', width=2),
            fillcolor='rgba(231,76,60,0.15)',
        ))
        fig_radar.update_layout(
            **_LAYOUT,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color='#555', showticklabels=False),
                angularaxis=dict(color='#fafafa'),
                bgcolor='#1a1d23',
            ),
            height=420,
            legend=dict(orientation='h', y=-0.08, x=0.3),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### 📋 Comparación de métricas")
        metrics = [
            ("⚽ Goles anotados/PJ",    home_stats['goals_scored_per_game'],   away_stats['goals_scored_per_game'],   True),
            ("🛡️ Goles recibidos/PJ",   home_stats['goals_conceded_per_game'], away_stats['goals_conceded_per_game'], False),
            ("🏆 % Victorias",          f"{home_stats['win_rate']*100:.1f}%",   f"{away_stats['win_rate']*100:.1f}%",  True),
            ("🤝 % Empates",            f"{home_stats['draw_rate']*100:.1f}%",  f"{away_stats['draw_rate']*100:.1f}%", None),
            ("❌ % Derrotas",           f"{home_stats['loss_rate']*100:.1f}%",  f"{away_stats['loss_rate']*100:.1f}%", False),
            ("⚡ Partidos con gol",     f"{home_stats['scoring_rate']*100:.1f}%", f"{away_stats['scoring_rate']*100:.1f}%", True),
            ("🔒 Porterías a cero",     f"{home_stats['clean_sheet_rate']*100:.1f}%", f"{away_stats['clean_sheet_rate']*100:.1f}%", True),
            ("📊 Total partidos",       home_stats['total_matches'],            away_stats['total_matches'],           None),
        ]

        for name, hv, av, higher_better in metrics:
            c1, c2, c3 = st.columns([2, 3, 2])
            hs = as_ = ""
            if higher_better is not None:
                try:
                    hn = float(str(hv).replace('%', ''))
                    an = float(str(av).replace('%', ''))
                    if higher_better:
                        hs = "color:#00d4aa;font-weight:bold" if hn >= an else "color:#888"
                        as_ = "color:#00d4aa;font-weight:bold" if an > hn else "color:#888"
                    else:
                        hs = "color:#00d4aa;font-weight:bold" if hn <= an else "color:#888"
                        as_ = "color:#00d4aa;font-weight:bold" if an < hn else "color:#888"
                except (ValueError, TypeError):
                    pass
            c1.markdown(f'<div style="text-align:right;padding:6px 0;{hs}">{hv}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div style="text-align:center;padding:6px 0;color:#aaa">{name}</div>', unsafe_allow_html=True)
            c3.markdown(f'<div style="text-align:left;padding:6px 0;{as_}">{av}</div>', unsafe_allow_html=True)

        st.markdown("### 📅 Forma últimos 10 partidos")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(f"**{home_team}**")
            badges = " ".join(_badge(r) for r in home_stats['form'])
            st.markdown(badges or "Sin datos", unsafe_allow_html=True)
        with fc2:
            st.markdown(f"**{away_team}**")
            badges = " ".join(_badge(r) for r in away_stats['form'])
            st.markdown(badges or "Sin datos", unsafe_allow_html=True)

    # ── TAB 3: HEAD-TO-HEAD ───────────────────────────────────────────────
    with tab3:
        total_h2h = h2h['home_wins'] + h2h['draws'] + h2h['away_wins']
        if total_h2h == 0:
            st.info(f"No hay enfrentamientos directos registrados entre {home_team} y {away_team}.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"🏆 Victorias {home_team}", h2h['home_wins'])
            c2.metric("🤝 Empates", h2h['draws'])
            c3.metric(f"🏆 Victorias {away_team}", h2h['away_wins'])

            dc1, dc2 = st.columns([1, 1])
            with dc1:
                fig_donut = go.Figure(go.Pie(
                    labels=[f"{home_team}", "Empate", f"{away_team}"],
                    values=[h2h['home_wins'], h2h['draws'], h2h['away_wins']],
                    hole=0.6,
                    marker_colors=['#00d4aa', '#f39c12', '#e74c3c'],
                    textinfo='label+percent',
                    textfont=dict(size=13),
                ))
                fig_donut.update_layout(**_LAYOUT, height=320, showlegend=False,
                                        title=dict(text="Distribución H2H", x=0.5))
                st.plotly_chart(fig_donut, use_container_width=True)

            with dc2:
                st.markdown("#### Goles históricos")
                gc1, gc2, gc3 = st.columns(3)
                gc1.metric(f"⚽ {home_team}", h2h['home_goals_total'])
                gc2.metric("📊 Promedio goles/PJ", h2h['avg_goals_per_game'])
                gc3.metric(f"⚽ {away_team}", h2h['away_goals_total'])

                if h2h['last_matches']:
                    dates = [m['date'] for m in reversed(h2h['last_matches'])]
                    hg = [m['home_score'] if m['home_team'] == home_team else m['away_score']
                          for m in reversed(h2h['last_matches'])]
                    ag = [m['away_score'] if m['home_team'] == home_team else m['home_score']
                          for m in reversed(h2h['last_matches'])]
                    fig_goals = go.Figure()
                    fig_goals.add_trace(go.Bar(name=home_team, x=dates, y=hg, marker_color='#00d4aa'))
                    fig_goals.add_trace(go.Bar(name=away_team, x=dates, y=ag, marker_color='#e74c3c'))
                    fig_goals.update_layout(**_LAYOUT, barmode='group', height=260,
                                            title=dict(text="Goles por partido", x=0.5),
                                            legend=dict(orientation='h', y=1.15))
                    st.plotly_chart(fig_goals, use_container_width=True)

            if h2h['last_matches']:
                st.markdown("### 📋 Últimos enfrentamientos")
                for m in h2h['last_matches']:
                    ht, at = m['home_team'], m['away_team']
                    hs_score, as_score = m['home_score'], m['away_score']
                    if ht == home_team:
                        team_res = 'W' if hs_score > as_score else ('D' if hs_score == as_score else 'L')
                    else:
                        team_res = 'W' if as_score > hs_score else ('D' if as_score == hs_score else 'L')
                    color = _RESULT_COLOR[team_res]
                    st.markdown(
                        f'<div style="background:#1a1d23;border-left:4px solid {color};'
                        f'padding:10px 16px;margin:5px 0;border-radius:6px">'
                        f'<span style="color:#666;font-size:12px">{m["date"]}</span>&nbsp;&nbsp;'
                        f'<b style="font-size:1rem">{ht} {hs_score} — {as_score} {at}</b>&nbsp;&nbsp;'
                        f'<span style="color:#555;font-size:12px">{m["tournament"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── TAB 4: HISTORIAL ─────────────────────────────────────────────────
    with tab4:
        hc1, hc2 = st.columns([2, 1])
        with hc1:
            selected_team = st.selectbox("Selecciona equipo", [home_team, away_team], key="hist_team")
        with hc2:
            n_opt = st.select_slider("Partidos", options=[10, 20, 50, 100, 'Todos'], value=20, key="hist_n")

        n_val = None if n_opt == 'Todos' else int(n_opt)
        recent = get_recent_matches(df, selected_team, n=n_val)

        if len(recent) == 0:
            st.info("Sin partidos registrados.")
        else:
            wins = int((recent['result'] == 'W').sum())
            draws_h = int((recent['result'] == 'D').sum())
            losses_h = int((recent['result'] == 'L').sum())
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("🏆 Victorias", wins)
            sc2.metric("🤝 Empates", draws_h)
            sc3.metric("❌ Derrotas", losses_h)
            sc4.metric("⚽ Goles a favor", int(recent['goals_for'].sum()))
            sc5.metric("🛡️ Goles en contra", int(recent['goals_against'].sum()))

            recent_asc = recent.sort_values('date')
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=recent_asc['date'], y=recent_asc['goals_for'],
                name='Goles anotados', line=dict(color='#00d4aa', width=2),
                mode='lines+markers', marker=dict(size=6),
            ))
            fig_timeline.add_trace(go.Scatter(
                x=recent_asc['date'], y=recent_asc['goals_against'],
                name='Goles recibidos', line=dict(color='#e74c3c', width=2),
                mode='lines+markers', marker=dict(size=6),
            ))
            fig_timeline.update_layout(
                **_LAYOUT, height=280,
                xaxis_title="Fecha", yaxis_title="Goles",
                legend=dict(orientation='h', y=1.12),
                title=dict(text=f"Evolución de goles — {selected_team}", x=0.5),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            st.markdown("### 📋 Partidos")
            for _, row in recent.iterrows():
                color = _RESULT_COLOR[row['result']]
                result_text = {'W': 'Victoria', 'D': 'Empate', 'L': 'Derrota'}[row['result']]
                st.markdown(
                    f'<div style="background:#1a1d23;border-left:4px solid {color};'
                    f'padding:10px 16px;margin:4px 0;border-radius:6px;'
                    f'display:flex;justify-content:space-between;align-items:center">'
                    f'<span style="color:#666;min-width:90px">{str(row["date"])[:10]}</span>'
                    f'<span style="flex:1;padding:0 12px"><b>vs {row["opponent"]}</b> '
                    f'<span style="color:#555;font-size:12px">({row["home_away"]})</span></span>'
                    f'<span style="color:{color};font-weight:bold;min-width:120px;text-align:center">'
                    f'{row["goals_for"]}–{row["goals_against"]} {result_text}</span>'
                    f'<span style="color:#444;font-size:11px;min-width:100px;text-align:right">'
                    f'{row["tournament"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
