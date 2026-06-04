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
    page_title="Football Predictor ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Professional CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp { background: #0a0c10 !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1200px; }

h1 {
    font-size: 2rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
}
h2 { font-size: 1.3rem !important; font-weight: 700 !important; color: #e8eaf0 !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #c8cad4 !important; margin-bottom: 0.5rem !important; }
p, li { color: #b0b3be !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: #555 !important; font-size: 0.8rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13161d !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 4px !important;
    border: 1px solid #1e2130 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 9px 22px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: #666 !important;
    letter-spacing: 0.2px;
    transition: all 0.2s ease !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d4aa, #00b4d8) !important;
    color: #000 !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(0,212,170,0.35) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%) !important;
    color: #000 !important;
    border: none !important;
    font-size: 1rem !important;
    padding: 0.7rem 1rem !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.3) !important;
    letter-spacing: 0.3px;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0,212,170,0.45) !important;
}

/* Selectbox */
.stSelectbox label { color: #888 !important; font-size: 0.72rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.5px; }
[data-baseweb="select"] > div:first-child {
    background: #13161d !important;
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-weight: 500 !important;
    transition: border-color 0.2s !important;
}
[data-baseweb="select"] > div:first-child:hover { border-color: #00d4aa !important; }

/* Checkbox */
.stCheckbox label p { color: #888 !important; font-size: 0.85rem !important; font-weight: 500 !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: #13161d !important;
    border: 1px solid #1e2130 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 1px rgba(0,212,170,0.15) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    color: #555 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #e8eaf0 !important;
    letter-spacing: -0.5px !important;
}

/* Divider */
hr { border-color: #1e2130 !important; margin: 1rem 0 !important; }

/* Alerts */
.stAlert { border-radius: 10px !important; border-left-width: 4px !important; }

/* Plotly container */
[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }

/* Spinner */
.stSpinner > div { border-top-color: #00d4aa !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #13161d; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4aa; }
</style>
""", unsafe_allow_html=True)

# ── Guards ─────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error("📂 Dataset no encontrado. Coloca `results.csv` en la carpeta `data/`.")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=['date'])


df = load_data()

if not os.path.exists(MODEL_PATH):
    with st.spinner("⚙️ Entrenando modelo por primera vez... (~30 segundos)"):
        from src.train import train
        train()

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("# ⚽ Football Match Predictor")
st.caption("Modelo Random Forest  ·  49 000+ partidos internacionales  ·  1872–2026")
st.markdown("<div style='margin-bottom:1.2rem'></div>", unsafe_allow_html=True)

teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
default_home = teams.index('Mexico') if 'Mexico' in teams else 0
default_away = teams.index('Argentina') if 'Argentina' in teams else min(1, len(teams) - 1)

col_h, col_vs, col_a = st.columns([5, 1, 5])
with col_h:
    home_team = st.selectbox("🏠 Equipo Local", teams, index=default_home)
with col_vs:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:100%;padding-top:22px;font-size:1.5rem;font-weight:900;"
        "color:#2a2d3a;letter-spacing:1px'>VS</div>",
        unsafe_allow_html=True,
    )
with col_a:
    away_team = st.selectbox("✈️ Equipo Visitante", teams, index=default_away)

opt_col, btn_col = st.columns([1, 3])
with opt_col:
    is_neutral = st.checkbox(
        "🌍 Cancha neutral",
        value=True,
        help="Actívalo para Mundial, Copa América u otros torneos en sede neutra",
    )
with btn_col:
    predict_btn = st.button("🎯  Predecir Partido", type="primary", use_container_width=True)

# ── Shared helpers ─────────────────────────────────────────────────────────
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#13161d",
    plot_bgcolor="#13161d",
    font=dict(color="#b0b3be", family="Inter, sans-serif", size=12),
    margin=dict(l=20, r=20, t=44, b=20),
)
_TC  = {'H': '#00d4aa', 'D': '#f59e0b', 'A': '#f43f5e'}
_RC  = {'W': '#00d4aa', 'D': '#f59e0b', 'L': '#f43f5e'}


def _badge(result):
    c = _RC.get(result, '#555')
    lbl = {'W': 'V', 'D': 'E', 'L': 'D'}.get(result, result)
    return (
        f'<span style="display:inline-block;background:{c};color:#000;'
        f'width:28px;height:28px;line-height:28px;text-align:center;'
        f'border-radius:6px;font-weight:800;font-size:12px;margin:2px">{lbl}</span>'
    )


def _card(inner, border="#1e2130"):
    return (
        f'<div style="background:#13161d;border:1px solid {border};'
        f'border-radius:14px;padding:20px 24px;margin-bottom:8px">{inner}</div>'
    )


def _match_row(date, label, score, tourn, color):
    return (
        f'<div style="display:grid;grid-template-columns:90px 1fr 160px 110px;'
        f'align-items:center;gap:12px;background:#13161d;'
        f'border-left:3px solid {color};border-radius:0 8px 8px 0;'
        f'padding:10px 16px;margin:4px 0">'
        f'<span style="color:#444;font-size:11px;font-weight:600">{date}</span>'
        f'<span style="color:#c8cad4;font-weight:600;font-size:0.9rem">{label}</span>'
        f'<span style="color:{color};font-weight:800;font-size:0.95rem;text-align:center">{score}</span>'
        f'<span style="color:#333;font-size:11px;text-align:right">{tourn}</span>'
        f'</div>'
    )


# ── Main prediction block ───────────────────────────────────────────────────
if predict_btn:
    if home_team == away_team:
        st.error("⚠️ Selecciona dos equipos diferentes.")
        st.stop()

    for team in [home_team, away_team]:
        if get_team_match_count(df, team) < 5:
            st.warning(f"⚠️ {team} tiene menos de 5 partidos en el historial.")

    try:
        prediction = predict_match(home_team, away_team, df, MODEL_PATH, is_neutral=is_neutral)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

    probs      = prediction['probabilities']
    res_key    = prediction['result']
    res_label  = prediction['result_label']
    confidence = max(probs.values()) * 100
    res_color  = _TC[res_key]

    home_stats = get_team_overall_stats(df, home_team)
    away_stats = get_team_overall_stats(df, away_team)
    radar      = get_radar_stats(df, home_team, away_team)
    h2h        = get_head_to_head(df, home_team, away_team)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯  Predicción", "📊  Estadísticas", "⚔️  Head-to-Head", "📋  Historial"]
    )

    # ═══════════════ TAB 1 · PREDICCIÓN ════════════════════════════════════
    with tab1:
        neutral_tag = (
            ' &nbsp;<span style="background:#1e2130;color:#555;font-size:11px;'
            'padding:3px 10px;border-radius:20px;font-weight:600">'
            '🌍 Cancha neutral</span>' if is_neutral else ''
        )
        st.markdown(_card(
            f'<div style="text-align:center">'
            f'<div style="font-size:2.4rem;font-weight:900;color:{res_color};'
            f'letter-spacing:-0.5px;line-height:1.1">{res_label.upper()}</div>'
            f'<div style="margin-top:10px;font-size:0.85rem;color:#555;font-weight:600">'
            f'CONFIANZA &nbsp;'
            f'<span style="color:#e8eaf0;font-size:1.1rem;font-weight:800">{confidence:.1f}%</span>'
            f'{neutral_tag}</div></div>',
            border=res_color,
        ), unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏠 {home_team}", f"{probs.get('H',0)*100:.1f}%")
        m2.metric("🤝 Empate",       f"{probs.get('D',0)*100:.1f}%")
        m3.metric(f"✈️ {away_team}", f"{probs.get('A',0)*100:.1f}%")

        h_v, d_v, a_v = probs.get('H',0), probs.get('D',0), probs.get('A',0)
        fig_prob = go.Figure(go.Bar(
            x=[h_v, d_v, a_v],
            y=[f"🏠 {home_team}", "🤝 Empate", f"✈️ {away_team}"],
            orientation='h',
            text=[f"<b>{v*100:.1f}%</b>" for v in [h_v, d_v, a_v]],
            textposition='inside',
            marker=dict(color=['#00d4aa','#f59e0b','#f43f5e'], line=dict(width=0)),
            insidetextfont=dict(size=14, color='#000'),
        ))
        fig_prob.update_layout(
            **_LAYOUT,
            xaxis=dict(tickformat='.0%', range=[0,1], showgrid=False, showticklabels=False),
            yaxis=dict(autorange='reversed', tickfont=dict(size=13, color='#c8cad4')),
            height=200, bargap=0.35,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for col, team, stats, flag in [
            (c1, home_team, home_stats, "🏠"),
            (c2, away_team, away_stats, "✈️"),
        ]:
            with col:
                badges = " ".join(_badge(r) for r in stats['form'][-5:]) or "—"
                st.markdown(_card(
                    f'<div style="font-size:1rem;font-weight:700;color:#e8eaf0;margin-bottom:14px">'
                    f'{flag} {team}</div>'
                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px">'
                    f'<div style="background:#0a0c10;border-radius:8px;padding:10px 12px">'
                    f'<div style="font-size:0.62rem;color:#444;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">⚽ Goles/PJ</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;color:#e8eaf0">{stats["goals_scored_per_game"]}</div></div>'
                    f'<div style="background:#0a0c10;border-radius:8px;padding:10px 12px">'
                    f'<div style="font-size:0.62rem;color:#444;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">🛡️ Recibidos/PJ</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;color:#e8eaf0">{stats["goals_conceded_per_game"]}</div></div>'
                    f'<div style="background:#0a0c10;border-radius:8px;padding:10px 12px">'
                    f'<div style="font-size:0.62rem;color:#444;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">🏆 Victorias</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;color:#00d4aa">{stats["win_rate"]*100:.0f}%</div></div>'
                    f'<div style="background:#0a0c10;border-radius:8px;padding:10px 12px">'
                    f'<div style="font-size:0.62rem;color:#444;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">📊 Partidos</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;color:#e8eaf0">{stats["total_matches"]}</div></div>'
                    f'</div>'
                    f'<div style="font-size:0.62rem;color:#444;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">FORMA RECIENTE</div>'
                    f'{badges}',
                ), unsafe_allow_html=True)

    # ═══════════════ TAB 2 · ESTADÍSTICAS ══════════════════════════════════
    with tab2:
        dims = radar['dimensions']
        fig_radar = go.Figure()
        for name, vals, color, fill in [
            (home_team, radar['home'], '#00d4aa', 'rgba(0,212,170,0.12)'),
            (away_team, radar['away'], '#f43f5e', 'rgba(244,63,94,0.12)'),
        ]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=dims + [dims[0]],
                fill='toself', name=name,
                line=dict(color=color, width=2.5),
                fillcolor=fill,
                marker=dict(size=6, color=color),
            ))
        fig_radar.update_layout(
            **_LAYOUT,
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1], color='#2a2d3a',
                                showticklabels=False, gridcolor='#1e2130'),
                angularaxis=dict(color='#b0b3be', gridcolor='#1e2130'),
                bgcolor='#13161d',
            ),
            height=420,
            legend=dict(orientation='h', y=-0.06, x=0.25, font=dict(size=13, color='#c8cad4')),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### 📋 Comparación de métricas")
        rows = [
            ("⚽ Goles anotados / PJ",   home_stats['goals_scored_per_game'],          away_stats['goals_scored_per_game'],          True),
            ("🛡️ Goles recibidos / PJ",  home_stats['goals_conceded_per_game'],        away_stats['goals_conceded_per_game'],        False),
            ("🏆 % Victorias",           f"{home_stats['win_rate']*100:.1f}%",         f"{away_stats['win_rate']*100:.1f}%",         True),
            ("🤝 % Empates",             f"{home_stats['draw_rate']*100:.1f}%",        f"{away_stats['draw_rate']*100:.1f}%",        None),
            ("❌ % Derrotas",            f"{home_stats['loss_rate']*100:.1f}%",        f"{away_stats['loss_rate']*100:.1f}%",        False),
            ("⚡ Partidos con gol",      f"{home_stats['scoring_rate']*100:.1f}%",     f"{away_stats['scoring_rate']*100:.1f}%",     True),
            ("🔒 Porterías a cero",      f"{home_stats['clean_sheet_rate']*100:.1f}%", f"{away_stats['clean_sheet_rate']*100:.1f}%", True),
            ("📊 Total partidos",        home_stats['total_matches'],                  away_stats['total_matches'],                  None),
        ]
        table = '<div style="background:#13161d;border:1px solid #1e2130;border-radius:14px;overflow:hidden">'
        for i, (name, hv, av, hib) in enumerate(rows):
            bg = "#0f1118" if i % 2 == 0 else "#13161d"
            hs = as2 = "color:#c8cad4"
            if hib is not None:
                try:
                    hn = float(str(hv).replace('%',''))
                    an = float(str(av).replace('%',''))
                    wh = (hn >= an) if hib else (hn <= an)
                    wa = (an > hn)  if hib else (an < hn)
                    hs  = "color:#00d4aa;font-weight:700" if wh else "color:#555"
                    as2 = "color:#00d4aa;font-weight:700" if wa else "color:#555"
                except (ValueError, TypeError):
                    pass
            table += (
                f'<div style="display:grid;grid-template-columns:1fr 220px 1fr;'
                f'align-items:center;padding:12px 20px;background:{bg}">'
                f'<div style="text-align:right;font-size:0.95rem;{hs}">{hv}</div>'
                f'<div style="text-align:center;font-size:0.75rem;font-weight:700;'
                f'color:#444;text-transform:uppercase;letter-spacing:0.4px">{name}</div>'
                f'<div style="text-align:left;font-size:0.95rem;{as2}">{av}</div>'
                f'</div>'
            )
        table += '</div>'
        st.markdown(table, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("### 📅 Forma — últimos 10 partidos")
        fc1, fc2 = st.columns(2)
        for col, team, stats in [(fc1, home_team, home_stats), (fc2, away_team, away_stats)]:
            with col:
                badges = " ".join(_badge(r) for r in stats['form']) or "Sin datos"
                st.markdown(
                    f'<div style="background:#13161d;border:1px solid #1e2130;'
                    f'border-radius:12px;padding:16px 20px">'
                    f'<div style="font-size:0.7rem;font-weight:700;color:#444;'
                    f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px">{team}</div>'
                    f'{badges}</div>',
                    unsafe_allow_html=True,
                )

    # ═══════════════ TAB 3 · HEAD-TO-HEAD ══════════════════════════════════
    with tab3:
        total_h2h = h2h['home_wins'] + h2h['draws'] + h2h['away_wins']
        if total_h2h == 0:
            st.info(f"ℹ️ No hay enfrentamientos directos entre **{home_team}** y **{away_team}**.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"🏆 {home_team}", h2h['home_wins'])
            c2.metric("🤝 Empates",      h2h['draws'])
            c3.metric(f"🏆 {away_team}", h2h['away_wins'])

            left, right = st.columns(2)
            with left:
                fig_donut = go.Figure(go.Pie(
                    labels=[home_team, "Empate", away_team],
                    values=[h2h['home_wins'], h2h['draws'], h2h['away_wins']],
                    hole=0.62,
                    marker=dict(colors=['#00d4aa','#f59e0b','#f43f5e'],
                                line=dict(color='#13161d', width=3)),
                    textinfo='label+percent',
                    textfont=dict(size=12),
                    pull=[0.03, 0.03, 0.03],
                ))
                fig_donut.update_layout(
                    **_LAYOUT, height=310, showlegend=False,
                    title=dict(text="<b>Distribución histórica</b>", x=0.5,
                               font=dict(size=13, color='#555')),
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with right:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                gc1, gc2, gc3 = st.columns(3)
                gc1.metric(f"⚽ {home_team}", int(h2h['home_goals_total']))
                gc2.metric("📊 Goles/PJ",    h2h['avg_goals_per_game'])
                gc3.metric(f"⚽ {away_team}", int(h2h['away_goals_total']))

                if h2h['last_matches']:
                    chron = list(reversed(h2h['last_matches']))
                    dates = [m['date'] for m in chron]
                    hg = [m['home_score'] if m['home_team']==home_team else m['away_score'] for m in chron]
                    ag = [m['away_score'] if m['home_team']==home_team else m['home_score'] for m in chron]
                    fig_g = go.Figure()
                    fig_g.add_trace(go.Bar(name=home_team, x=dates, y=hg,
                                          marker=dict(color='#00d4aa', line=dict(width=0))))
                    fig_g.add_trace(go.Bar(name=away_team, x=dates, y=ag,
                                          marker=dict(color='#f43f5e', line=dict(width=0))))
                    fig_g.update_layout(
                        **_LAYOUT, barmode='group', height=250,
                        title=dict(text="<b>Goles por partido</b>", x=0.5,
                                   font=dict(size=13, color='#555')),
                        legend=dict(orientation='h', y=1.15, font=dict(size=11)),
                        xaxis=dict(tickfont=dict(size=10)),
                        bargap=0.25, bargroupgap=0.08,
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

            if h2h['last_matches']:
                st.markdown("### 📋 Últimos enfrentamientos")
                for m in h2h['last_matches']:
                    ht, at = m['home_team'], m['away_team']
                    hs, as_ = m['home_score'], m['away_score']
                    res = ('W' if (ht==home_team and hs>as_) or (at==home_team and as_>hs)
                           else 'D' if hs==as_ else 'L')
                    st.markdown(
                        _match_row(m['date'], f"{ht} {hs} — {as_} {at}",
                                   "", m['tournament'], _RC[res]),
                        unsafe_allow_html=True,
                    )

    # ═══════════════ TAB 4 · HISTORIAL ═════════════════════════════════════
    with tab4:
        hc1, hc2 = st.columns([2, 1])
        with hc1:
            sel = st.selectbox("Equipo", [home_team, away_team], key="hist_team")
        with hc2:
            n_opt = st.select_slider("Partidos", options=[10, 20, 50, 100, 'Todos'],
                                     value=20, key="hist_n")

        recent = get_recent_matches(df, sel, n=None if n_opt=='Todos' else int(n_opt))

        if len(recent) == 0:
            st.info("ℹ️ Sin partidos registrados.")
        else:
            wins_h  = int((recent['result']=='W').sum())
            draws_h = int((recent['result']=='D').sum())
            loss_h  = int((recent['result']=='L').sum())
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("🏆 Victorias",       wins_h)
            sc2.metric("🤝 Empates",         draws_h)
            sc3.metric("❌ Derrotas",        loss_h)
            sc4.metric("⚽ Goles a favor",   int(recent['goals_for'].sum()))
            sc5.metric("🛡️ Goles en contra", int(recent['goals_against'].sum()))

            asc = recent.sort_values('date')
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=asc['date'], y=asc['goals_for'], name='Anotados',
                line=dict(color='#00d4aa', width=2.5),
                mode='lines+markers',
                marker=dict(size=7, color='#00d4aa', line=dict(width=2, color='#0a0c10')),
                fill='tozeroy', fillcolor='rgba(0,212,170,0.06)',
            ))
            fig_tl.add_trace(go.Scatter(
                x=asc['date'], y=asc['goals_against'], name='Recibidos',
                line=dict(color='#f43f5e', width=2.5),
                mode='lines+markers',
                marker=dict(size=7, color='#f43f5e', line=dict(width=2, color='#0a0c10')),
                fill='tozeroy', fillcolor='rgba(244,63,94,0.06)',
            ))
            fig_tl.update_layout(
                **_LAYOUT, height=290,
                xaxis_title=None, yaxis_title="Goles",
                legend=dict(orientation='h', y=1.12, font=dict(size=12)),
                title=dict(text=f"<b>Evolución de goles — {sel}</b>", x=0.5,
                           font=dict(size=13, color='#555')),
                yaxis=dict(gridcolor='#1e2130'),
                xaxis=dict(gridcolor='#1e2130'),
            )
            st.plotly_chart(fig_tl, use_container_width=True)

            st.markdown("### 📋 Partidos")
            for _, row in recent.iterrows():
                color   = _RC[row['result']]
                res_txt = {'W':'Victoria','D':'Empate','L':'Derrota'}[row['result']]
                gf, ga  = int(row['goals_for']), int(row['goals_against'])
                st.markdown(
                    _match_row(
                        str(row['date'])[:10],
                        f"vs <b>{row['opponent']}</b> "
                        f'<span style="color:#2a2d3a;font-size:11px">({row["home_away"]})</span>',
                        f"{gf}–{ga} · {res_txt}",
                        row['tournament'],
                        color,
                    ),
                    unsafe_allow_html=True,
                )
