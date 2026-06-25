# -*- coding: utf-8 -*-
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from src.predict import predict_match, get_team_match_count, predict_scoreline
from src.stats import (
    get_team_overall_stats,
    get_radar_stats,
    get_head_to_head,
    get_recent_matches,
    get_recent_performance,
    get_stat_sparklines,
)
from src.player_features import load_players, get_team_player_features, get_team_squad
from src.betting import compute_all_markets, best_bet, expected_value, recommend_combined

DATA_PATH = 'data/all_matches.csv' if os.path.exists('data/all_matches.csv') else 'data/results.csv'
MODEL_PATH = 'model.pkl'

st.set_page_config(page_title="Tonini Predictor", page_icon="⚽", layout="wide")

# ── Professional CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {
    font-family: 'Rajdhani', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Streamlit's icons are ligature text in the Material Symbols font; the global
   Inter override above must not apply to them or the icon name renders as text */
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
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
h2 { font-size: 1.2rem !important; font-weight: 700 !important; color: #00d4aa !important; text-shadow: 0 0 20px rgba(0,212,170,0.5) !important; letter-spacing: 0.5px; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #c8cad4 !important; margin-bottom: 0.5rem !important; }
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

/* Market boxes */
.market-box {
    background: #13161d;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.market-box.hot { border-color: #00d4aa; }
</style>
""", unsafe_allow_html=True)

# ── Guards ─────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Place `results.csv` in the `data/` folder.")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=['date'])


df = load_data()


@st.cache_data
def load_players_cached():
    return load_players()


players_df = load_players_cached()

if not os.path.exists(MODEL_PATH):
    with st.spinner("Training model for the first time... (~30 seconds)"):
        from src.train import train
        train()

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("# Tonini Predictor")
st.caption("Modelo XGBoost · Mundial 2026 · Análisis completo de apuestas")
st.markdown("<div style='margin-bottom:1.2rem'></div>", unsafe_allow_html=True)

# ── Competition selector ────────────────────────────────────────────────────
if 'league' in df.columns:
    league_list = sorted(df['league'].dropna().unique())
    competition_type_map = (
        df.groupby('league')['competition_type'].first().to_dict()
        if 'competition_type' in df.columns else {}
    )
else:
    league_list = []
    competition_type_map = {}

if 'selected_league' not in st.session_state:
    st.session_state.selected_league = 'All'

st.markdown(
    '<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
    'letter-spacing:0.8px;color:#444;margin-bottom:8px">Competition</div>',
    unsafe_allow_html=True,
)

other_options = ['All'] + league_list
other_idx = (other_options.index(st.session_state.selected_league)
             if st.session_state.selected_league in other_options else 0)
chosen = st.selectbox("Other competitions", other_options, index=other_idx, label_visibility='collapsed')
if chosen != st.session_state.selected_league:
    st.session_state.selected_league = chosen
    st.rerun()

selected_league = st.session_state.selected_league

if selected_league == 'All' or 'league' not in df.columns:
    filtered_df = df
    league_param = None
    comp_type_param = 'international' if 'league' not in df.columns else None
else:
    filtered_df = df[df['league'] == selected_league]
    league_param = selected_league
    comp_type_param = competition_type_map.get(selected_league, 'club')

teams = sorted(set(filtered_df['home_team'].unique()) | set(filtered_df['away_team'].unique()))
default_home = teams.index('Mexico') if 'Mexico' in teams else 0
default_away = teams.index('Argentina') if 'Argentina' in teams else min(1, len(teams) - 1)

col_h, col_vs, col_a = st.columns([5, 1, 5])
with col_h:
    home_team = st.selectbox("Home", teams, index=default_home)
with col_vs:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:100%;padding-top:22px;font-size:1.5rem;font-weight:900;"
        "color:#2a2d3a;letter-spacing:1px'>VS</div>",
        unsafe_allow_html=True,
    )
with col_a:
    away_team = st.selectbox("Away", teams, index=default_away)

# ── Odds expander (full grid) ───────────────────────────────────────────────
with st.expander("Cuotas del corredor (opcional) — cuotas decimales de tu casa de apuestas"):
    row1 = st.columns(6)
    odds_h      = row1[0].number_input("Local (1)",        min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_d      = row1[1].number_input("Empate (X)",       min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_a      = row1[2].number_input("Visitante (2)",    min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_o15    = row1[3].number_input("O1.5",             min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_u15    = row1[4].number_input("U1.5",             min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_o25    = row1[5].number_input("O2.5",             min_value=0.0, value=0.0, step=0.05, format="%.2f")
    row2 = st.columns(6)
    odds_u25    = row2[0].number_input("U2.5",             min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_o35    = row2[1].number_input("O3.5",             min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_u35    = row2[2].number_input("U3.5",             min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_btts_y = row2[3].number_input("BTTS Sí",          min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_btts_n = row2[4].number_input("BTTS No",          min_value=0.0, value=0.0, step=0.05, format="%.2f")

# Map market key -> user-entered odd
_ODDS_MAP = {
    '1': odds_h, 'X': odds_d, '2': odds_a,
    'O15': odds_o15, 'U15': odds_u15,
    'O25': odds_o25, 'U25': odds_u25,
    'O35': odds_o35, 'U35': odds_u35,
    'BTTS_Y': odds_btts_y, 'BTTS_N': odds_btts_n,
}

_WC_HOSTS = {'United States', 'Mexico', 'Canada'}
# En el Mundial 2026 solo USA/México/Canadá juegan de locales; el resto es sede neutral.
_auto_neutral = home_team not in _WC_HOSTS

opt_col, btn_col = st.columns([1, 3])
with opt_col:
    is_neutral = st.checkbox(
        "Sede neutral",
        value=_auto_neutral,
        key=f"neutral_{home_team}",  # resetea al valor auto cuando cambia el equipo local
        help="Se activa automáticamente salvo que el local sea USA, México o Canadá (sedes del Mundial 2026)",
    )
with btn_col:
    predict_btn = st.button("Predecir Partido", type="primary", use_container_width=True)

# ── Shared helpers ─────────────────────────────────────────────────────────
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#13161d",
    plot_bgcolor="#13161d",
    font=dict(color="#b0b3be", family="Rajdhani, sans-serif", size=12),
    margin=dict(l=20, r=20, t=44, b=20),
)
_TC = {'H': '#00d4aa', 'D': '#f59e0b', 'A': '#f43f5e'}
_RC = {'W': '#00d4aa', 'D': '#f59e0b', 'L': '#f43f5e'}


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


def _market_box(label, prob, odd=None):
    """Render a single market box as HTML. Returns an HTML string."""
    if prob is None:
        prob_text = "—"
        is_hot = False
        color = "#e8eaf0"
        border = "#1e2130"
    else:
        prob_pct = prob * 100
        prob_text = f"{prob_pct:.0f}%"
        is_hot = prob >= 0.60
        color = "#00d4aa" if is_hot else "#e8eaf0"
        border = "#00d4aa" if is_hot else "#1e2130"

    tag = ""
    if is_hot:
        tag = '<div style="font-size:0.6rem;font-weight:800;color:#00d4aa;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px">APOSTAR</div>'

    if prob is not None and odd is not None and odd > 1.0:
        ev = expected_value(prob, odd)
        if ev is not None:
            ev_color = "#00d4aa" if ev > 0 else "#f43f5e"
            ev_glow = "rgba(0,212,170,0.4)" if ev > 0 else "rgba(244,63,94,0.4)"
            ev_html = (
                f'<div style="font-size:0.82rem;font-weight:800;color:{ev_color};'
                f'margin-top:5px;letter-spacing:0.4px;'
                f'text-shadow:0 0 8px {ev_glow}">EV {ev:+.2f}</div>'
                f'<div style="font-size:0.62rem;color:{ev_color};opacity:0.7">({ev*100:+.1f}%)</div>'
            )
        else:
            ev_html = '<div style="font-size:0.7rem;color:#333;margin-top:5px">EV —</div>'
    elif prob is not None and odd is not None:
        ev_html = '<div style="font-size:0.7rem;color:#333;margin-top:5px">EV —</div>'
    elif prob is not None:
        ev_html = '<div style="font-size:0.65rem;color:#2a2d3a;margin-top:5px">cuota →</div>'
    else:
        ev_html = ""

    return (
        f'<div style="background:#13161d;border:1px solid {border};border-radius:10px;'
        f'padding:12px 16px;text-align:center;transition:border-color 0.2s">'
        f'<div style="font-size:0.65rem;text-transform:uppercase;color:#555;font-weight:600;letter-spacing:0.4px;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:1.3rem;font-weight:800;color:{color}">{prob_text}</div>'
        f'{tag}{ev_html}'
        f'</div>'
    )


def _section_header(title):
    return (
        f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.8px;color:#444;margin:18px 0 10px 0;border-bottom:1px solid #1e2130;'
        f'padding-bottom:6px">{title}</div>'
    )


# ── Main prediction block ───────────────────────────────────────────────────
if predict_btn:
    if home_team == away_team:
        st.error("Please select two different teams.")
        st.stop()

    for team in [home_team, away_team]:
        if get_team_match_count(df, team) < 5:
            st.warning(f"{team} has fewer than 5 matches in the dataset.")

    try:
        prediction = predict_match(
            home_team, away_team, df, MODEL_PATH,
            is_neutral=is_neutral,
            league=league_param,
            competition_type=comp_type_param,
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    try:
        scorelines, lam_h, lam_a = predict_scoreline(df, home_team, away_team, is_neutral=is_neutral)
    except Exception:
        scorelines, lam_h, lam_a = [], None, None

    probs       = prediction['probabilities']
    res_key     = prediction['result']
    res_label   = prediction['result_label']
    confidence  = max(probs.values()) * 100
    res_color   = _TC[res_key]

    over_1_5_prob = prediction.get('over_1_5_prob')
    over_2_5_prob = prediction.get('over_2_5_prob')
    over_3_5_prob = prediction.get('over_3_5_prob')
    btts_prob     = prediction.get('btts_prob')

    home_stats = get_team_overall_stats(df, home_team)
    away_stats = get_team_overall_stats(df, away_team)
    radar      = get_radar_stats(df, home_team, away_team)
    h2h        = get_head_to_head(df, home_team, away_team)

    # Compute all markets
    markets = compute_all_markets(probs, over_1_5_prob, over_2_5_prob, over_3_5_prob, btts_prob)

    tab1, tab2, tab3 = st.tabs(["Pronóstico", "Estadísticas", "Cara a Cara & Historial"])

    # ═══════════════ TAB 1 · PRONÓSTICO ════════════════════════════════════
    with tab1:
        # 1. Hero card
        neutral_tag = (
            ' &nbsp;<span style="background:#1e2130;color:#555;font-size:11px;'
            'padding:3px 10px;border-radius:20px;font-weight:600">'
            'Sede neutral</span>' if is_neutral else ''
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

        # 1b. Scoreline heatmap
        if scorelines and lam_h is not None:
            import math as _math

            top_h, top_a, top_p = scorelines[0]

            # Build full probability matrix 0–6 x 0–6
            _max_g = 6
            _goals = list(range(_max_g + 1))

            def _pmf(k, lam):
                try:
                    return (lam ** k) * _math.exp(-lam) / _math.factorial(k)
                except OverflowError:
                    return 0.0

            z_mat = [[_pmf(h, lam_h) * _pmf(a, lam_a) * 100 for a in _goals] for h in _goals]
            text_mat = [[f"{z_mat[h][a]:.1f}%" for a in _goals] for h in _goals]

            fig_sc = go.Figure(go.Heatmap(
                z=z_mat,
                x=[f"{a}" for a in _goals],
                y=[f"{h}" for h in _goals],
                text=text_mat,
                texttemplate="%{text}",
                textfont=dict(size=11, color='#e8eaf0', family='Rajdhani, sans-serif'),
                colorscale=[
                    [0.0,  '#0a0c10'],
                    [0.25, '#0d2e2a'],
                    [0.55, '#005c4b'],
                    [0.8,  '#00a884'],
                    [1.0,  '#00d4aa'],
                ],
                showscale=False,
                hovertemplate=(
                    f"<b>{home_team} %{{y}} — %{{x}} {away_team}</b><br>"
                    "Probabilidad: %{text}<extra></extra>"
                ),
            ))

            # Highlight most probable cell with a contrasting border shape
            fig_sc.add_shape(
                type='rect',
                x0=top_a - 0.5, x1=top_a + 0.5,
                y0=top_h - 0.5, y1=top_h + 0.5,
                line=dict(color='#00d4aa', width=3),
                fillcolor='rgba(0,0,0,0)',
                layer='above',
            )

            _sc_layout = {
                **_LAYOUT,
                'margin': dict(l=60, r=20, t=80, b=20),
                'height': 380,
                'title': dict(
                    text=(
                        f"<b>MARCADOR MÁS PROBABLE: "
                        f"<span style='color:#00d4aa'>{top_h} — {top_a}</span>"
                        f"  ({top_p*100:.1f}%)</b>"
                        f"&nbsp;&nbsp;&nbsp;"
                        f"<span style='font-size:11px;color:#555'>xG: {home_team} {lam_h:.2f} · {away_team} {lam_a:.2f}</span>"
                    ),
                    font=dict(size=13, color='#e8eaf0'),
                    x=0,
                ),
                'xaxis': dict(
                    title=dict(text=f"Goles {away_team} (Visitante)", font=dict(size=11)),
                    tickfont=dict(size=12, color='#b0b3be'),
                    side='top',
                ),
                'yaxis': dict(
                    title=dict(text=f"Goles {home_team} (Local)", font=dict(size=11)),
                    tickfont=dict(size=12, color='#b0b3be'),
                    autorange='reversed',
                ),
            }
            fig_sc.update_layout(**_sc_layout)
            st.plotly_chart(fig_sc, width='stretch', theme=None)

        # 2. Barra de probabilidades 1X2
        h_v, d_v, a_v = probs.get('H', 0), probs.get('D', 0), probs.get('A', 0)
        fig_prob = go.Figure(go.Bar(
            x=[h_v, d_v, a_v],
            y=[f"Local — {home_team}", "Empate", f"Visit. — {away_team}"],
            orientation='h',
            text=[f"<b>{v * 100:.1f}%</b>" for v in [h_v, d_v, a_v]],
            textposition='inside',
            marker=dict(color=['#00d4aa', '#f59e0b', '#f43f5e'], line=dict(width=0)),
            insidetextfont=dict(size=14, color='#000'),
        ))
        fig_prob.update_layout(
            **_LAYOUT,
            xaxis=dict(tickformat='.0%', range=[0, 1], showgrid=False, showticklabels=False),
            yaxis=dict(autorange='reversed', tickfont=dict(size=13, color='#c8cad4')),
            height=180, bargap=0.35,
        )
        st.plotly_chart(fig_prob, width='stretch', theme=None)

        # 3. MERCADOS DE APUESTA
        st.markdown(_section_header("MERCADOS DE APUESTA — RESULTADO"), unsafe_allow_html=True)

        # Row 1: 1, X, 2
        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(_market_box("Local (1)", markets['1']['prob'], _ODDS_MAP.get('1')), unsafe_allow_html=True)
        rc2.markdown(_market_box("Empate (X)", markets['X']['prob'], _ODDS_MAP.get('X')), unsafe_allow_html=True)
        rc3.markdown(_market_box("Visitante (2)", markets['2']['prob'], _ODDS_MAP.get('2')), unsafe_allow_html=True)

        # Row 2: 1X, X2, 12
        dc1, dc2, dc3 = st.columns(3)
        dc1.markdown(_market_box("Doble op. 1X", markets['1X']['prob']), unsafe_allow_html=True)
        dc2.markdown(_market_box("Doble op. X2", markets['X2']['prob']), unsafe_allow_html=True)
        dc3.markdown(_market_box("Doble op. 12", markets['12']['prob']), unsafe_allow_html=True)

        # Row 3: DNB
        dnb1, dnb2, _ = st.columns(3)
        dnb1.markdown(_market_box("DNB Local", markets['DNB1']['prob']), unsafe_allow_html=True)
        dnb2.markdown(_market_box("DNB Visitante", markets['DNB2']['prob']), unsafe_allow_html=True)

        st.markdown(_section_header("MERCADOS DE APUESTA — GOLES"), unsafe_allow_html=True)

        # Goles row 1: O1.5, U1.5, O2.5, U2.5
        gr1, gr2, gr3, gr4 = st.columns(4)
        gr1.markdown(_market_box("O 1.5", markets['O15']['prob'], _ODDS_MAP.get('O15')), unsafe_allow_html=True)
        gr2.markdown(_market_box("U 1.5", markets['U15']['prob'], _ODDS_MAP.get('U15')), unsafe_allow_html=True)
        gr3.markdown(_market_box("O 2.5", markets['O25']['prob'], _ODDS_MAP.get('O25')), unsafe_allow_html=True)
        gr4.markdown(_market_box("U 2.5", markets['U25']['prob'], _ODDS_MAP.get('U25')), unsafe_allow_html=True)

        # Goles row 2: O3.5, U3.5, BTTS_Y, BTTS_N
        gr5, gr6, gr7, gr8 = st.columns(4)
        gr5.markdown(_market_box("O 3.5", markets['O35']['prob'], _ODDS_MAP.get('O35')), unsafe_allow_html=True)
        gr6.markdown(_market_box("U 3.5", markets['U35']['prob'], _ODDS_MAP.get('U35')), unsafe_allow_html=True)
        gr7.markdown(_market_box("BTTS Sí", markets['BTTS_Y']['prob'], _ODDS_MAP.get('BTTS_Y')), unsafe_allow_html=True)
        gr8.markdown(_market_box("BTTS No", markets['BTTS_N']['prob'], _ODDS_MAP.get('BTTS_N')), unsafe_allow_html=True)

        st.markdown(_section_header("MERCADOS DE APUESTA — HÁNDICAP ASIÁTICO"), unsafe_allow_html=True)

        ah1, ah2, ah3, ah4 = st.columns(4)
        ah1.markdown(_market_box("Local -0.5", markets['AH_H05']['prob']), unsafe_allow_html=True)
        ah2.markdown(_market_box("Local +0.5", markets['AH_H05P']['prob']), unsafe_allow_html=True)
        ah3.markdown(_market_box("Visit. -0.5", markets['AH_A05']['prob']), unsafe_allow_html=True)
        ah4.markdown(_market_box("Visit. +0.5", markets['AH_A05P']['prob']), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # 4. MEJOR APUESTA — card destacada
        best = best_bet(markets, threshold=0.60)
        combined = recommend_combined(probs, over_2_5_prob)

        if best is not None:
            bk, bv = best
            b_prob = bv['prob']
            b_label = bv['label']
            b_ev = expected_value(b_prob, _ODDS_MAP.get(bk)) if _ODDS_MAP.get(bk, 0) > 1.0 else None
            if b_ev is not None:
                _ec = "#00d4aa" if b_ev > 0 else "#f43f5e"
                _eg = "rgba(0,212,170,0.5)" if b_ev > 0 else "rgba(244,63,94,0.5)"
                ev_line = (
                    f'<div style="font-size:1.2rem;font-weight:900;color:{_ec};'
                    f'margin-top:8px;letter-spacing:0.5px;text-shadow:0 0 14px {_eg}">'
                    f'EV {b_ev:+.2f} &nbsp;<span style="font-size:0.85rem;font-weight:600;opacity:0.8">({b_ev*100:+.1f}%)</span></div>'
                )
            else:
                ev_line = ''
            best_inner = (
                f'<div style="font-size:0.62rem;color:#00d4aa;font-weight:800;text-transform:uppercase;'
                f'letter-spacing:0.6px;margin-bottom:10px">⭐ APUESTA RECOMENDADA</div>'
                f'<div style="font-size:1.4rem;font-weight:900;color:#00d4aa">{b_label}</div>'
                f'<div style="font-size:0.95rem;font-weight:700;color:#e8eaf0;margin-top:4px">{b_prob * 100:.1f}% de probabilidad</div>'
                f'{ev_line}'
                f'<div style="margin-top:12px;font-size:0.82rem;font-weight:600;color:#555">'
                f'→ {combined["combined_label"]}</div>'
            )
            st.markdown(_card(best_inner, border='#00d4aa'), unsafe_allow_html=True)
        else:
            no_bet_inner = (
                f'<div style="font-size:0.62rem;color:#555;font-weight:800;text-transform:uppercase;'
                f'letter-spacing:0.6px;margin-bottom:10px">SIN APUESTA CLARA</div>'
                f'<div style="font-size:0.95rem;font-weight:600;color:#888">'
                f'Ningún mercado supera el umbral del 60%. Proceder con cautela.</div>'
                f'<div style="margin-top:10px;font-size:0.82rem;font-weight:600;color:#555">'
                f'→ {combined["combined_label"]}</div>'
            )
            st.markdown(_card(no_bet_inner, border='#2a2d3a'), unsafe_allow_html=True)

        st.caption(
            "Herramienta estadística — las predicciones no garantizan resultados. "
            "Apuesta con responsabilidad."
        )

        # 5. Squads
        home_squad = get_team_squad(players_df, home_team)
        away_squad = get_team_squad(players_df, away_team)
        if players_df is None:
            st.caption("Squad data unavailable — download the FIFA players dataset "
                       "from Kaggle and save it as `data/players.csv` to enable "
                       "squad tables and player-quality features.")
        elif len(home_squad) or len(away_squad):
            with st.expander("Squads — top players by rating"):
                sq1, sq2 = st.columns(2)
                squad_cols = {'short_name': 'Player', 'player_positions': 'Pos',
                              'age': 'Age', 'overall': 'Rating', 'shooting': 'Shooting',
                              'attacking_finishing': 'Finishing', 'pace': 'Pace'}
                for col, team, squad in [(sq1, home_team, home_squad),
                                         (sq2, away_team, away_squad)]:
                    with col:
                        st.markdown(f"**{team}**")
                        if len(squad):
                            st.dataframe(squad.rename(columns=squad_cols),
                                         hide_index=True, use_container_width=True)
                        else:
                            st.caption("No player data for this team.")

            with st.expander("Squad comparison"):
                h_pf = get_team_player_features(players_df, home_team)
                a_pf = get_team_player_features(players_df, away_team)
                dims = ['Rating', 'Attack', 'Defense']
                fig_sq = go.Figure()
                for name, pf, color in [(home_team, h_pf, '#00d4aa'),
                                        (away_team, a_pf, '#f43f5e')]:
                    fig_sq.add_trace(go.Bar(
                        name=name, x=dims,
                        y=[pf['team_rating'] * 100, pf['team_attack'] * 100,
                           pf['team_defense'] * 100],
                        marker_color=color,
                    ))
                fig_sq.update_layout(**_LAYOUT, barmode='group', height=300,
                                     yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_sq, width='stretch', theme=None)

        # 6. Rendimiento reciente (solo aqui, no en otros tabs)
        with st.expander("Rendimiento reciente — últimos 5 partidos"):
            h_perf = get_recent_performance(df, home_team)
            a_perf = get_recent_performance(df, away_team)
            h_spark = get_stat_sparklines(df, home_team)
            a_spark = get_stat_sparklines(df, away_team)

            stat_keys = ['shots', 'shots_on_target', 'corners', 'yellow']
            stat_labels = ['Tiros', 'Tiros al arco', 'Córners', 'Amarillas']

            has_data = any(h_perf.get(k) is not None for k in stat_keys) or \
                       any(a_perf.get(k) is not None for k in stat_keys)

            if has_data:
                h_avgs = [h_perf.get(k) or 0 for k in stat_keys]
                a_avgs = [a_perf.get(k) or 0 for k in stat_keys]

                fig_perf = go.Figure()
                fig_perf.add_trace(go.Bar(
                    name=home_team, x=stat_labels, y=h_avgs,
                    marker=dict(color='#00d4aa', line=dict(width=0)),
                ))
                fig_perf.add_trace(go.Bar(
                    name=away_team, x=stat_labels, y=a_avgs,
                    marker=dict(color='#f43f5e', line=dict(width=0)),
                ))
                fig_perf.update_layout(
                    **_LAYOUT, barmode='group', height=220,
                    yaxis=dict(gridcolor='#1e2130'),
                    legend=dict(orientation='h', y=1.18, font=dict(size=11)),
                    bargap=0.25, bargroupgap=0.1,
                )
                st.plotly_chart(fig_perf, width='stretch', theme=None)

                fig_spark = make_subplots(rows=2, cols=2, subplot_titles=stat_labels,
                                          vertical_spacing=0.18, horizontal_spacing=0.12)
                for idx, key in enumerate(stat_keys):
                    row, col = divmod(idx, 2)
                    h_vals = h_spark.get(key, [])
                    a_vals = a_spark.get(key, [])
                    show_legend = (idx == 0)
                    if len(h_vals) >= 2:
                        fig_spark.add_trace(go.Scatter(
                            x=list(range(1, len(h_vals) + 1)), y=h_vals,
                            name=home_team, line=dict(color='#00d4aa', width=2),
                            mode='lines+markers', marker=dict(size=5),
                            showlegend=show_legend,
                        ), row=row + 1, col=col + 1)
                    if len(a_vals) >= 2:
                        fig_spark.add_trace(go.Scatter(
                            x=list(range(1, len(a_vals) + 1)), y=a_vals,
                            name=away_team, line=dict(color='#f43f5e', width=2),
                            mode='lines+markers', marker=dict(size=5),
                            showlegend=show_legend,
                        ), row=row + 1, col=col + 1)
                fig_spark.update_xaxes(showticklabels=False, showgrid=False)
                fig_spark.update_yaxes(showgrid=False)
                fig_spark.update_layout(
                    **_LAYOUT, height=200,
                    legend=dict(orientation='h', y=1.12, font=dict(size=11)),
                )
                st.plotly_chart(fig_spark, width='stretch', theme=None)
            else:
                st.caption(
                    "Las estadísticas de partido solo están disponibles para ligas de "
                    "clubes (no para partidos internacionales)."
                )

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
                radialaxis=dict(visible=True, range=[0, 1], color='#2a2d3a',
                                showticklabels=False, gridcolor='#1e2130'),
                angularaxis=dict(color='#b0b3be', gridcolor='#1e2130'),
                bgcolor='#13161d',
            ),
            height=420,
            legend=dict(orientation='h', y=-0.06, x=0.25, font=dict(size=13, color='#c8cad4')),
        )
        st.plotly_chart(fig_radar, width='stretch', theme=None)

        st.markdown("### Comparativa de métricas")
        rows = [
            ("Goles marcados / Partido",   home_stats['goals_scored_per_game'],          away_stats['goals_scored_per_game'],          True),
            ("Goles encajados / Partido",  home_stats['goals_conceded_per_game'],         away_stats['goals_conceded_per_game'],         False),
            ("% victorias",                f"{home_stats['win_rate']*100:.1f}%",          f"{away_stats['win_rate']*100:.1f}%",          True),
            ("% empates",                  f"{home_stats['draw_rate']*100:.1f}%",         f"{away_stats['draw_rate']*100:.1f}%",         None),
            ("% derrotas",                 f"{home_stats['loss_rate']*100:.1f}%",         f"{away_stats['loss_rate']*100:.1f}%",         False),
            ("% partidos con gol",         f"{home_stats['scoring_rate']*100:.1f}%",      f"{away_stats['scoring_rate']*100:.1f}%",      True),
            ("% portería a 0",             f"{home_stats['clean_sheet_rate']*100:.1f}%",  f"{away_stats['clean_sheet_rate']*100:.1f}%",  True),
            ("Total partidos",             home_stats['total_matches'],                   away_stats['total_matches'],                   None),
        ]
        table = '<div style="background:#13161d;border:1px solid #1e2130;border-radius:14px;overflow:hidden">'
        for i, (name, hv, av, hib) in enumerate(rows):
            bg = "#0f1118" if i % 2 == 0 else "#13161d"
            hs = as2 = "color:#c8cad4"
            if hib is not None:
                try:
                    hn = float(str(hv).replace('%', ''))
                    an = float(str(av).replace('%', ''))
                    wh = (hn >= an) if hib else (hn <= an)
                    wa = (an > hn) if hib else (an < hn)
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
        st.markdown("### Forma reciente — últimos 10 partidos")
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

    # ═══════════════ TAB 3 · CARA A CARA & HISTORIAL ═══════════════════════
    with tab3:
        # ── H2H section ──
        total_h2h = h2h['home_wins'] + h2h['draws'] + h2h['away_wins']
        if total_h2h == 0:
            st.info(f"No se encontraron partidos directos entre **{home_team}** y **{away_team}**.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{home_team}", h2h['home_wins'])
            c2.metric("Empates",     h2h['draws'])
            c3.metric(f"{away_team}", h2h['away_wins'])

            left, right = st.columns(2)
            with left:
                fig_donut = go.Figure(go.Pie(
                    labels=[home_team, "Draw", away_team],
                    values=[h2h['home_wins'], h2h['draws'], h2h['away_wins']],
                    hole=0.62,
                    marker=dict(colors=['#00d4aa', '#f59e0b', '#f43f5e'],
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
                st.plotly_chart(fig_donut, width='stretch', theme=None)

            with right:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                gc1, gc2, gc3 = st.columns(3)
                gc1.metric(f"{home_team}", int(h2h['home_goals_total']))
                gc2.metric("Goles / Partido", h2h['avg_goals_per_game'])
                gc3.metric(f"{away_team}", int(h2h['away_goals_total']))

                if h2h['last_matches']:
                    chron = list(reversed(h2h['last_matches']))
                    dates = [m['date'] for m in chron]
                    hg = [m['home_score'] if m['home_team'] == home_team else m['away_score'] for m in chron]
                    ag = [m['away_score'] if m['home_team'] == home_team else m['home_score'] for m in chron]
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
                    st.plotly_chart(fig_g, width='stretch', theme=None)

            if h2h['last_matches']:
                st.markdown("### Últimos partidos")
                for m in h2h['last_matches']:
                    ht, at = m['home_team'], m['away_team']
                    hs, as_ = m['home_score'], m['away_score']
                    res = ('W' if (ht == home_team and hs > as_) or (at == home_team and as_ > hs)
                           else 'D' if hs == as_ else 'L')
                    st.markdown(
                        _match_row(m['date'], f"{ht} {hs} — {as_} {at}",
                                   "", m['tournament'], _RC[res]),
                        unsafe_allow_html=True,
                    )

        # ── Separador ──
        st.markdown("<hr style='margin:2rem 0'>", unsafe_allow_html=True)
        st.markdown("### Historial de partidos", unsafe_allow_html=False)

        # ── Historial selector ──
        hc1, hc2 = st.columns([2, 1])
        with hc1:
            sel = st.selectbox("Equipo", [home_team, away_team], key="hist_team")
        with hc2:
            n_opt = st.select_slider("Partidos", options=[10, 20, 50, 100, 'Todos'],
                                     value=20, key="hist_n")

        recent = get_recent_matches(df, sel, n=None if n_opt == 'Todos' else int(n_opt))

        if len(recent) == 0:
            st.info("No se encontraron partidos.")
        else:
            wins_h  = int((recent['result'] == 'W').sum())
            draws_h = int((recent['result'] == 'D').sum())
            loss_h  = int((recent['result'] == 'L').sum())
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            sc1.metric("Victorias",       wins_h)
            sc2.metric("Empates",         draws_h)
            sc3.metric("Derrotas",        loss_h)
            sc4.metric("Goles a favor",   int(recent['goals_for'].sum()))
            sc5.metric("Goles en contra", int(recent['goals_against'].sum()))

            asc = recent.sort_values('date')
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=asc['date'], y=asc['goals_for'], name='Marcados',
                line=dict(color='#00d4aa', width=2.5),
                mode='lines+markers',
                marker=dict(size=7, color='#00d4aa', line=dict(width=2, color='#0a0c10')),
                fill='tozeroy', fillcolor='rgba(0,212,170,0.06)',
            ))
            fig_tl.add_trace(go.Scatter(
                x=asc['date'], y=asc['goals_against'], name='Encajados',
                line=dict(color='#f43f5e', width=2.5),
                mode='lines+markers',
                marker=dict(size=7, color='#f43f5e', line=dict(width=2, color='#0a0c10')),
                fill='tozeroy', fillcolor='rgba(244,63,94,0.06)',
            ))
            fig_tl.update_layout(
                **_LAYOUT, height=290,
                xaxis_title=None, yaxis_title="Goals",
                legend=dict(orientation='h', y=1.12, font=dict(size=12)),
                title=dict(text=f"<b>Tendencia de goles — {sel}</b>", x=0.5,
                           font=dict(size=13, color='#555')),
                yaxis=dict(gridcolor='#1e2130'),
                xaxis=dict(gridcolor='#1e2130'),
            )
            st.plotly_chart(fig_tl, width='stretch', theme=None)

            st.markdown("### Partidos")
            for _, row in recent.iterrows():
                color   = _RC[row['result']]
                res_txt = {'W': 'Victoria', 'D': 'Empate', 'L': 'Derrota'}[row['result']]
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
