# -*- coding: utf-8 -*-
"""
2_Analytics.py
Dashboard de análisis avanzado estilo Wyscout/StatsBomb con look holográfico.
Powered by StatsBomb Open Data · Plotly (sin matplotlib ni mplsoccer).
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics · Tonini Predictor",
    page_icon="📊",
    layout="wide",
)

# ── CSS holográfico ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {
    font-family: 'Rajdhani', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* Fondo más oscuro que app.py */
.stApp { background: #020408 !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1300px; }

/* Títulos holográficos */
h1 {
    font-size: 2rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    margin-bottom: 0 !important;
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #00d4aa !important;
    text-shadow: 0 0 20px rgba(0,212,170,0.5) !important;
    letter-spacing: 0.5px;
}
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #c8cad4 !important; }
p, li { color: #b0b3be !important; }

/* Cards holográficas */
.holo-card {
    background: #060c12;
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 0 20px rgba(0,212,170,0.08), inset 0 0 20px rgba(0,0,0,0.5);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.holo-card:hover {
    border-color: rgba(0,212,170,0.4);
    box-shadow: 0 0 30px rgba(0,212,170,0.15), inset 0 0 20px rgba(0,0,0,0.5);
}

/* Section headers */
.section-label {
    font-size: 0.65rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #00d4aa;
    text-shadow: 0 0 10px rgba(0,212,170,0.6);
    border-bottom: 1px solid rgba(0,212,170,0.15);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #060c12 !important;
    border: 1px solid rgba(0,212,170,0.15) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 0 20px rgba(0,212,170,0.05) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0,212,170,0.35) !important;
    box-shadow: 0 0 20px rgba(0,212,170,0.12) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #444 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #00d4aa !important;
    letter-spacing: -0.5px !important;
}

/* Selectbox */
.stSelectbox label { color: #444 !important; font-size: 0.68rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.6px; }
[data-baseweb="select"] > div:first-child {
    background: #060c12 !important;
    border: 1px solid rgba(0,212,170,0.15) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-weight: 500 !important;
}
[data-baseweb="select"] > div:first-child:hover { border-color: #00d4aa !important; }

/* Plotly chart container */
[data-testid="stPlotlyChart"] {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(0,212,170,0.06);
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #020408; }
::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4aa; }

/* Spinner */
.stSpinner > div { border-top-color: #00d4aa !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #030609 !important; border-right: 1px solid rgba(0,212,170,0.08) !important; }

/* Alerts */
.stAlert { border-radius: 10px !important; }

/* Divider */
hr { border-color: rgba(0,212,170,0.1) !important; margin: 1.2rem 0 !important; }

/* Tabla de rendimiento */
.match-row {
    display: grid;
    grid-template-columns: 90px 1fr 80px 70px 70px 60px 80px;
    align-items: center;
    gap: 8px;
    background: #060c12;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 3px 0;
    border-left: 3px solid #1e2130;
    font-size: 0.82rem;
}

/* Animated border effect */
@keyframes borderGlow {
    0%   { border-color: rgba(0,212,170,0.2); }
    50%  { border-color: rgba(0,212,170,0.5); }
    100% { border-color: rgba(0,212,170,0.2); }
}
.pitch-container {
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 14px;
    overflow: hidden;
    animation: borderGlow 3s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# ── Importa el loader ───────────────────────────────────────────────────────
try:
    from src.statsbomb_loader import (
        get_matches,
        get_team_shots,
        get_team_match_stats,
        get_team_list,
        COMPETITIONS,
    )
    SB_AVAILABLE = True
except ImportError:
    SB_AVAILABLE = False

# ── Helpers de layout ──────────────────────────────────────────────────────
_BG = '#020408'
_CARD_BG = '#060c12'
_CYAN = '#00d4aa'
_LAYOUT_BASE = dict(
    template='plotly_dark',
    paper_bgcolor=_CARD_BG,
    plot_bgcolor=_CARD_BG,
    font=dict(color='#b0b3be', family='Rajdhani, sans-serif', size=12),
)
_MARGIN_DEFAULT = dict(l=16, r=16, t=44, b=16)

_OUTCOME_COLORS = {
    'Goal':     '#00d4aa',
    'Saved':    '#f59e0b',
    'Blocked':  '#6366f1',
    'Off T':    '#f43f5e',
    'Post':     '#ec4899',
    'Wayward':  '#64748b',
}
_OUTCOME_DEFAULT = '#64748b'

_RESULT_COLORS = {'W': '#00d4aa', 'D': '#f59e0b', 'L': '#f43f5e'}


def _section(title):
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)


# ── Pitch drawing ──────────────────────────────────────────────────────────
def draw_pitch_plotly():
    """Retorna lista de Plotly shapes para un campo StatsBomb (120x80 yards).

    Coordenadas StatsBomb:
        x: 0 (línea de fondo izq) → 120 (línea de fondo der)
        y: 0 (banda superior) → 80 (banda inferior)

    Usamos eje Y invertido para que 0=arriba coincida con la visión
    habitual del campo (banda superior arriba).
    """
    line_kw = dict(
        type='rect',
        line=dict(color='rgba(0,212,170,0.4)', width=1.5),
        fillcolor='rgba(0,0,0,0)',
        layer='below',
    )
    circle_kw = dict(
        type='circle',
        line=dict(color='rgba(0,212,170,0.4)', width=1.5),
        fillcolor='rgba(0,0,0,0)',
        layer='below',
    )

    shapes = []

    # Campo completo
    shapes.append({**line_kw, 'x0': 0, 'y0': 0, 'x1': 120, 'y1': 80})

    # Línea de medio campo
    shapes.append({
        'type': 'line',
        'x0': 60, 'y0': 0, 'x1': 60, 'y1': 80,
        'line': dict(color='rgba(0,212,170,0.4)', width=1.5),
        'layer': 'below',
    })

    # Círculo central (radio 10)
    shapes.append({**circle_kw, 'x0': 50, 'y0': 30, 'x1': 70, 'y1': 50})

    # Punto central
    shapes.append({
        **circle_kw,
        'x0': 59.5, 'y0': 39.5, 'x1': 60.5, 'y1': 40.5,
        'fillcolor': 'rgba(0,212,170,0.4)',
    })

    # Área grande izquierda (penalty area): x 0-18, y 18-62
    shapes.append({**line_kw, 'x0': 0, 'y0': 18, 'x1': 18, 'y1': 62})

    # Área grande derecha: x 102-120, y 18-62
    shapes.append({**line_kw, 'x0': 102, 'y0': 18, 'x1': 120, 'y1': 62})

    # Área chica izquierda: x 0-6, y 30-50
    shapes.append({**line_kw, 'x0': 0, 'y0': 30, 'x1': 6, 'y1': 50})

    # Área chica derecha: x 114-120, y 30-50
    shapes.append({**line_kw, 'x0': 114, 'y0': 30, 'x1': 120, 'y1': 50})

    # Portería izquierda: x -2-0, y 36-44
    shapes.append({**line_kw,
                   'x0': -2, 'y0': 36, 'x1': 0, 'y1': 44,
                   'fillcolor': 'rgba(0,212,170,0.06)'})

    # Portería derecha: x 120-122, y 36-44
    shapes.append({**line_kw,
                   'x0': 120, 'y0': 36, 'x1': 122, 'y1': 44,
                   'fillcolor': 'rgba(0,212,170,0.06)'})

    # Punto de penalti izquierdo (x=12, y=40)
    shapes.append({**circle_kw,
                   'x0': 11.5, 'y0': 39.5, 'x1': 12.5, 'y1': 40.5,
                   'fillcolor': 'rgba(0,212,170,0.4)'})

    # Punto de penalti derecho (x=108, y=40)
    shapes.append({**circle_kw,
                   'x0': 107.5, 'y0': 39.5, 'x1': 108.5, 'y1': 40.5,
                   'fillcolor': 'rgba(0,212,170,0.4)'})

    # Arco del área izquierda (semicírculo): centro (12,40), radio 10
    # Usamos un path SVG para el arco
    import math
    # Arco desde ángulo -53° a +53° respecto al centro (12,40)
    # Solo la parte que queda fuera del área grande (x > 18)
    arc_points_left = []
    for deg in range(-53, 54, 3):
        rad = math.radians(deg)
        px = 12 + 10 * math.cos(rad)
        py = 40 + 10 * math.sin(rad)
        arc_points_left.append((px, py))
    if arc_points_left:
        path_d = 'M ' + ' L '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in arc_points_left)
        shapes.append({
            'type': 'path',
            'path': path_d,
            'line': dict(color='rgba(0,212,170,0.4)', width=1.5),
            'fillcolor': 'rgba(0,0,0,0)',
            'layer': 'below',
        })

    # Arco del área derecha: centro (108,40)
    arc_points_right = []
    for deg in range(127, 234, 3):
        rad = math.radians(deg)
        px = 108 + 10 * math.cos(rad)
        py = 40 + 10 * math.sin(rad)
        arc_points_right.append((px, py))
    if arc_points_right:
        path_d = 'M ' + ' L '.join(f'{p[0]:.2f},{p[1]:.2f}' for p in arc_points_right)
        shapes.append({
            'type': 'path',
            'path': path_d,
            'line': dict(color='rgba(0,212,170,0.4)', width=1.5),
            'fillcolor': 'rgba(0,0,0,0)',
            'layer': 'below',
        })

    return shapes


# ── Funciones cacheadas de Streamlit ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_team_list(competition_key):
    return get_team_list(competition_key)


@st.cache_data(show_spinner=False)
def cached_shots(competition_key, team_name):
    return get_team_shots(competition_key, team_name)


@st.cache_data(show_spinner=False)
def cached_match_stats(competition_key, team_name):
    return get_team_match_stats(competition_key, team_name)


# ══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════

# Header
st.markdown("# 📊 Analytics Dashboard")
st.caption("Análisis avanzado · StatsBomb Open Data · WC 2022 / WC 2018")
st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

if not SB_AVAILABLE:
    st.error("statsbombpy no está instalado. Ejecuta: `pip install statsbombpy`")
    st.stop()

# ── Sidebar / Selectores ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:1rem;font-weight:800;color:#00d4aa;'
        'text-shadow:0 0 12px rgba(0,212,170,0.5);margin-bottom:1rem">⚙️ Filtros</div>',
        unsafe_allow_html=True,
    )

    competition_key = st.selectbox(
        "Competición",
        list(COMPETITIONS.keys()),
        index=0,
    )

    with st.spinner("Cargando equipos..."):
        teams = cached_team_list(competition_key)

    if not teams:
        st.error("No se pudieron cargar los equipos.")
        st.stop()

    default_idx = teams.index('Argentina') if 'Argentina' in teams else 0
    team_name = st.selectbox("Equipo", teams, index=default_idx)

    opponent_options = ['Todos'] + [t for t in teams if t != team_name]
    opponent_filter = st.selectbox("Rival (H2H)", opponent_options, index=0)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.65rem;color:#333;text-align:center;line-height:1.6">'
        'StatsBomb Open Data<br>Uso libre para investigación<br>'
        '<span style="color:#00d4aa">statsbombpy</span></div>',
        unsafe_allow_html=True,
    )

# ── Carga de datos ─────────────────────────────────────────────────────────
with st.spinner("Cargando datos StatsBomb..."):
    shots_df = cached_shots(competition_key, team_name)
    match_stats_df = cached_match_stats(competition_key, team_name)

# Aplica filtro de rival si corresponde
if opponent_filter != 'Todos':
    shots_df_filtered = shots_df[shots_df['opponent'] == opponent_filter] if not shots_df.empty else shots_df
    match_stats_filtered = match_stats_df[match_stats_df['opponent'] == opponent_filter] if not match_stats_df.empty else match_stats_df
else:
    shots_df_filtered = shots_df
    match_stats_filtered = match_stats_df

if match_stats_df.empty and shots_df.empty:
    st.info(f"No se encontraron eventos para **{team_name}** en {competition_key}.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 — RESUMEN DEL TORNEO
# ══════════════════════════════════════════════════════════════════════════
_section(f"RESUMEN DEL TORNEO — {competition_key}")

if not match_stats_filtered.empty:
    ms = match_stats_filtered
    total_matches = len(ms)
    total_goals = int(ms['goals_for'].sum())
    total_xg = float(ms['xg_for'].sum())
    xg_per_game = total_xg / total_matches if total_matches else 0.0
    total_shots = int(ms['shots'].sum())
    total_sot = int(ms['shots_on_target'].sum())
    conversion_rate = (total_goals / total_shots * 100) if total_shots else 0.0
    xg_per_shot = (total_xg / total_shots) if total_shots else 0.0

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Partidos", total_matches)
    r1c2.metric("Goles", total_goals)
    r1c3.metric("xG Total", f"{total_xg:.2f}")
    r1c4.metric("xG / Partido", f"{xg_per_game:.2f}")

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Tiros Totales", total_shots)
    r2c2.metric("Tiros a Portería", total_sot)
    r2c3.metric("Conversión", f"{conversion_rate:.1f}%")
    r2c4.metric("xG / Tiro", f"{xg_per_shot:.3f}")
else:
    st.info("No hay estadísticas de partido disponibles.")

# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 2 — SHOT MAP HOLOGRÁFICO
# ══════════════════════════════════════════════════════════════════════════
_section(f"MAPA DE TIROS — {team_name.upper()}")

if shots_df_filtered.empty:
    st.info(f"No se encontraron tiros de {team_name}" +
            (f" contra {opponent_filter}" if opponent_filter != 'Todos' else "") + ".")
else:
    df_shots = shots_df_filtered.copy()

    # StatsBomb y=0 es arriba → invertimos para que se vea natural
    # (banda superior = y=0 en pantalla)
    # No invertimos: mantenemos eje-y tal como StatsBomb (0 arriba, 80 abajo)
    # El campo se dibuja con y: 0 arriba, 80 abajo

    # Agrupa por outcome para trazas separadas
    outcomes_in_data = df_shots['outcome'].unique()

    fig_pitch = go.Figure()

    pitch_shapes = draw_pitch_plotly()

    # Fondo del campo
    fig_pitch.add_shape(
        type='rect',
        x0=-3, y0=-2, x1=123, y1=82,
        fillcolor='#020408',
        line=dict(width=0),
        layer='below',
    )

    # Agrega trazas por outcome
    for outcome in ['Goal', 'Saved', 'Blocked', 'Off T', 'Post', 'Wayward']:
        sub = df_shots[df_shots['outcome'] == outcome]
        if sub.empty:
            continue

        color = _OUTCOME_COLORS.get(outcome, _OUTCOME_DEFAULT)
        is_goal = (outcome == 'Goal')

        # Tamaño proporcional a xG
        sizes = []
        for xg_val in sub['xg']:
            try:
                s = float(xg_val or 0.05)
            except (ValueError, TypeError):
                s = 0.05
            sizes.append(max(8, min(25, 8 + s * 130)))

        hover_texts = []
        for _, row in sub.iterrows():
            xg_v = row.get('xg')
            try:
                xg_str = f"{float(xg_v):.3f}" if xg_v is not None else 'N/A'
            except (ValueError, TypeError):
                xg_str = 'N/A'
            hover_texts.append(
                f"<b>{row.get('player', '?')}</b><br>"
                f"Min: {row.get('minute', '?')}<br>"
                f"xG: {xg_str}<br>"
                f"Resultado: {outcome}<br>"
                f"Rival: {row.get('opponent', '?')}"
            )

        fig_pitch.add_trace(go.Scatter(
            x=sub['x'],
            y=sub['y'],
            mode='markers',
            name=outcome,
            marker=dict(
                symbol='star' if is_goal else 'circle',
                size=sizes,
                color=color,
                opacity=0.88 if is_goal else 0.72,
                line=dict(color=color, width=3 if is_goal else 1.5),
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
        ))

    _pitch_layout = {
        **_LAYOUT_BASE,
        'paper_bgcolor': '#020408',
        'plot_bgcolor':  '#020408',
        'margin':        dict(l=0, r=0, t=50, b=40),
    }
    fig_pitch.update_layout(
        **_pitch_layout,
        shapes=pitch_shapes,
        xaxis=dict(
            range=[-4, 124],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor='y',
            scaleratio=1,
        ),
        yaxis=dict(
            range=[82, -2],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        height=500,
        legend=dict(
            orientation='h',
            y=-0.05,
            x=0.5,
            xanchor='center',
            font=dict(size=11, color='#888'),
            bgcolor='rgba(0,0,0,0)',
        ),
        title=dict(
            text=f'<b>MAPA DE TIROS — {team_name.upper()}</b>'
                 + (f' vs {opponent_filter.upper()}' if opponent_filter != 'Todos' else ''),
            x=0.5,
            font=dict(size=14, color='#00d4aa'),
        ),
    )

    st.markdown('<div class="pitch-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_pitch, width='stretch', theme=None)
    st.markdown('</div>', unsafe_allow_html=True)

    # Leyenda de colores
    legend_items = [
        ('Gol', '#00d4aa'), ('Parado', '#f59e0b'), ('Bloqueado', '#6366f1'),
        ('Fuera', '#f43f5e'), ('Poste', '#ec4899'), ('Perdido', '#64748b'),
    ]
    legend_html = '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:4px;margin-bottom:8px">'
    for label, color in legend_items:
        legend_html += (
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{color};'
            f'box-shadow:0 0 6px {color}"></div>'
            f'<span style="font-size:0.75rem;color:#666;font-weight:600">{label}</span>'
            f'</div>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 — DOS COLUMNAS: xG Acumulado + Distribución de zonas
# ══════════════════════════════════════════════════════════════════════════
_section("ANÁLISIS DE xG Y DISTRIBUCIÓN DE TIROS")

col_left, col_right = st.columns([3, 2])

# ── Col izquierda: xG acumulado por partido ──────────────────────────────
with col_left:
    st.markdown(
        '<div style="font-size:0.75rem;font-weight:700;color:#555;'
        'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px">'
        'xG por Partido</div>',
        unsafe_allow_html=True,
    )
    if not match_stats_filtered.empty and len(match_stats_filtered) >= 1:
        ms = match_stats_filtered.reset_index(drop=True)
        x_vals = list(range(1, len(ms) + 1))
        hover_for = [f"Partido {i+1}<br>vs {row['opponent']}<br>xG: {row['xg_for']:.2f}"
                     for i, (_, row) in enumerate(ms.iterrows())]
        hover_ag = [f"Partido {i+1}<br>vs {row['opponent']}<br>xG en contra: {row['xg_against']:.2f}"
                    for i, (_, row) in enumerate(ms.iterrows())]

        fig_xg = go.Figure()
        fig_xg.add_trace(go.Scatter(
            x=x_vals,
            y=ms['xg_for'].tolist(),
            name='xG a favor',
            mode='lines+markers',
            line=dict(color='#00d4aa', width=2.5),
            marker=dict(size=7, color='#00d4aa',
                        line=dict(width=2, color='#020408')),
            fill='tozeroy',
            fillcolor='rgba(0,212,170,0.08)',
            text=hover_for,
            hovertemplate='%{text}<extra></extra>',
        ))
        fig_xg.add_trace(go.Scatter(
            x=x_vals,
            y=ms['xg_against'].tolist(),
            name='xG en contra',
            mode='lines+markers',
            line=dict(color='#f43f5e', width=2.5),
            marker=dict(size=7, color='#f43f5e',
                        line=dict(width=2, color='#020408')),
            fill='tozeroy',
            fillcolor='rgba(244,63,94,0.06)',
            text=hover_ag,
            hovertemplate='%{text}<extra></extra>',
        ))
        fig_xg.update_layout(
            **_LAYOUT_BASE,
            height=280,
            margin=_MARGIN_DEFAULT,
            xaxis=dict(title='Partido', gridcolor='rgba(0,212,170,0.05)',
                       tickmode='linear', dtick=1, tickfont=dict(size=10)),
            yaxis=dict(title='xG', gridcolor='rgba(0,212,170,0.05)',
                       zeroline=False),
            legend=dict(orientation='h', y=1.12, font=dict(size=11)),
            hovermode='x unified',
        )
        st.plotly_chart(fig_xg, width='stretch', theme=None)
    else:
        st.info("No hay datos de partidos suficientes.")

# ── Col derecha: Distribución de tiros por zona ───────────────────────────
with col_right:
    st.markdown(
        '<div style="font-size:0.75rem;font-weight:700;color:#555;'
        'text-transform:uppercase;letter-spacing:0.6px;margin-bottom:8px">'
        'Zonas de Tiro</div>',
        unsafe_allow_html=True,
    )
    if not shots_df_filtered.empty:
        # Zona según distancia a portería contraria (x > 102 es área contraria)
        # StatsBomb: el equipo ataca hacia x=120 (portería derecha)
        # Área grande: x >= 102
        # Media distancia: 84 <= x < 102  (aprox 18-36 m)
        # Larga distancia: x < 84

        def _zona(x_val):
            try:
                x = float(x_val)
            except (TypeError, ValueError):
                return 'Media distancia'
            dist_from_goal = 120 - x  # distancia a la portería contraria (x=120)
            if dist_from_goal <= 18:
                return 'Área'
            elif dist_from_goal <= 30:
                return 'Media distancia'
            else:
                return 'Larga distancia'

        zona_counts = shots_df_filtered['x'].apply(_zona).value_counts()
        zona_order = ['Área', 'Media distancia', 'Larga distancia']
        labels_z = [z for z in zona_order if z in zona_counts.index]
        values_z = [int(zona_counts.get(z, 0)) for z in labels_z]
        colors_z = ['#00d4aa', '#f59e0b', '#6366f1'][:len(labels_z)]

        fig_donut = go.Figure(go.Pie(
            labels=labels_z,
            values=values_z,
            hole=0.58,
            marker=dict(
                colors=colors_z,
                line=dict(color='#020408', width=3),
            ),
            textinfo='label+percent',
            textfont=dict(size=11, color='#c8cad4'),
            pull=[0.04] * len(labels_z),
            hovertemplate='<b>%{label}</b><br>%{value} tiros (%{percent})<extra></extra>',
        ))
        fig_donut.update_layout(
            **_LAYOUT_BASE,
            height=280,
            showlegend=False,
            margin=dict(l=8, r=8, t=20, b=8),
        )
        st.plotly_chart(fig_donut, width='stretch', theme=None)
    else:
        st.info("Sin datos de tiros para mostrar zonas.")

# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 — RENDIMIENTO POR PARTIDO (tabla estilo FotMob)
# ══════════════════════════════════════════════════════════════════════════
_section("RENDIMIENTO POR PARTIDO")

if not match_stats_filtered.empty:
    # Header de la tabla
    header_html = (
        '<div style="display:grid;grid-template-columns:90px 1fr 80px 70px 70px 60px 80px;'
        'gap:8px;padding:8px 16px;margin-bottom:4px">'
        '<span style="font-size:0.6rem;font-weight:800;color:#333;text-transform:uppercase;letter-spacing:0.5px">Fecha</span>'
        '<span style="font-size:0.6rem;font-weight:800;color:#333;text-transform:uppercase;letter-spacing:0.5px">Rival</span>'
        '<span style="font-size:0.6rem;font-weight:800;color:#333;text-transform:uppercase;letter-spacing:0.5px;text-align:center">Res.</span>'
        '<span style="font-size:0.6rem;font-weight:800;color:#333;text-transform:uppercase;letter-spacing:0.5px;text-align:center">Goles</span>'
        '<span style="font-size:0.6rem;font-weight:800;color:#333;text-transform:uppercase;letter-spacing:0.5px;text-align:center">xG</span>'
        '<span style="font-size:0.6rem;font-weight:800;color:#333;text-transform:uppercase;letter-spacing:0.5px;text-align:center">Tiros</span>'
        '<span style="font-size:0.6rem;font-weight:800;color:#333;text-transform:uppercase;letter-spacing:0.5px;text-align:center">T.Port.</span>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    for _, row in match_stats_filtered.iterrows():
        result = row.get('result', 'D')
        color = _RESULT_COLORS.get(result, '#555')
        gf = int(row.get('goals_for', 0))
        ga = int(row.get('goals_against', 0))
        xg_v = float(row.get('xg_for', 0.0))
        shots_v = int(row.get('shots', 0))
        sot_v = int(row.get('shots_on_target', 0))
        date_v = str(row.get('date', ''))[:10]
        opp_v = str(row.get('opponent', '?'))
        res_label = {'W': 'V', 'D': 'E', 'L': 'D'}.get(result, result)

        row_html = (
            f'<div style="display:grid;grid-template-columns:90px 1fr 80px 70px 70px 60px 80px;'
            f'align-items:center;gap:8px;background:#060c12;'
            f'border-left:3px solid {color};border-radius:0 8px 8px 0;'
            f'padding:10px 16px;margin:3px 0">'
            f'<span style="color:#444;font-size:0.75rem;font-weight:600">{date_v}</span>'
            f'<span style="color:#c8cad4;font-weight:600;font-size:0.82rem">{opp_v}</span>'
            f'<span style="color:{color};font-weight:800;font-size:0.82rem;text-align:center">'
            f'{gf}–{ga} <span style="font-size:0.68rem">{res_label}</span></span>'
            f'<span style="color:#e8eaf0;font-weight:700;font-size:0.82rem;text-align:center">{gf}</span>'
            f'<span style="color:#b0b3be;font-size:0.82rem;text-align:center">{xg_v:.2f}</span>'
            f'<span style="color:#b0b3be;font-size:0.82rem;text-align:center">{shots_v}</span>'
            f'<span style="color:#b0b3be;font-size:0.82rem;text-align:center">{sot_v}</span>'
            f'</div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)
else:
    st.info("No hay datos de partidos para mostrar.")

# ══════════════════════════════════════════════════════════════════════════
#  SECCIÓN 5 — TOP TIRADORES POR xG
# ══════════════════════════════════════════════════════════════════════════
if not shots_df_filtered.empty and 'player' in shots_df_filtered.columns:
    _section(f"TOP JUGADORES POR xG — {team_name.upper()}")

    try:
        player_xg = (
            shots_df_filtered.groupby('player')['xg']
            .sum()
            .dropna()
            .sort_values(ascending=True)
            .tail(10)
        )

        if not player_xg.empty:
            n = len(player_xg)
            # Gradiente de azul oscuro a cyan
            import colorsys

            def _gradient_color(i, total):
                ratio = i / max(total - 1, 1)
                r = int(0 + ratio * 0)
                g = int(150 + ratio * (212 - 150))
                b = int(180 + ratio * (170 - 180))
                return f'rgb({r},{g},{b})'

            colors_bar = [_gradient_color(i, n) for i in range(n)]

            fig_players = go.Figure(go.Bar(
                x=player_xg.values.tolist(),
                y=player_xg.index.tolist(),
                orientation='h',
                marker=dict(
                    color=colors_bar,
                    line=dict(width=0),
                ),
                text=[f"  {v:.2f} xG" for v in player_xg.values],
                textposition='outside',
                textfont=dict(size=11, color='#888'),
                hovertemplate='<b>%{y}</b><br>xG: %{x:.3f}<extra></extra>',
            ))
            fig_players.update_layout(
                **_LAYOUT_BASE,
                height=max(280, n * 38),
                xaxis=dict(
                    title='xG Total',
                    gridcolor='rgba(0,212,170,0.05)',
                    showticklabels=True,
                ),
                yaxis=dict(
                    tickfont=dict(size=11, color='#c8cad4'),
                ),
                margin=dict(l=10, r=80, t=30, b=20),
            )
            st.plotly_chart(fig_players, width='stretch', theme=None)
    except Exception:
        pass

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:0.7rem;color:#1e2130;padding:1rem">'
    'Datos: StatsBomb Open Data · statsbombpy · Solo para uso no comercial'
    '</div>',
    unsafe_allow_html=True,
)
