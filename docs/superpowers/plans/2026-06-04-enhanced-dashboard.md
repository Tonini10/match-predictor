# Enhanced Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-screen Streamlit app with a rich dark-mode 4-tab dashboard featuring comprehensive team stats, head-to-head history, and auto-training for Streamlit Community Cloud deployment.

**Architecture:** New `src/stats.py` module owns all UI statistics (separate from ML preprocess.py). `app.py` is fully rewritten with 4 Streamlit tabs using Plotly dark charts. `.streamlit/config.toml` sets the dark theme globally.

**Tech Stack:** Python, Streamlit, Plotly, pandas, scikit-learn, joblib

---

## File Map

| File | Change |
|---|---|
| `.streamlit/config.toml` | New — dark theme colors |
| `.gitignore` | Remove `data/results.csv` line |
| `src/stats.py` | New — 4 pure stats functions for UI |
| `tests/test_stats.py` | New — 12 TDD tests for stats.py |
| `app.py` | Full rewrite — 4-tab dark dashboard |

No changes to: `src/preprocess.py`, `src/train.py`, `src/predict.py`, `requirements.txt`

---

## Task 1: Dark Theme Config + Gitignore

**Files:**
- Create: `C:\Users\Esteban\match-predictor\.streamlit\config.toml`
- Modify: `C:\Users\Esteban\match-predictor\.gitignore`

- [ ] **Step 1: Create .streamlit directory and config.toml**

```powershell
New-Item -ItemType Directory -Force "C:\Users\Esteban\match-predictor\.streamlit"
```

Create `C:\Users\Esteban\match-predictor\.streamlit\config.toml`:

```toml
[theme]
base = "dark"
primaryColor = "#00d4aa"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1d23"
textColor = "#fafafa"
font = "sans serif"
```

- [ ] **Step 2: Remove data/results.csv from .gitignore**

Read `C:\Users\Esteban\match-predictor\.gitignore` and remove the line `data/results.csv`. The resulting file should be:

```
__pycache__/
.pytest_cache/
*.pyc
*.pyo
.coverage
venv/
.venv/
.vscode/
.idea/
*.pkl
*.joblib
.streamlit/
```

Note: `data/results.csv` line is removed. `.streamlit/` remains (we don't want to commit cache files Streamlit generates inside `.streamlit/`, only `config.toml`).

Wait — `.streamlit/` in .gitignore would ignore `config.toml` too. Remove `.streamlit/` from .gitignore and instead add `.streamlit/*.cache` to be safe. Actually, the simplest fix: remove `.streamlit/` from .gitignore entirely. Streamlit doesn't generate sensitive cache files there — it only writes `config.toml` and optionally `secrets.toml`. The resulting `.gitignore`:

```
__pycache__/
.pytest_cache/
*.pyc
*.pyo
.coverage
venv/
.venv/
.vscode/
.idea/
*.pkl
*.joblib
```

- [ ] **Step 3: Commit**

```powershell
cd C:\Users\Esteban\match-predictor
git add .streamlit/config.toml .gitignore
git commit -m "feat: dark theme config and untrack results.csv from gitignore"
```

---

## Task 2: Statistics Module (src/stats.py) — TDD

**Files:**
- Create: `C:\Users\Esteban\match-predictor\tests\test_stats.py`
- Create: `C:\Users\Esteban\match-predictor\src\stats.py`

- [ ] **Step 1: Write the failing tests**

Create `C:\Users\Esteban\match-predictor\tests\test_stats.py`:

```python
import pandas as pd
import pytest
from src.stats import (
    get_team_overall_stats,
    get_radar_stats,
    get_head_to_head,
    get_recent_matches,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'date': [
            '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
            '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
        ],
        'home_team': ['Brazil', 'Brazil', 'Argentina', 'Brazil', 'Argentina', 'Brazil', 'Mexico', 'Argentina'],
        'away_team': ['Germany', 'Argentina', 'Germany', 'Mexico', 'Brazil', 'Argentina', 'Brazil', 'Mexico'],
        'home_score': [3, 2, 1, 2, 0, 1, 1, 2],
        'away_score': [1, 1, 0, 0, 2, 1, 2, 0],
        'neutral': [False] * 8,
        'tournament': ['Friendly'] * 8,
    })


# ── get_team_overall_stats ────────────────────────────────────────────────

def test_overall_stats_counts_home_and_away_matches(sample_df):
    # Brazil: home rows 0,1,3,5 + away rows 4,6 = 6 total
    stats = get_team_overall_stats(sample_df, 'Brazil')
    assert stats['total_matches'] == 6


def test_overall_stats_win_draw_loss_counts(sample_df):
    # Brazil: W(row0), W(row1), W(row3), W(row4 away), D(row5), W(row6 away) = 5W 1D 0L
    stats = get_team_overall_stats(sample_df, 'Brazil')
    assert stats['wins'] == 5
    assert stats['draws'] == 1
    assert stats['losses'] == 0


def test_overall_stats_goals_scored_per_game(sample_df):
    # Brazil goals_for: 3+2+2+2+1+2 = 12 across 6 matches → 2.0
    stats = get_team_overall_stats(sample_df, 'Brazil')
    assert stats['goals_scored_per_game'] == pytest.approx(2.0, rel=0.01)


def test_overall_stats_form_length_respects_n_recent(sample_df):
    stats = get_team_overall_stats(sample_df, 'Brazil', n_recent=3)
    assert len(stats['form']) == 3


def test_overall_stats_unknown_team_returns_zero_matches(sample_df):
    stats = get_team_overall_stats(sample_df, 'Unknown')
    assert stats['total_matches'] == 0
    assert stats['win_rate'] == 0.0
    assert stats['form'] == []


# ── get_radar_stats ───────────────────────────────────────────────────────

def test_radar_stats_returns_six_dimensions(sample_df):
    radar = get_radar_stats(sample_df, 'Brazil', 'Argentina')
    assert len(radar['dimensions']) == 6
    assert len(radar['home']) == 6
    assert len(radar['away']) == 6


def test_radar_stats_all_values_between_0_and_1(sample_df):
    radar = get_radar_stats(sample_df, 'Brazil', 'Argentina')
    for v in radar['home'] + radar['away']:
        assert 0.0 <= v <= 1.0, f"Value {v} out of range [0,1]"


# ── get_head_to_head ──────────────────────────────────────────────────────

def test_h2h_total_encounters(sample_df):
    # Brazil vs Argentina: row1 (Brazil H 2-1), row4 (Argentina H 0-2), row5 (Brazil H 1-1) = 3
    h2h = get_head_to_head(sample_df, 'Brazil', 'Argentina')
    assert h2h['home_wins'] + h2h['draws'] + h2h['away_wins'] == 3


def test_h2h_correct_win_counts(sample_df):
    # Brazil wins: row1 (2>1), row4 (away: 2>0) = 2; Draws: row5 (1==1) = 1; Argentina wins: 0
    h2h = get_head_to_head(sample_df, 'Brazil', 'Argentina')
    assert h2h['home_wins'] == 2
    assert h2h['draws'] == 1
    assert h2h['away_wins'] == 0


def test_h2h_no_encounters_returns_empty(sample_df):
    h2h = get_head_to_head(sample_df, 'Germany', 'Mexico')
    assert h2h['home_wins'] == 0
    assert h2h['last_matches'] == []


# ── get_recent_matches ────────────────────────────────────────────────────

def test_recent_matches_returns_dataframe_with_required_columns(sample_df):
    result = get_recent_matches(sample_df, 'Brazil')
    assert isinstance(result, pd.DataFrame)
    for col in ['date', 'opponent', 'home_away', 'goals_for', 'goals_against', 'result', 'tournament']:
        assert col in result.columns


def test_recent_matches_result_values_are_valid(sample_df):
    result = get_recent_matches(sample_df, 'Brazil')
    assert set(result['result'].unique()).issubset({'W', 'D', 'L'})
    assert (result['result'] == 'W').sum() == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
cd C:\Users\Esteban\match-predictor
python -m pytest tests/test_stats.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.stats'`

- [ ] **Step 3: Implement src/stats.py**

Create `C:\Users\Esteban\match-predictor\src\stats.py`:

```python
import pandas as pd


def _all_matches_for_team(df, team):
    """Return unified DataFrame of all matches for a team with goals_for/against and result."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    home = df[df['home_team'] == team].copy()
    home['goals_for'] = home['home_score']
    home['goals_against'] = home['away_score']
    home['result'] = home.apply(
        lambda r: 'W' if r['home_score'] > r['away_score']
        else ('D' if r['home_score'] == r['away_score'] else 'L'), axis=1
    )

    away = df[df['away_team'] == team].copy()
    away['goals_for'] = away['away_score']
    away['goals_against'] = away['home_score']
    away['result'] = away.apply(
        lambda r: 'W' if r['away_score'] > r['home_score']
        else ('D' if r['away_score'] == r['home_score'] else 'L'), axis=1
    )

    combined = pd.concat([home, away]).sort_values('date').reset_index(drop=True)
    return combined


def get_team_overall_stats(df, team, n_recent=10):
    all_m = _all_matches_for_team(df, team)

    if len(all_m) == 0:
        return {
            'total_matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'win_rate': 0.0, 'draw_rate': 0.0, 'loss_rate': 0.0,
            'goals_scored_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'clean_sheet_rate': 0.0, 'scoring_rate': 0.0, 'form': [],
        }

    total = len(all_m)
    wins = int((all_m['result'] == 'W').sum())
    draws = int((all_m['result'] == 'D').sum())
    losses = int((all_m['result'] == 'L').sum())
    recent = all_m.tail(n_recent)

    return {
        'total_matches': total,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'win_rate': round(wins / total, 3),
        'draw_rate': round(draws / total, 3),
        'loss_rate': round(losses / total, 3),
        'goals_scored_per_game': round(float(all_m['goals_for'].mean()), 2),
        'goals_conceded_per_game': round(float(all_m['goals_against'].mean()), 2),
        'clean_sheet_rate': round(float((all_m['goals_against'] == 0).mean()), 3),
        'scoring_rate': round(float((all_m['goals_for'] > 0).mean()), 3),
        'form': recent['result'].tolist(),
    }


def get_radar_stats(df, home_team, away_team):
    hs = get_team_overall_stats(df, home_team, n_recent=10)
    as_ = get_team_overall_stats(df, away_team, n_recent=10)

    def norm_higher(h, a):
        denom = max(h, a, 1e-9)
        return round(h / denom, 3), round(a / denom, 3)

    def norm_lower(h, a):
        # lower is better (goals conceded): invert
        max_v = max(h, a, 1e-9)
        return round(1 - h / max_v, 3), round(1 - a / max_v, 3)

    home_form = sum(1 for r in hs['form'] if r == 'W') / max(len(hs['form']), 1)
    away_form = sum(1 for r in as_['form'] if r == 'W') / max(len(as_['form']), 1)

    att_h, att_a = norm_higher(hs['goals_scored_per_game'], as_['goals_scored_per_game'])
    def_h, def_a = norm_lower(hs['goals_conceded_per_game'], as_['goals_conceded_per_game'])
    frm_h, frm_a = norm_higher(home_form, away_form)
    con_h, con_a = norm_higher(1 - hs['loss_rate'], 1 - as_['loss_rate'])
    eff_h, eff_a = norm_higher(hs['scoring_rate'], as_['scoring_rate'])
    sol_h, sol_a = norm_higher(hs['clean_sheet_rate'], as_['clean_sheet_rate'])

    dimensions = ['Ataque', 'Defensa', 'Forma', 'Consistencia', 'Efectividad', 'Solidez']
    return {
        'dimensions': dimensions,
        'home': [att_h, def_h, frm_h, con_h, eff_h, sol_h],
        'away': [att_a, def_a, frm_a, con_a, eff_a, sol_a],
    }


def get_head_to_head(df, home_team, away_team, last_n=5):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    h2h = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
    ].copy()

    if len(h2h) == 0:
        return {
            'home_wins': 0, 'draws': 0, 'away_wins': 0,
            'home_goals_total': 0, 'away_goals_total': 0,
            'avg_goals_per_game': 0.0, 'last_matches': [],
        }

    home_wins = draws = away_wins = 0
    home_goals = away_goals = 0

    for _, row in h2h.iterrows():
        if row['home_team'] == home_team:
            home_goals += row['home_score']
            away_goals += row['away_score']
            if row['home_score'] > row['away_score']:
                home_wins += 1
            elif row['home_score'] == row['away_score']:
                draws += 1
            else:
                away_wins += 1
        else:
            home_goals += row['away_score']
            away_goals += row['home_score']
            if row['away_score'] > row['home_score']:
                home_wins += 1
            elif row['away_score'] == row['home_score']:
                draws += 1
            else:
                away_wins += 1

    total_goals = home_goals + away_goals
    last_matches = []
    for _, row in h2h.tail(last_n).iterrows():
        last_matches.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': int(row['home_score']),
            'away_score': int(row['away_score']),
            'tournament': str(row.get('tournament', 'N/A')),
        })
    last_matches.reverse()  # most recent first

    return {
        'home_wins': home_wins,
        'draws': draws,
        'away_wins': away_wins,
        'home_goals_total': home_goals,
        'away_goals_total': away_goals,
        'avg_goals_per_game': round(total_goals / len(h2h), 2),
        'last_matches': last_matches,
    }


def get_recent_matches(df, team, n=10):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    home = df[df['home_team'] == team].copy()
    home['opponent'] = home['away_team']
    home['home_away'] = 'Local'
    home['goals_for'] = home['home_score']
    home['goals_against'] = home['away_score']
    home['result'] = home.apply(
        lambda r: 'W' if r['home_score'] > r['away_score']
        else ('D' if r['home_score'] == r['away_score'] else 'L'), axis=1
    )

    away = df[df['away_team'] == team].copy()
    away['opponent'] = away['home_team']
    away['home_away'] = 'Visitante'
    away['goals_for'] = away['away_score']
    away['goals_against'] = away['home_score']
    away['result'] = away.apply(
        lambda r: 'W' if r['away_score'] > r['home_score']
        else ('D' if r['away_score'] == r['home_score'] else 'L'), axis=1
    )

    combined = pd.concat([home, away]).sort_values('date')
    cols = ['date', 'opponent', 'home_away', 'goals_for', 'goals_against', 'result', 'tournament']
    if n is None:
        recent = combined[cols].copy()
    else:
        recent = combined[cols].tail(n).copy()
    return recent.sort_values('date', ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
cd C:\Users\Esteban\match-predictor
python -m pytest tests/test_stats.py -v
```

Expected: `12 passed`

- [ ] **Step 5: Run all tests to verify no regressions**

```powershell
python -m pytest -v
```

Expected: `34 passed` (22 existing + 12 new)

- [ ] **Step 6: Commit**

```powershell
cd C:\Users\Esteban\match-predictor
git add src/stats.py tests/test_stats.py
git commit -m "feat: statistics module with team stats, radar, h2h, and history"
```

---

## Task 3: App Rewrite (app.py) — 4-Tab Dark Dashboard

**Files:**
- Modify: `C:\Users\Esteban\match-predictor\app.py` (full rewrite)

- [ ] **Step 1: Read the current app.py before overwriting**

Read `C:\Users\Esteban\match-predictor\app.py` to confirm current state.

- [ ] **Step 2: Write the new app.py**

Overwrite `C:\Users\Esteban\match-predictor\app.py` with:

```python
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
```

- [ ] **Step 3: Verify all unit tests still pass**

```powershell
cd C:\Users\Esteban\match-predictor
python -m pytest -v
```

Expected: `34 passed`

- [ ] **Step 4: Commit**

```powershell
git add app.py
git commit -m "feat: 4-tab dark dashboard with stats, radar, h2h, and history"
```

---

## Task 4: Commit Dataset for Cloud Deploy

**Files:**
- Track: `C:\Users\Esteban\match-predictor\data\results.csv`

- [ ] **Step 1: Verify results.csv exists in data/ folder**

```powershell
Test-Path "C:\Users\Esteban\match-predictor\data\results.csv"
```

Expected: `True`

If `False`: the Kaggle CSV hasn't been placed yet. Stop here and ask the user to download `results.csv` from `https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017` and place it at `data/results.csv`.

- [ ] **Step 2: Add results.csv to git**

```powershell
cd C:\Users\Esteban\match-predictor
git add data/results.csv
git commit -m "data: add international football results dataset for cloud deploy"
```

Expected: commit with `1 file changed`, `44000+` insertions.

- [ ] **Step 3: Verify final git log**

```powershell
git log --oneline
```

Expected: 5 commits including the dataset commit.
