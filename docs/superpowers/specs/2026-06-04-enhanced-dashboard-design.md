# Enhanced Dashboard — Design Spec

**Date:** 2026-06-04
**Status:** Approved

---

## Overview

Enhance the existing football match predictor Streamlit app with a rich dark-mode dashboard featuring 4 tabs (Prediction, Stats, Head-to-Head, History), comprehensive team metrics, and deployment to Streamlit Community Cloud for public access.

---

## Goals

- Replace the single-screen app with a professional 4-tab dark dashboard
- Add comprehensive team statistics (attack/defense metrics, form, clean sheets, etc.)
- Add head-to-head historical analysis between selected teams
- Add per-team match history with timeline charts
- Deploy publicly on Streamlit Community Cloud

---

## Architecture

Two new files are introduced:

```
src/stats.py      — all statistical computations for the UI (separate from ML preprocess.py)
app.py            — rewritten with 4-tab layout, dark theme, Plotly dark charts
.streamlit/
  config.toml     — dark theme configuration
requirements.txt  — already exists, no changes needed
```

`src/stats.py` exposes pure functions that take the raw DataFrame and team names, returning structured data. `app.py` calls them and renders with Streamlit + Plotly. No changes to `src/preprocess.py`, `src/train.py`, or `src/predict.py`.

---

## Dark Theme

File: `.streamlit/config.toml`

```toml
[theme]
base = "dark"
primaryColor = "#00d4aa"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1d23"
textColor = "#fafafa"
font = "sans serif"
```

All Plotly charts use `template="plotly_dark"` with `paper_bgcolor="#1a1d23"` and `plot_bgcolor="#1a1d23"`.

---

## New Module: src/stats.py

Functions exposed:

### `get_team_overall_stats(df, team, n_recent=10) -> dict`
Returns overall stats for a team across all their matches (home + away):
- `total_matches`: int
- `wins`, `draws`, `losses`: int
- `win_rate`, `draw_rate`, `loss_rate`: float (0–1)
- `goals_scored_per_game`: float
- `goals_conceded_per_game`: float
- `clean_sheet_rate`: float — % matches where team conceded 0
- `scoring_rate`: float — % matches where team scored at least 1
- `form`: list[str] — last `n_recent` results as 'W'/'D'/'L', chronological order

### `get_radar_stats(df, home_team, away_team) -> dict`
Returns normalized (0–1) scores for 6 radar dimensions for each team:
- `attack`: avg goals scored (normalized against dataset max)
- `defense`: inverse avg goals conceded (normalized)
- `form`: win rate in last 10 matches
- `consistency`: 1 - std deviation of results (lower variance = more consistent)
- `effectiveness`: scoring_rate (% games where scored)
- `solidity`: clean_sheet_rate

Returns `{'home': {dim: value}, 'away': {dim: value}}`.

### `get_head_to_head(df, home_team, away_team, last_n=5) -> dict`
Returns H2H stats:
- `home_wins`, `draws`, `away_wins`: int (all-time)
- `home_goals_total`, `away_goals_total`: int (all-time)
- `avg_goals_per_game`: float
- `last_matches`: list[dict] — last `last_n` encounters, each with `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`

### `get_recent_matches(df, team, n=10) -> DataFrame`
Returns last `n` matches for a team (home or away), sorted by date descending.
Columns: `date`, `opponent`, `home_away`, `goals_for`, `goals_against`, `result` ('W'/'D'/'L'), `tournament`.

---

## App Layout: app.py

### Always-visible header
- Title + subtitle
- Two-column selectboxes (Equipo Local / Equipo Visitante)
- Primary "Predecir" button
- On click: auto-train model if `model.pkl` missing (with `st.spinner`)

### Tab 1 — 🎯 Predicción

1. Result headline — large colored text (green=H, yellow=D, red=A) + confidence %
2. Three big metric columns: Home Win % | Draw % | Away Win %
3. Horizontal probability bar chart (existing, dark-themed)
4. Mini comparison panel (2 columns, one per team):
   - Goals scored per game ⚽
   - Goals conceded per game 🛡️
   - Win % / Draw % summary
   - Form last 5: colored badges [W][D][L] (green/yellow/red)

### Tab 2 — 📊 Estadísticas

1. Radar chart (Plotly `Scatterpolar`) comparing both teams on 6 dimensions
2. Side-by-side metrics table (3 columns: home value | metric name | away value):
   - Goles anotados/PJ
   - Goles recibidos/PJ
   - % Victorias
   - % Empates
   - % Derrotas
   - % Partidos con gol (scoring rate)
   - % Porterías a cero (clean sheets)
3. Form last 10 matches — colored badge row for each team

### Tab 3 — ⚔️ Head-to-Head

1. All-time record summary: 3 big numbers (Home wins | Draws | Away wins)
2. Donut chart showing H2H win distribution
3. Total goals: "Home X — Away Y" + avg goals per game
4. Table of last 5 encounters (date, score, tournament) — row background color by result
5. Goals per H2H match bar chart (grouped: home goals vs away goals per match)

### Tab 4 — 📋 Historial

1. Team selector (defaults to home team) + match count selector (10/20/all)
2. Results table with colored row backgrounds (green/yellow/red)
3. Goals timeline line chart: goals scored (line) vs goals conceded (line) over time
4. Period summary: X wins, Y draws, Z losses | Goals: A favor / B contra

---

## Automatic Model Training

In `app.py`, if `model.pkl` does not exist:
```python
with st.spinner("Entrenando modelo por primera vez... (~30 segundos)"):
    from src.train import train
    train()
```
This runs once on first deploy. Subsequent page loads skip it (model.pkl already exists).

**Problem:** `model.pkl` is in `.gitignore` and Streamlit Cloud doesn't persist files between deploys. The model must be retrained on every cold start.

**Solution:** Keep training on startup. Add `st.cache_resource` so within a session the model is only loaded once. The ~30 second wait only happens on cold start (first visitor after deploy).

---

## Dataset in Repo

- Remove `data/results.csv` from `.gitignore`
- Commit the Kaggle CSV (~4MB) to the repo
- This is required for Streamlit Community Cloud (no local filesystem)

---

## Deployment: Streamlit Community Cloud

Steps (manual, done after implementation):
1. Push repo to GitHub (public or private)
2. Go to share.streamlit.io → New app → select repo + branch + `app.py`
3. Deploy — public URL generated: `https://<username>-match-predictor-app-xxxxx.streamlit.app`

No secrets or env vars needed — app uses only the bundled CSV and trains its own model.

---

## File Changes Summary

| File | Change |
|---|---|
| `app.py` | Full rewrite — 4-tab dark layout |
| `src/stats.py` | New — all UI statistics functions |
| `.streamlit/config.toml` | New — dark theme config |
| `.gitignore` | Remove `data/results.csv` entry |
| `data/results.csv` | Now tracked in git |

No changes to: `src/preprocess.py`, `src/train.py`, `src/predict.py`, `requirements.txt`, tests.

---

## Out of Scope

- Live / real-time data
- User accounts or saved preferences
- Multilingual support
- Mobile-specific layout optimization
- Custom team logos/flags (emoji only)
