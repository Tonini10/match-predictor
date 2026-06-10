# Jugadores, Rendimiento y Recomendación de Apuesta — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activar features de plantilla FIFA, añadir 10 features de rendimiento reciente (tiros, tiros al arco, córners, amarillas, rojas) y una recomendación de apuesta con detección de value bets en la app Streamlit.

**Architecture:** Incremental sobre el pipeline existente: `ingest.py` conserva columnas de stats de los CSVs crudos, `preprocess.py` añade rolling means, `player_features.py` expone la plantilla top-N, nuevo módulo `betting.py` con la lógica de recomendación/EV, y `app.py` muestra todo tras la predicción. El modelo se reentrena y se compara la precisión contra la línea base.

**Tech Stack:** Python, pandas, XGBoost, scikit-learn, Streamlit, Plotly, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-jugadores-rendimiento-apuestas-design.md`

**Convenciones del repo:** tests con pytest y fixtures de DataFrames sintéticos (ver `tests/test_preprocess.py`). Ejecutar siempre desde la raíz del repo con `python -m pytest`. Los CSVs crudos ya están cacheados en `data/raw/` — `python -m src.ingest` NO descarga nada si los archivos existen.

---

### Task 0: Línea base de precisión

Antes de tocar código, registrar la precisión actual para comparar al final.

- [ ] **Step 0.1: Entrenar con el código actual y anotar la precisión**

Run: `python -m src.train`
Expected: imprime `Test accuracy: 0.XXX`. **Anota ese número** — se usa en Task 6.

---

### Task 1: `ingest.py` — conservar stats de partido

**Files:**
- Modify: `src/ingest.py` (función `normalize_csv`, líneas 37-73)
- Test: `tests/test_ingest.py`

- [ ] **Step 1.1: Escribir tests que fallan**

Añadir al final de `tests/test_ingest.py`:

```python
@pytest.fixture
def raw_df_with_stats():
    return pd.DataFrame({
        'Date': ['16/08/2024', '17/08/2024'],
        'HomeTeam': ['Man United', 'Arsenal'],
        'AwayTeam': ['Fulham', 'Wolves'],
        'FTHG': [1, 2], 'FTAG': [0, 0], 'FTR': ['H', 'H'],
        'HS': [14, 18], 'AS': [10, 6],
        'HST': [5, 9], 'AST': [2, 1],
        'HC': [7, 8], 'AC': [8, 2],
        'HY': [2, 1], 'AY': [3, 2],
        'HR': [0, 0], 'AR': [0, 1],
    })


def test_normalize_csv_keeps_match_stats(raw_df_with_stats):
    result = normalize_csv(raw_df_with_stats, 'Premier League')
    assert result.iloc[0]['home_shots'] == 14
    assert result.iloc[0]['away_shots'] == 10
    assert result.iloc[0]['home_shots_on_target'] == 5
    assert result.iloc[0]['home_corners'] == 7
    assert result.iloc[0]['home_yellow'] == 2
    assert result.iloc[1]['away_red'] == 1


def test_normalize_csv_stats_nan_when_missing():
    raw = pd.DataFrame({
        'Date': ['16/08/2024'],
        'HomeTeam': ['Man United'], 'AwayTeam': ['Fulham'],
        'FTHG': [1], 'FTAG': [0], 'FTR': ['H'],
    })
    result = normalize_csv(raw, 'Premier League')
    assert 'home_shots' in result.columns
    assert pd.isna(result.iloc[0]['home_shots'])
    assert pd.isna(result.iloc[0]['away_red'])
```

- [ ] **Step 1.2: Verificar que fallan**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: los 2 tests nuevos FAIL con `KeyError: 'home_shots'`; el resto PASS.

- [ ] **Step 1.3: Implementar en `src/ingest.py`**

Añadir constante a nivel de módulo (después de `BASE_URL`):

```python
STAT_COL_MAP = {
    'HS': 'home_shots', 'AS': 'away_shots',
    'HST': 'home_shots_on_target', 'AST': 'away_shots_on_target',
    'HC': 'home_corners', 'AC': 'away_corners',
    'HY': 'home_yellow', 'AY': 'away_yellow',
    'HR': 'home_red', 'AR': 'away_red',
}
```

En `normalize_csv`, cambiar el rename inicial:

```python
df = raw_df.rename(columns={**col_map, **STAT_COL_MAP}).copy()
```

y cambiar la selección de columnas `keep`:

```python
stat_cols = list(STAT_COL_MAP.values())
keep = required + (['result'] if 'result' in df.columns else []) \
       + [c for c in stat_cols if c in df.columns]
df = df[keep].copy()
```

Después de las conversiones de score (`df['away_score'] = ...`), añadir:

```python
for c in stat_cols:
    if c not in df.columns:
        df[c] = pd.NA
    df[c] = pd.to_numeric(df[c], errors='coerce')
```

`combine_datasets` no cambia: al concatenar con `results.csv` (sin esas columnas), pandas rellena con `NaN` automáticamente.

- [ ] **Step 1.4: Verificar que pasan**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: todos PASS.

- [ ] **Step 1.5: Suite completa y commit**

Run: `python -m pytest -q`
Expected: todos PASS (los 79 + 2 nuevos).

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: keep shots, corners and card stats when ingesting raw CSVs"
```

---

### Task 2: `preprocess.py` — 10 features de rendimiento reciente

**Files:**
- Modify: `src/preprocess.py` (FEATURE_COLS, `build_training_data`, `build_features_for_prediction`)
- Test: `tests/test_preprocess.py`

- [ ] **Step 2.1: Escribir tests que fallan**

Añadir a `tests/test_preprocess.py`:

```python
@pytest.fixture
def sample_df_with_stats():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01'],
        'home_team': ['Arsenal', 'Arsenal', 'Chelsea'],
        'away_team': ['Chelsea', 'Liverpool', 'Arsenal'],
        'home_score': [2, 1, 0],
        'away_score': [1, 2, 1],
        'neutral': [False] * 3,
        'tournament': ['Premier League'] * 3,
        'league': ['Premier League'] * 3,
        'competition_type': ['club'] * 3,
        'home_shots': [10.0, 20.0, 8.0],
        'away_shots': [5.0, 12.0, 15.0],
        'home_shots_on_target': [4.0, 8.0, 3.0],
        'away_shots_on_target': [2.0, 5.0, 6.0],
        'home_corners': [6.0, 10.0, 4.0],
        'away_corners': [3.0, 7.0, 9.0],
        'home_yellow': [1.0, 3.0, 2.0],
        'away_yellow': [2.0, 1.0, 0.0],
        'home_red': [0.0, 1.0, 0.0],
        'away_red': [0.0, 0.0, 1.0],
    })


def test_feature_cols_include_match_stats():
    for col in ['home_avg_shots', 'home_avg_shots_on_target', 'home_avg_corners',
                'home_avg_yellow', 'home_avg_red',
                'away_avg_shots', 'away_avg_shots_on_target', 'away_avg_corners',
                'away_avg_yellow', 'away_avg_red']:
        assert col in FEATURE_COLS


def test_training_rolling_stats_use_previous_matches(sample_df_with_stats):
    df, _ = build_training_data(sample_df_with_stats)
    # Primer partido de Arsenal como local: sin historial -> 0
    assert df.iloc[0]['home_avg_shots'] == 0.0
    # Segundo partido de Arsenal como local: promedio del partido anterior (10 tiros)
    assert df.iloc[1]['home_avg_shots'] == 10.0
    assert df.iloc[1]['home_avg_yellow'] == 1.0


def test_training_stats_zero_when_columns_missing(sample_df):
    df, _ = build_training_data(sample_df)
    assert (df['home_avg_shots'] == 0.0).all()
    assert (df['away_avg_red'] == 0.0).all()


def test_prediction_features_include_stats(sample_df_with_stats):
    feats = build_features_for_prediction(sample_df_with_stats, 'Arsenal', 'Chelsea')
    # Arsenal de local: partidos con 10 y 20 tiros -> promedio 15
    assert feats['home_avg_shots'] == 15.0
    assert feats['home_avg_corners'] == 8.0
    # Chelsea de visitante: 1 partido con 5 tiros... (away en 2020-01-01)
    assert feats['away_avg_shots'] == 5.0


def test_prediction_stats_zero_when_columns_missing(sample_df):
    feats = build_features_for_prediction(sample_df, 'Brazil', 'Germany')
    assert feats['home_avg_shots'] == 0.0
    assert feats['away_avg_red'] == 0.0
```

- [ ] **Step 2.2: Verificar que fallan**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: los 5 tests nuevos FAIL (KeyError / AssertionError); el resto PASS.

- [ ] **Step 2.3: Implementar en `src/preprocess.py`**

Añadir constante a nivel de módulo (después de los imports):

```python
# feature suffix -> (columna del equipo local, columna del visitante)
MATCH_STAT_SOURCES = {
    'avg_shots': ('home_shots', 'away_shots'),
    'avg_shots_on_target': ('home_shots_on_target', 'away_shots_on_target'),
    'avg_corners': ('home_corners', 'away_corners'),
    'avg_yellow': ('home_yellow', 'away_yellow'),
    'avg_red': ('home_red', 'away_red'),
}
```

Ampliar `FEATURE_COLS` (al final de la lista existente):

```python
    'home_avg_shots',
    'home_avg_shots_on_target',
    'home_avg_corners',
    'home_avg_yellow',
    'home_avg_red',
    'away_avg_shots',
    'away_avg_shots_on_target',
    'away_avg_corners',
    'away_avg_yellow',
    'away_avg_red',
```

En `build_training_data`, después de `df['home_is_neutral'] = ...`, añadir:

```python
    for feat, (h_col, a_col) in MATCH_STAT_SOURCES.items():
        if h_col in df.columns and a_col in df.columns:
            df[f'home_{feat}'] = rolling_mean('home_team', h_col)
            df[f'away_{feat}'] = rolling_mean('away_team', a_col)
        else:
            df[f'home_{feat}'] = 0.0
            df[f'away_{feat}'] = 0.0
```

Nota: `rolling_mean` ya hace `shift(1).rolling(n, min_periods=1).mean().fillna(0)` — con valores `NaN` (partidos internacionales) el rolling mean los ignora y `fillna(0)` cubre el caso todo-NaN. No requiere cambios.

En `build_features_for_prediction`, después del bloque de `players_df`, añadir:

```python
    def stat_avg(matches, col):
        if col not in matches.columns:
            return 0.0
        return safe_mean(pd.to_numeric(matches[col], errors='coerce').dropna())

    match_stat_features = {}
    for feat, (h_col, a_col) in MATCH_STAT_SOURCES.items():
        match_stat_features[f'home_{feat}'] = stat_avg(hm, h_col)
        match_stat_features[f'away_{feat}'] = stat_avg(am, a_col)
```

y en el `return { ... }` final, añadir al cierre del dict:

```python
        **match_stat_features,
    }
```

- [ ] **Step 2.4: Verificar que pasan**

Run: `python -m pytest tests/test_preprocess.py -v`
Expected: todos PASS.

- [ ] **Step 2.5: Suite completa y commit**

Run: `python -m pytest -q`
Expected: todos PASS. Nota: `test_train.py` y `test_predict.py` usan `FEATURE_COLS` — si alguno asume una longitud fija de la lista, actualizar el valor esperado.

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: add rolling match-stat features (shots, corners, cards) to model"
```

---

### Task 3: `player_features.py` — plantilla top-N para la UI

**Files:**
- Modify: `src/player_features.py`
- Test: `tests/test_player_features.py`

- [ ] **Step 3.1: Escribir tests que fallan**

Añadir a `tests/test_player_features.py`. La fixture `players_df` existente no tiene `short_name`/`age`/`pace`/`attacking_finishing`, así que se añade una fixture nueva:

```python
@pytest.fixture
def squad_players_df():
    return pd.DataFrame({
        'short_name': ['Saka', 'Odegaard', 'Saliba', 'Lewandowski', 'Pedri'],
        'club_name': ['Arsenal', 'Arsenal', 'Arsenal', 'Barcelona', 'Barcelona'],
        'nationality_name': ['England', 'Norway', 'France', 'Poland', 'Spain'],
        'age': [22, 25, 23, 35, 21],
        'overall': [86, 87, 84, 91, 85],
        'shooting': [80, 78, 50, 92, 74],
        'attacking_finishing': [82, 76, 45, 94, 70],
        'pace': [85, 72, 80, 75, 79],
        'player_positions': ['RW', 'CAM', 'CB', 'ST', 'CM'],
    })


def test_get_team_squad_returns_top_n_by_overall(squad_players_df):
    squad = get_team_squad(squad_players_df, 'Arsenal', n=2)
    assert len(squad) == 2
    assert squad.iloc[0]['short_name'] == 'Odegaard'   # overall 87
    assert squad.iloc[1]['short_name'] == 'Saka'        # overall 86


def test_get_team_squad_includes_offensive_attributes(squad_players_df):
    squad = get_team_squad(squad_players_df, 'Barcelona')
    for col in ['short_name', 'player_positions', 'age', 'overall',
                'shooting', 'attacking_finishing', 'pace']:
        assert col in squad.columns


def test_get_team_squad_falls_back_to_nationality(squad_players_df):
    squad = get_team_squad(squad_players_df, 'Poland')
    assert len(squad) == 1
    assert squad.iloc[0]['short_name'] == 'Lewandowski'


def test_get_team_squad_unknown_team_returns_empty(squad_players_df):
    squad = get_team_squad(squad_players_df, 'Real Madrid')
    assert len(squad) == 0


def test_get_team_squad_none_players_returns_empty():
    squad = get_team_squad(None, 'Arsenal')
    assert len(squad) == 0
```

Actualizar el import del test:

```python
from src.player_features import (load_players, get_team_player_features,
                                 get_team_squad, TEAM_NAME_MAP)
```

- [ ] **Step 3.2: Verificar que fallan**

Run: `python -m pytest tests/test_player_features.py -v`
Expected: ImportError `cannot import name 'get_team_squad'`.

- [ ] **Step 3.3: Implementar en `src/player_features.py`**

Añadir al final del archivo:

```python
SQUAD_COLS = ['short_name', 'player_positions', 'age', 'overall',
              'shooting', 'attacking_finishing', 'pace']


def get_team_squad(players_df, team_name, n=11):
    """Top n players by overall rating for a club or national team.
    Returns an empty DataFrame when the team or the dataset is missing.
    """
    empty = pd.DataFrame(columns=SQUAD_COLS)
    if players_df is None or len(players_df) == 0:
        return empty

    normalized = TEAM_NAME_MAP.get(team_name, team_name)
    team = players_df[players_df['club_name'] == normalized]
    if len(team) == 0:
        team = players_df[players_df['nationality_name'] == normalized]
    if len(team) == 0:
        return empty

    cols = [c for c in SQUAD_COLS if c in team.columns]
    return team.nlargest(n, 'overall')[cols].reset_index(drop=True)
```

- [ ] **Step 3.4: Verificar que pasan**

Run: `python -m pytest tests/test_player_features.py -v`
Expected: todos PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/player_features.py tests/test_player_features.py
git commit -m "feat: add get_team_squad for top-N squad display"
```

---

### Task 4: `stats.py` — rendimiento reciente para visualización

**Files:**
- Modify: `src/stats.py`
- Test: `tests/test_stats.py`

- [ ] **Step 4.1: Escribir tests que fallan**

Añadir a `tests/test_stats.py` (importar `get_recent_performance` junto a los imports existentes de `src.stats`):

```python
@pytest.fixture
def df_with_match_stats():
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-08', '2024-01-15'],
        'home_team': ['Arsenal', 'Chelsea', 'Arsenal'],
        'away_team': ['Chelsea', 'Arsenal', 'Liverpool'],
        'home_score': [2, 0, 1],
        'away_score': [0, 1, 1],
        'home_shots': [15.0, 8.0, 12.0],
        'away_shots': [6.0, 14.0, 10.0],
        'home_shots_on_target': [7.0, 3.0, 5.0],
        'away_shots_on_target': [2.0, 6.0, 4.0],
        'home_corners': [8.0, 4.0, 6.0],
        'away_corners': [2.0, 7.0, 5.0],
        'home_yellow': [1.0, 2.0, 0.0],
        'away_yellow': [3.0, 1.0, 2.0],
        'home_red': [0.0, 0.0, 0.0],
        'away_red': [0.0, 1.0, 0.0],
    })


def test_recent_performance_averages_team_perspective(df_with_match_stats):
    perf = get_recent_performance(df_with_match_stats, 'Arsenal', n=5)
    # Arsenal: local 15 tiros, visitante 14, local 12 -> promedio 41/3
    assert perf['shots'] == round(41 / 3, 1)
    assert perf['red'] == round(1 / 3, 1)


def test_recent_performance_none_when_no_stat_columns():
    df = pd.DataFrame({
        'date': ['2024-01-01'],
        'home_team': ['Brazil'], 'away_team': ['Germany'],
        'home_score': [2], 'away_score': [0],
    })
    perf = get_recent_performance(df, 'Brazil')
    assert perf['shots'] is None
    assert perf['yellow'] is None


def test_recent_performance_none_when_all_nan(df_with_match_stats):
    df = df_with_match_stats.copy()
    for c in df.columns:
        if c.startswith(('home_', 'away_')) and c not in (
                'home_team', 'away_team', 'home_score', 'away_score'):
            df[c] = float('nan')
    perf = get_recent_performance(df, 'Arsenal')
    assert perf['shots'] is None
```

- [ ] **Step 4.2: Verificar que fallan**

Run: `python -m pytest tests/test_stats.py -v`
Expected: ImportError `cannot import name 'get_recent_performance'`.

- [ ] **Step 4.3: Implementar en `src/stats.py`**

Añadir al final del archivo:

```python
RECENT_STAT_SOURCES = {
    'shots': ('home_shots', 'away_shots'),
    'shots_on_target': ('home_shots_on_target', 'away_shots_on_target'),
    'corners': ('home_corners', 'away_corners'),
    'yellow': ('home_yellow', 'away_yellow'),
    'red': ('home_red', 'away_red'),
}


def get_recent_performance(df, team, n=5):
    """Average match stats over the team's last n matches (home or away).
    Values are None when the stat is unavailable for that team.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
    matches = matches.sort_values('date').tail(n)

    out = {}
    for key, (h_col, a_col) in RECENT_STAT_SOURCES.items():
        if h_col not in matches.columns or a_col not in matches.columns or len(matches) == 0:
            out[key] = None
            continue
        vals = matches.apply(
            lambda r: r[h_col] if r['home_team'] == team else r[a_col], axis=1)
        vals = pd.to_numeric(vals, errors='coerce').dropna()
        out[key] = round(float(vals.mean()), 1) if len(vals) else None
    return out
```

- [ ] **Step 4.4: Verificar que pasan**

Run: `python -m pytest tests/test_stats.py -v`
Expected: todos PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/stats.py tests/test_stats.py
git commit -m "feat: add get_recent_performance for last-5 match stat display"
```

---

### Task 5: `betting.py` — recomendación y value bets

**Files:**
- Create: `src/betting.py`
- Create: `tests/test_betting.py`

- [ ] **Step 5.1: Escribir tests que fallan**

Crear `tests/test_betting.py`:

```python
import pytest
from src.betting import recommend, expected_values


def test_recommend_high_confidence_single_outcome():
    rec = recommend({'H': 0.65, 'D': 0.20, 'A': 0.15})
    assert rec['market'] == 'H'
    assert rec['confidence'] == 'high'


def test_recommend_medium_confidence_single_outcome():
    rec = recommend({'H': 0.15, 'D': 0.30, 'A': 0.55})
    assert rec['market'] == 'A'
    assert rec['confidence'] == 'medium'


def test_recommend_double_chance():
    rec = recommend({'H': 0.45, 'D': 0.35, 'A': 0.20})
    assert rec['market'] == 'HD'      # doble oportunidad 1X
    assert rec['confidence'] == 'medium'


def test_recommend_double_chance_away():
    rec = recommend({'H': 0.20, 'D': 0.35, 'A': 0.45})
    assert rec['market'] == 'DA'      # doble oportunidad X2


def test_recommend_no_bet_when_unpredictable():
    rec = recommend({'H': 0.35, 'D': 0.33, 'A': 0.32})
    assert rec['market'] is None
    assert rec['confidence'] is None


def test_recommend_labels_are_strings():
    rec = recommend({'H': 0.70, 'D': 0.20, 'A': 0.10})
    assert isinstance(rec['label'], str) and len(rec['label']) > 0


def test_expected_values_positive_and_negative():
    probs = {'H': 0.50, 'D': 0.30, 'A': 0.20}
    evs = expected_values(probs, {'H': 2.50, 'D': 3.00, 'A': 4.00})
    assert evs['H'] == pytest.approx(0.25)    # 0.5*2.5-1
    assert evs['D'] == pytest.approx(-0.10)   # 0.3*3.0-1
    assert evs['A'] == pytest.approx(-0.20)   # 0.2*4.0-1


def test_expected_values_skips_invalid_odds():
    probs = {'H': 0.50, 'D': 0.30, 'A': 0.20}
    evs = expected_values(probs, {'H': 1.0, 'D': None, 'A': 0.0})
    assert evs == {}
```

- [ ] **Step 5.2: Verificar que fallan**

Run: `python -m pytest tests/test_betting.py -v`
Expected: ModuleNotFoundError `No module named 'src.betting'`.

- [ ] **Step 5.3: Crear `src/betting.py`**

```python
"""Bet recommendation from model probabilities.

Statistical tool only — does not guarantee outcomes.
"""

MARKET_LABELS = {
    'H': 'Home win (1)',
    'D': 'Draw (X)',
    'A': 'Away win (2)',
    'HD': 'Double chance 1X (home win or draw)',
    'DA': 'Double chance X2 (draw or away win)',
    'HA': 'Double chance 12 (home or away win)',
    None: 'No bet — match too unpredictable',
}

_ORDER = 'HDA'


def recommend(probs):
    """Threshold-based recommendation from {H, D, A} probabilities.

    >= 0.60 -> single outcome, high confidence
    >= 0.50 -> single outcome, medium confidence
    >= 0.40 -> double chance with the two most likely outcomes, medium
    <  0.40 -> no bet
    """
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top, top_p = ordered[0]
    if top_p >= 0.60:
        return {'market': top, 'label': MARKET_LABELS[top], 'confidence': 'high'}
    if top_p >= 0.50:
        return {'market': top, 'label': MARKET_LABELS[top], 'confidence': 'medium'}
    if top_p >= 0.40:
        second = ordered[1][0]
        market = ''.join(sorted(top + second, key=_ORDER.index))
        return {'market': market, 'label': MARKET_LABELS[market], 'confidence': 'medium'}
    return {'market': None, 'label': MARKET_LABELS[None], 'confidence': None}


def expected_values(probs, odds):
    """EV per 1X2 market for user-supplied decimal odds.

    EV = model probability x decimal odd - 1. Markets with missing or
    invalid odds (<= 1.0) are omitted.
    """
    out = {}
    for market in ('H', 'D', 'A'):
        odd = odds.get(market)
        if odd is None or odd <= 1.0:
            continue
        out[market] = round(probs.get(market, 0.0) * odd - 1.0, 3)
    return out
```

- [ ] **Step 5.4: Verificar que pasan**

Run: `python -m pytest tests/test_betting.py -v`
Expected: todos PASS.

- [ ] **Step 5.5: Commit**

```bash
git add src/betting.py tests/test_betting.py
git commit -m "feat: add betting recommendation and expected-value module"
```

---

### Task 6: Regenerar datos y reentrenar

**Files:**
- Modify: `data/all_matches.csv`, `data/clubs.csv`, `model.pkl` (generados)

- [ ] **Step 6.1: Regenerar datasets con las columnas de stats**

Run: `python -m src.ingest data/raw`
Expected: usa los CSVs cacheados (sin descargas), regenera `data/clubs.csv` y `data/all_matches.csv`. Verificar columnas:

Run: `python -c "import pandas as pd; df = pd.read_csv('data/all_matches.csv', nrows=2); print([c for c in df.columns if 'shots' in c or 'corners' in c or 'yellow' in c or 'red' in c])"`
Expected: las 10 columnas de stats.

- [ ] **Step 6.2: Reentrenar y comparar con la línea base**

Run: `python -m src.train`
Expected: imprime `Test accuracy: 0.XXX`. Comparar contra el número de Task 0 — debe ser igual o mejor. Si empeora notablemente (> 1 punto porcentual), reportarlo al usuario antes de continuar.

- [ ] **Step 6.3: Suite completa**

Run: `python -m pytest -q`
Expected: todos PASS.

- [ ] **Step 6.4: Commit**

`model.pkl` y los CSVs de data/ — verificar primero qué trackea git (`git status`). Si `model.pkl` está versionado (lo está, para el deploy de Streamlit):

```bash
git add model.pkl
git commit -m "chore: retrain model with match-stat features"
```

No commitear `data/all_matches.csv` / `data/clubs.csv` si no estaban versionados.

---

### Task 7: `app.py` — UI de plantillas, rendimiento y apuesta

**Files:**
- Modify: `app.py`

La app no tiene tests automatizados (es Streamlit); la verificación es manual. Importante: los campos de cuotas van ANTES del botón Predict — un `number_input` después del botón haría desaparecer los resultados al editar (rerun de Streamlit deja `predict_btn` en False).

- [ ] **Step 7.1: Imports y carga de jugadores**

En los imports de `app.py`, añadir:

```python
from src.player_features import load_players, get_team_player_features, get_team_squad
from src.stats import get_recent_performance
from src.betting import recommend, expected_values
```

(verificar el bloque de imports existente de `src.stats` y extenderlo en lugar de duplicarlo).

Después de la definición de `load_data()` (~línea 166), añadir:

```python
@st.cache_data
def load_players_cached():
    return load_players()

players_df = load_players_cached()
```

- [ ] **Step 7.2: Inputs de cuotas antes del botón Predict**

Justo antes del bloque `opt_col, btn_col = st.columns([1, 3])` (~línea 241), añadir:

```python
with st.expander("Betting odds (optional) — decimal odds from your bookmaker"):
    oc1, oc2, oc3 = st.columns(3)
    odds_h = oc1.number_input("Home (1)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_d = oc2.number_input("Draw (X)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
    odds_a = oc3.number_input("Away (2)", min_value=0.0, value=0.0, step=0.05, format="%.2f")
```

(`0.0` significa "no ingresada".)

- [ ] **Step 7.3: Tarjeta de recomendación de apuesta en Tab 1**

Dentro de `with tab1:`, después del bloque de team cards (tras el `for col, team, stats in [...]`, ~línea 398), añadir:

```python
        # ── Bet recommendation ──
        rec = recommend(probs)
        conf_text = {'high': 'HIGH CONFIDENCE', 'medium': 'MEDIUM CONFIDENCE', None: ''}[rec['confidence']]
        rec_color = '#00d4aa' if rec['market'] else '#f59e0b'
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(_card(
            f'<div style="font-size:0.62rem;color:#444;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">Bet recommendation</div>'
            f'<div style="font-size:1.2rem;font-weight:800;color:{rec_color}">{rec["label"]}</div>'
            + (f'<div style="font-size:0.7rem;color:#555;font-weight:700;margin-top:4px">{conf_text}</div>'
               if conf_text else ''),
            border=rec_color,
        ), unsafe_allow_html=True)

        user_odds = {'H': odds_h or None, 'D': odds_d or None, 'A': odds_a or None}
        evs = expected_values(probs, user_odds)
        if evs:
            ev_cols = st.columns(len(evs))
            names = {'H': f'Home {odds_h:.2f}', 'D': f'Draw {odds_d:.2f}', 'A': f'Away {odds_a:.2f}'}
            for col, (mkt, ev) in zip(ev_cols, evs.items()):
                col.metric(names[mkt], f"EV {ev:+.1%}",
                           delta="value bet" if ev > 0 else "no value",
                           delta_color="normal" if ev > 0 else "inverse")

        st.caption("Statistical tool only — predictions and recommendations do not "
                   "guarantee outcomes. Bet responsibly.")
```

- [ ] **Step 7.4: Expanders de plantillas, comparativa y rendimiento**

A continuación del bloque anterior (mismo nivel de indentación, dentro de `tab1`):

```python
        # ── Squads ──
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
                st.plotly_chart(fig_sq, use_container_width=True)

        # ── Recent performance ──
        with st.expander("Recent performance — last 5 matches"):
            h_perf = get_recent_performance(df, home_team)
            a_perf = get_recent_performance(df, away_team)
            stat_names = {'shots': 'Shots', 'shots_on_target': 'Shots on target',
                          'corners': 'Corners', 'yellow': 'Yellow cards',
                          'red': 'Red cards'}
            perf_df = pd.DataFrame({
                'Stat': list(stat_names.values()),
                home_team: [h_perf[k] if h_perf[k] is not None else '—' for k in stat_names],
                away_team: [a_perf[k] if a_perf[k] is not None else '—' for k in stat_names],
            })
            st.dataframe(perf_df, hide_index=True, use_container_width=True)
            if all(v is None for v in h_perf.values()) or all(v is None for v in a_perf.values()):
                st.caption("Match stats are only available for club league games "
                           "(not international fixtures).")
```

Verificar que `pd` (pandas) está importado en `app.py`; si no, añadir `import pandas as pd`.

- [ ] **Step 7.5: Prueba manual**

Run: `python -m streamlit run app.py --server.headless true --server.port 8501`

Verificar en el navegador:
1. Expander de cuotas visible antes del botón; predicción con y sin cuotas.
2. Tarjeta de recomendación aparece tras predecir; con cuotas, métricas de EV (verde para EV positivo).
3. Sin `players.csv`: caption con instrucciones; sin expanders de plantilla.
4. Equipos de liga (p.ej. Arsenal vs Chelsea): expander de rendimiento con números.
5. Selecciones (p.ej. Mexico vs Argentina): rendimiento muestra "—" y nota.

- [ ] **Step 7.6: Suite y commit**

Run: `python -m pytest -q`
Expected: todos PASS.

```bash
git add app.py
git commit -m "feat: squad tables, recent performance and bet recommendation in UI"
```

---

### Task 8: README y documentación

**Files:**
- Modify: `README.md`

- [ ] **Step 8.1: Documentar `players.csv` y las nuevas secciones**

En el README, en la sección de setup/datos, añadir:

```markdown
### Player data (optional)

Download the [FIFA 23 complete player dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)
from Kaggle (the `players_22.csv` or newer file) and save it as `data/players.csv`.
This enables:

- Squad-quality features in the model (`team_rating`, `team_attack`, `team_defense`)
- Squad tables and team comparison charts in the app

After adding the file, retrain: `python -m src.train`
```

En la lista de features de la app, añadir:

```markdown
- Recent performance: shots, shots on target, corners and cards over the last 5 matches
- Bet recommendation (1X2 / double chance / no bet) with optional value-bet
  detection from your bookmaker's odds. Statistical tool only — no guarantees.
```

- [ ] **Step 8.2: Commit**

```bash
git add README.md
git commit -m "docs: document players.csv setup and new app features"
```

---

## Self-Review Checklist (ejecutado al escribir el plan)

- **Cobertura del spec:** §1 jugadores → Tasks 3, 7.4, 8; §2 rendimiento → Tasks 1, 2, 4, 6; §3 apuestas → Tasks 5, 7.2-7.3; §4 UI → Task 7; §5 errores → cubierto en Tasks 1-5 (NaN/None/cuotas inválidas) y 7.4-7.5; §6 testing → tests en Tasks 1-5.
- **Placeholders:** ninguno — todo el código está completo.
- **Consistencia de tipos:** `recommend` → `{market, label, confidence}` usado igual en Task 5 y 7.3; `expected_values(probs, odds)` → dict por mercado; `get_team_squad` → DataFrame con `SQUAD_COLS`; `get_recent_performance` → dict con claves `shots/shots_on_target/corners/yellow/red`.
