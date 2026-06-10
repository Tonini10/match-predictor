# Ratings de Jugadores FIFA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incorporar ratings de jugadores FIFA/EA FC como features del modelo para mejorar la accuracy de predicción de ~47.7% a ~51-54%.

**Architecture:** Nuevo módulo `src/player_features.py` calcula 3 features por equipo (rating global, ataque, defensa) a partir de `data/players.csv`. `preprocess.py` las integra en `FEATURE_COLS` (de 11 a 18 features). `train.py` carga el CSV y lo guarda en el artifact. `predict.py` lo recupera del artifact para predicciones en producción.

**Tech Stack:** pandas, XGBoost, scikit-learn, joblib, pytest

---

## File Map

| Acción | Archivo | Descripción |
|--------|---------|-------------|
| Create | `src/player_features.py` | Carga players.csv y calcula features por equipo |
| Create | `tests/test_player_features.py` | Tests para player_features |
| Modify | `src/preprocess.py` | FEATURE_COLS crece a 18; build_training_data y build_features_for_prediction reciben players_df |
| Modify | `src/train.py` | Carga players_df, lo pasa a preprocess, lo guarda en artifact |
| Modify | `src/predict.py` | Recupera players_df del artifact, lo pasa a preprocess |
| Modify | `tests/test_preprocess.py` | Tests para nuevo parámetro players_df |
| Modify | `tests/test_train.py` | Test para players_df en artifact |
| Modify | `tests/test_predict.py` | Test para players_df en artifact y predict |

---

## Task 1: Descargar data/players.csv de Kaggle

**Files:**
- Create: `data/players.csv` (descarga manual)

- [ ] **Step 1: Descargar el dataset de Kaggle**

Ir a: https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset

Descargar el archivo `male_players.csv` (o equivalente con columnas `overall`, `shooting`, `dribbling`, `defending`, `physic`, `club_name`, `nationality_name`, `player_positions`).

Guardarlo como `data/players.csv` en la raíz del proyecto.

- [ ] **Step 2: Verificar las columnas requeridas**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/players.csv', nrows=5, low_memory=False)
required = ['short_name', 'club_name', 'nationality_name', 'overall',
            'shooting', 'dribbling', 'defending', 'physic', 'player_positions']
missing = [c for c in required if c not in df.columns]
print('Columnas disponibles:', list(df.columns[:20]))
print('Columnas faltantes:', missing)
"
```

Expected: `Columnas faltantes: []`

Si alguna columna tiene nombre distinto (ej. `club_team_id` en lugar de `club_name`), ajustar el nombre en `TEAM_NAME_MAP` de `player_features.py` según corresponda.

- [ ] **Step 3: Agregar players.csv al .gitignore si es grande (>50 MB)**

```bash
python -c "import os; size=os.path.getsize('data/players.csv')/1e6; print(f'{size:.1f} MB')"
```

Si supera 50 MB, agregar a `.gitattributes`:
```
data/players.csv filter=lfs diff=lfs merge=lfs -text
```

Si es menor a 50 MB, commitear directamente.

---

## Task 2: Crear src/player_features.py (TDD)

**Files:**
- Create: `tests/test_player_features.py`
- Create: `src/player_features.py`

- [ ] **Step 1: Crear tests/test_player_features.py**

```python
import pandas as pd
import pytest
from src.player_features import load_players, get_team_player_features, TEAM_NAME_MAP


@pytest.fixture
def players_df():
    return pd.DataFrame({
        'club_name': ['Arsenal', 'Arsenal', 'Arsenal', 'Barcelona', 'Barcelona'],
        'nationality_name': ['England', 'England', 'France', 'Spain', 'Spain'],
        'overall': [85, 80, 75, 90, 88],
        'shooting': [80, 70, 60, 88, 75],
        'dribbling': [75, 65, 55, 92, 80],
        'defending': [60, 70, 80, 55, 65],
        'physic': [70, 75, 80, 65, 72],
        'player_positions': ['ST', 'CM', 'CB', 'LW', 'CDM'],
    })


def test_load_players_returns_none_if_missing(tmp_path):
    result = load_players(str(tmp_path / 'nonexistent.csv'))
    assert result is None


def test_load_players_returns_dataframe_if_exists(tmp_path):
    path = tmp_path / 'players.csv'
    pd.DataFrame({'overall': [80]}).to_csv(path, index=False)
    result = load_players(str(path))
    assert isinstance(result, pd.DataFrame)


def test_get_team_player_features_returns_three_keys(players_df):
    result = get_team_player_features(players_df, 'Arsenal')
    assert set(result.keys()) == {'team_rating', 'team_attack', 'team_defense'}


def test_get_team_player_features_rating_is_normalized(players_df):
    result = get_team_player_features(players_df, 'Arsenal')
    assert 0.0 <= result['team_rating'] <= 1.0


def test_get_team_player_features_rating_correct_value(players_df):
    result = get_team_player_features(players_df, 'Arsenal')
    expected = (85 + 80 + 75) / 3 / 100.0
    assert result['team_rating'] == pytest.approx(expected, abs=0.01)


def test_get_team_player_features_returns_zeros_for_unknown_team(players_df):
    result = get_team_player_features(players_df, 'Unknown FC')
    assert result == {'team_rating': 0.0, 'team_attack': 0.0, 'team_defense': 0.0}


def test_get_team_player_features_normalizes_alias(players_df):
    players_df_mc = players_df.copy()
    players_df_mc['club_name'] = ['Manchester City'] * 5
    result_alias = get_team_player_features(players_df_mc, 'Man City')
    result_full = get_team_player_features(players_df_mc, 'Manchester City')
    assert result_alias['team_rating'] == pytest.approx(result_full['team_rating'])


def test_get_team_player_features_uses_nationality_for_international(players_df):
    result = get_team_player_features(players_df, 'England')
    assert result['team_rating'] > 0.0


def test_team_name_map_has_common_aliases():
    assert 'Man City' in TEAM_NAME_MAP
    assert 'Man United' in TEAM_NAME_MAP
```

- [ ] **Step 2: Run tests — verificar que fallan**

```bash
cd C:\Users\Esteban\match-predictor
python -m pytest tests/test_player_features.py -v 2>&1 | Select-Object -First 10
```

Expected: `ImportError: cannot import name 'load_players'`

- [ ] **Step 3: Crear src/player_features.py**

```python
from pathlib import Path
import pandas as pd

TEAM_NAME_MAP = {
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Spurs': 'Tottenham Hotspur',
    'Newcastle': 'Newcastle United',
    'West Ham': 'West Ham United',
    'Leicester': 'Leicester City',
    'Brighton': 'Brighton & Hove Albion',
    'Wolves': 'Wolverhampton Wanderers',
    'Nottm Forest': 'Nottingham Forest',
    'Atletico Madrid': 'Atlético de Madrid',
    'Betis': 'Real Betis',
    'Celta Vigo': 'Celta de Vigo',
    'USA': 'United States',
    'Korea Republic': 'Korea Republic',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
}

_ATTACK_RE = r'ST|CF|LW|RW|CAM|CM|LM|RM'
_DEFEND_RE = r'CB|LB|RB|LWB|RWB|GK|CDM'


def load_players(path='data/players.csv'):
    """Load FIFA players CSV. Returns None if file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False)


def get_team_player_features(players_df, team_name):
    """Return dict with team_rating, team_attack, team_defense (all normalized 0-1).
    Returns zeros if team not found in dataset.
    """
    zeros = {'team_rating': 0.0, 'team_attack': 0.0, 'team_defense': 0.0}
    if players_df is None or len(players_df) == 0:
        return zeros

    normalized = TEAM_NAME_MAP.get(team_name, team_name)

    team = players_df[players_df['club_name'] == normalized]
    if len(team) == 0:
        team = players_df[players_df['nationality_name'] == normalized]
    if len(team) == 0:
        return zeros

    rating = float(team['overall'].mean()) / 100.0

    attackers = team[team['player_positions'].str.contains(_ATTACK_RE, na=False, regex=True)]
    if len(attackers) > 0:
        attack = (float(attackers['shooting'].mean()) + float(attackers['dribbling'].mean())) / 2.0 / 100.0
    else:
        attack = rating

    defenders = team[team['player_positions'].str.contains(_DEFEND_RE, na=False, regex=True)]
    if len(defenders) > 0:
        defense = (float(defenders['defending'].mean()) + float(defenders['physic'].mean())) / 2.0 / 100.0
    else:
        defense = rating

    return {'team_rating': rating, 'team_attack': attack, 'team_defense': defense}
```

- [ ] **Step 4: Run tests — verificar que pasan**

```bash
python -m pytest tests/test_player_features.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Commit**

```bash
git add src/player_features.py tests/test_player_features.py
git commit -m "feat: add player_features module with FIFA rating extraction"
```

---

## Task 3: Extender src/preprocess.py con features de jugadores (TDD)

**Files:**
- Modify: `src/preprocess.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step 1: Agregar estos tests al FINAL de tests/test_preprocess.py**

```python
from src.player_features import get_team_player_features


@pytest.fixture
def players_df_simple():
    return pd.DataFrame({
        'club_name': ['Brazil', 'Germany', 'Argentina'],
        'nationality_name': ['Brazil', 'Germany', 'Argentina'],
        'overall': [88, 85, 90],
        'shooting': [85, 80, 88],
        'dribbling': [90, 78, 87],
        'defending': [70, 80, 72],
        'physic': [75, 82, 74],
        'player_positions': ['ST', 'CM', 'LW'],
    })


def test_build_training_data_with_players_df_adds_player_cols(sample_df, players_df_simple):
    df, _ = build_training_data(sample_df, players_df=players_df_simple)
    for col in ['home_team_rating', 'away_team_rating', 'home_team_attack',
                'away_team_attack', 'home_team_defense', 'away_team_defense', 'rating_diff']:
        assert col in df.columns, f"Missing column: {col}"


def test_build_training_data_without_players_df_player_cols_are_zero(sample_df):
    df, _ = build_training_data(sample_df, players_df=None)
    assert df['home_team_rating'].sum() == 0.0
    assert df['away_team_rating'].sum() == 0.0
    assert df['rating_diff'].sum() == 0.0


def test_build_features_for_prediction_with_players_df(sample_df, players_df_simple):
    features = build_features_for_prediction(
        sample_df, 'Brazil', 'Germany', players_df=players_df_simple
    )
    assert set(features.keys()) == set(FEATURE_COLS)
    assert features['home_team_rating'] > 0.0
    assert features['away_team_rating'] > 0.0


def test_build_features_for_prediction_without_players_df_zeros(sample_df):
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', players_df=None)
    assert features['home_team_rating'] == 0.0
    assert features['rating_diff'] == 0.0
```

- [ ] **Step 2: Run tests — verificar que fallan**

```bash
python -m pytest tests/test_preprocess.py -k "player" -v 2>&1 | Select-Object -First 15
```

Expected: FAIL — `build_training_data` no acepta `players_df`.

- [ ] **Step 3: Actualizar FEATURE_COLS en src/preprocess.py**

Reemplazar la lista `FEATURE_COLS` con:

```python
FEATURE_COLS = [
    'home_avg_goals_scored',
    'home_avg_goals_conceded',
    'home_win_rate',
    'away_avg_goals_scored',
    'away_avg_goals_conceded',
    'away_win_rate',
    'home_is_neutral',
    'league_encoded',
    'is_international',
    'home_league_win_rate',
    'away_league_win_rate',
    'home_team_rating',
    'away_team_rating',
    'home_team_attack',
    'away_team_attack',
    'home_team_defense',
    'away_team_defense',
    'rating_diff',
]
```

- [ ] **Step 4: Agregar import de player_features al inicio de src/preprocess.py**

```python
from src.player_features import get_team_player_features
```

- [ ] **Step 5: Actualizar build_training_data para aceptar players_df**

Cambiar la firma:
```python
def build_training_data(df, n=5, players_df=None):
```

Agregar este bloque ANTES de `return df[FEATURE_COLS + ['result']].copy(), le`:

```python
    # Player features — vectorized over unique teams for performance
    if players_df is not None:
        unique_teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        team_cache = {t: get_team_player_features(players_df, t) for t in unique_teams}
        df['home_team_rating']  = df['home_team'].map(lambda t: team_cache[t]['team_rating'])
        df['away_team_rating']  = df['away_team'].map(lambda t: team_cache[t]['team_rating'])
        df['home_team_attack']  = df['home_team'].map(lambda t: team_cache[t]['team_attack'])
        df['away_team_attack']  = df['away_team'].map(lambda t: team_cache[t]['team_attack'])
        df['home_team_defense'] = df['home_team'].map(lambda t: team_cache[t]['team_defense'])
        df['away_team_defense'] = df['away_team'].map(lambda t: team_cache[t]['team_defense'])
        df['rating_diff'] = df['home_team_rating'] - df['away_team_rating']
    else:
        for col in ['home_team_rating', 'away_team_rating', 'home_team_attack',
                    'away_team_attack', 'home_team_defense', 'away_team_defense', 'rating_diff']:
            df[col] = 0.0
```

- [ ] **Step 6: Actualizar build_features_for_prediction para aceptar players_df**

Cambiar la firma:
```python
def build_features_for_prediction(df, home_team, away_team, is_neutral=False, n=5,
                                   before_date=None, league=None, label_encoder=None,
                                   competition_type=None, players_df=None):
```

Agregar este bloque ANTES de `return {`:

```python
    if players_df is not None:
        h_p = get_team_player_features(players_df, home_team)
        a_p = get_team_player_features(players_df, away_team)
    else:
        h_p = {'team_rating': 0.0, 'team_attack': 0.0, 'team_defense': 0.0}
        a_p = {'team_rating': 0.0, 'team_attack': 0.0, 'team_defense': 0.0}
```

Agregar estas claves al dict de retorno:

```python
        'home_team_rating':  h_p['team_rating'],
        'away_team_rating':  a_p['team_rating'],
        'home_team_attack':  h_p['team_attack'],
        'away_team_attack':  a_p['team_attack'],
        'home_team_defense': h_p['team_defense'],
        'away_team_defense': a_p['team_defense'],
        'rating_diff':       h_p['team_rating'] - a_p['team_rating'],
```

- [ ] **Step 7: Run tests — verificar que pasan**

```bash
python -m pytest tests/test_preprocess.py -v
```

Expected: todos en PASS (incluyendo los 4 nuevos de jugadores).

- [ ] **Step 8: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: add player rating features to preprocess pipeline"
```

---

## Task 4: Actualizar src/train.py (TDD)

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Agregar test al FINAL de tests/test_train.py**

```python
def test_train_artifact_contains_players_df_key(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    artifact = joblib.load(model_path)
    # players_df can be None (no data/players.csv in test env) but key must exist
    assert 'players_df' in artifact
```

- [ ] **Step 2: Run test — verificar que falla**

```bash
python -m pytest tests/test_train.py::test_train_artifact_contains_players_df_key -v
```

Expected: FAIL — `'players_df' not in artifact`

- [ ] **Step 3: Actualizar src/train.py**

Agregar import al inicio:
```python
from src.player_features import load_players
```

Dentro de `train()`, reemplazar:
```python
    training_df, le = build_training_data(df, n=n)
```
Con:
```python
    players_df = load_players()
    training_df, le = build_training_data(df, n=n, players_df=players_df)
```

En la llamada a `joblib.dump`, agregar `'players_df': players_df`:
```python
    joblib.dump({
        'model': clf,
        'feature_cols': FEATURE_COLS,
        'n': n,
        'league_encoder': le,
        'result_encoder': result_encoder,
        'players_df': players_df,
    }, model_path)
```

- [ ] **Step 4: Run todos los tests de train**

```bash
python -m pytest tests/test_train.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: load and save players_df in training artifact"
```

---

## Task 5: Actualizar src/predict.py (TDD)

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Agregar test al FINAL de tests/test_predict.py**

```python
def test_predict_match_uses_players_df_from_artifact(sample_df, tmp_path):
    le = LabelEncoder()
    le.fit(['International', 'Premier League'])
    re = LabelEncoder()
    re.fit(['A', 'D', 'H'])

    players_df = pd.DataFrame({
        'club_name': ['Brazil', 'Germany'],
        'nationality_name': ['Brazil', 'Germany'],
        'overall': [88, 85],
        'shooting': [85, 80],
        'dribbling': [90, 78],
        'defending': [70, 80],
        'physic': [75, 82],
        'player_positions': ['ST', 'CM'],
    })

    clf = XGBClassifier(n_estimators=2, random_state=42, eval_metric='mlogloss')
    X = pd.DataFrame([[0.5] * len(FEATURE_COLS)] * 6, columns=FEATURE_COLS)
    y_enc = re.transform(['H', 'D', 'A', 'H', 'D', 'A'])
    clf.fit(X, y_enc)

    path = str(tmp_path / 'model_with_players.pkl')
    joblib.dump({
        'model': clf, 'feature_cols': FEATURE_COLS, 'n': 5,
        'league_encoder': le, 'result_encoder': re, 'players_df': players_df,
    }, path)

    # tmp_path gives a unique path — no cache collision risk
    result = predict_match('Brazil', 'Germany', sample_df, path)
    assert result['result'] in ['H', 'D', 'A']
```

- [ ] **Step 2: Run test — verificar que falla**

```bash
python -m pytest tests/test_predict.py::test_predict_match_uses_players_df_from_artifact -v
```

Expected: FAIL o ERROR (players_df no se pasa a build_features_for_prediction).

- [ ] **Step 3: Actualizar src/predict.py**

En la función `predict_match`, extraer `players_df` del artifact y pasarlo a `build_features_for_prediction`:

Reemplazar:
```python
    label_encoder = artifact.get('league_encoder')

    features = build_features_for_prediction(
        df, home_team, away_team,
        is_neutral=is_neutral, n=n,
        league=league, label_encoder=label_encoder,
        competition_type=competition_type,
    )
```

Con:
```python
    label_encoder = artifact.get('league_encoder')
    players_df = artifact.get('players_df')

    features = build_features_for_prediction(
        df, home_team, away_team,
        is_neutral=is_neutral, n=n,
        league=league, label_encoder=label_encoder,
        competition_type=competition_type,
        players_df=players_df,
    )
```

- [ ] **Step 4: Run todos los tests de predict**

```bash
python -m pytest tests/test_predict.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: pass players_df from artifact to prediction pipeline"
```

---

## Task 6: Reentrenar modelo y verificar mejora

**Files:** ninguno (operación de datos)

- [ ] **Step 1: Run todos los tests para confirmar que nada está roto**

```bash
python -m pytest tests/ -v 2>&1 | Select-Object -Last 5
```

Expected: todos en PASS.

- [ ] **Step 2: Reentrenar el modelo con players.csv**

```bash
python -m src.train
```

Expected:
```
Test accuracy: 0.XXX   ← esperado: >0.50 (mejora sobre 0.477)
Model saved to model.pkl
```

- [ ] **Step 3: Verificar smoke test**

```bash
python -c "
import joblib, pandas as pd
from src.predict import predict_match, _load_artifact
_load_artifact.cache_clear()
df = pd.read_csv('data/all_matches.csv', parse_dates=['date'])
r = predict_match('Arsenal', 'Chelsea', df, 'model.pkl',
                  league='Premier League', competition_type='club')
print('Result:', r['result_label'])
print('Probabilities:', {k: round(v,3) for k,v in r['probabilities'].items()})
print('SUCCESS' if r['result'] in ['H','D','A'] else 'FAIL')
"
```

Expected: predicción válida con `SUCCESS`.

- [ ] **Step 4: Commitear el modelo actualizado**

```bash
git add model.pkl data/players.csv
git commit -m "feat: retrain model with FIFA player rating features"
git push origin master
git push hf master   # si ya está desplegado en HF Spaces
```

---

## Self-review notes

- `get_team_player_features` usa primero `club_name` y luego `nationality_name` como fallback para selecciones nacionales.
- El caché de equipos en `build_training_data` es O(equipos únicos) en lugar de O(filas) — crítico para 234K filas.
- `players_df=None` en todos los parámetros garantiza backward compat — el modelo entrena con features = 0.0 si no hay CSV.
- El nuevo `FEATURE_COLS` de 18 features **requiere reentrenar** — el model.pkl anterior no es compatible.
