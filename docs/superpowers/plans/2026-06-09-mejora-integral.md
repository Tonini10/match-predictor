# Mejora Integral — Match Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expandir el predictor de selecciones a todas las ligas de clubes disponibles, mejorar el modelo con XGBoost y nuevas features de liga, y actualizar la UI para que sea profesional, sin emojis, con selector de competición y sección destacada para el Mundial.

**Architecture:** Se crea `src/ingest.py` que descarga y normaliza CSVs de football-data.co.uk y produce `data/all_matches.csv` combinando datos de clubes con el dataset internacional existente. `src/preprocess.py` se extiende con cuatro nuevas features de liga y devuelve un `LabelEncoder` que se guarda en `model.pkl`. El modelo migra de `RandomForestClassifier` a `XGBClassifier`. `app.py` recibe un selector de competición que filtra equipos y pasa `league` al predictor.

**Tech Stack:** Python 3.x, pandas, scikit-learn, XGBoost, Streamlit, Plotly, joblib, pytest

---

## File Map

| Acción | Archivo | Responsabilidad |
|--------|---------|-----------------|
| Create | `src/ingest.py` | Descarga, normalización y combinación de datasets |
| Create | `tests/test_ingest.py` | Tests para ingest |
| Modify | `src/preprocess.py` | Nuevas features: league_encoded, is_international, home/away_league_win_rate; retorno de LabelEncoder |
| Modify | `src/train.py` | Migración a XGBClassifier; guarda league_encoder en artifact |
| Modify | `src/predict.py` | Parámetro `league` y `competition_type`; carga league_encoder del artifact |
| Modify | `src/stats.py` | Nueva función `get_league_stats()` |
| Modify | `app.py` | Selector de competición, filtro de equipos, pestaña League, eliminación de emojis |
| Modify | `tests/test_preprocess.py` | Adaptar a tuple return y nuevas features |
| Modify | `tests/test_train.py` | Verificar league_encoder en artifact |
| Modify | `tests/test_predict.py` | Verificar parámetro league; fixture con XGBoost |
| Modify | `tests/test_stats.py` | Tests para get_league_stats |
| Modify | `requirements.txt` | Agregar xgboost>=2.0.0 |

---

## Task 1: Add xgboost dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Agregar xgboost a requirements.txt**

El archivo debe quedar:
```
streamlit>=1.36.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
plotly>=5.18.0
pytest>=7.0.0
```

- [ ] **Step 2: Instalar la dependencia**

```bash
pip install xgboost>=2.0.0
```

Expected: instalación exitosa sin errores.

- [ ] **Step 3: Verificar que los tests existentes siguen pasando**

```bash
cd C:\Users\Esteban\match-predictor
pytest tests/ -v
```

Expected: todos los tests en PASS (no se rompió nada con el cambio de requirements).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add xgboost dependency"
```

---

## Task 2: Create src/ingest.py — normalize_csv() (TDD)

**Files:**
- Create: `tests/test_ingest.py`
- Create: `src/ingest.py` (parcial — solo normalize_csv)

- [ ] **Step 1: Crear tests/test_ingest.py con tests de normalize_csv**

```python
import pandas as pd
import pytest
from src.ingest import normalize_csv


def _raw(date='01/08/2024', home='Arsenal', away='Chelsea', fthg=2, ftag=1, ftr='H'):
    return pd.DataFrame({
        'Date': [date], 'HomeTeam': [home], 'AwayTeam': [away],
        'FTHG': [fthg], 'FTAG': [ftag], 'FTR': [ftr],
    })


def test_normalize_csv_renames_columns():
    result = normalize_csv(_raw(), 'Premier League')
    expected_cols = {'date', 'home_team', 'away_team', 'home_score', 'away_score',
                     'result', 'neutral', 'tournament', 'league', 'competition_type'}
    assert expected_cols == set(result.columns)


def test_normalize_csv_sets_league_name():
    result = normalize_csv(_raw(), 'La Liga')
    assert result['league'].iloc[0] == 'La Liga'


def test_normalize_csv_sets_competition_type_club_by_default():
    result = normalize_csv(_raw(), 'Premier League')
    assert result['competition_type'].iloc[0] == 'club'


def test_normalize_csv_parses_date():
    result = normalize_csv(_raw(date='15/03/2023'), 'Serie A')
    assert str(result['date'].iloc[0].date()) == '2023-03-15'


def test_normalize_csv_drops_rows_with_missing_scores():
    raw = pd.DataFrame({
        'Date': ['01/08/2024', '08/08/2024'],
        'HomeTeam': ['Arsenal', 'Man City'],
        'AwayTeam': ['Chelsea', 'Liverpool'],
        'FTHG': [2, None], 'FTAG': [1, 1], 'FTR': ['H', None],
    })
    result = normalize_csv(raw, 'Premier League')
    assert len(result) == 1


def test_normalize_csv_infers_result_when_ftr_missing():
    raw = pd.DataFrame({
        'Date': ['01/08/2024'], 'HomeTeam': ['Arsenal'], 'AwayTeam': ['Chelsea'],
        'FTHG': [0], 'FTAG': [0],
    })
    result = normalize_csv(raw, 'La Liga')
    assert result['result'].iloc[0] == 'D'


def test_normalize_csv_infers_home_win():
    raw = pd.DataFrame({
        'Date': ['01/08/2024'], 'HomeTeam': ['Arsenal'], 'AwayTeam': ['Chelsea'],
        'FTHG': [3], 'FTAG': [1],
    })
    result = normalize_csv(raw, 'La Liga')
    assert result['result'].iloc[0] == 'H'


def test_normalize_csv_infers_away_win():
    raw = pd.DataFrame({
        'Date': ['01/08/2024'], 'HomeTeam': ['Arsenal'], 'AwayTeam': ['Chelsea'],
        'FTHG': [0], 'FTAG': [2],
    })
    result = normalize_csv(raw, 'La Liga')
    assert result['result'].iloc[0] == 'A'


def test_normalize_csv_neutral_is_false():
    result = normalize_csv(_raw(), 'Bundesliga')
    assert result['neutral'].iloc[0] is False
```

- [ ] **Step 2: Run tests — verificar que fallan**

```bash
pytest tests/test_ingest.py -v
```

Expected: `ImportError: cannot import name 'normalize_csv' from 'src.ingest'`

- [ ] **Step 3: Crear src/ingest.py con normalize_csv**

```python
import io
import urllib.request
from pathlib import Path

import pandas as pd

LEAGUES = {
    'E0': 'Premier League',
    'E1': 'Championship',
    'E2': 'League One',
    'SC0': 'Scottish Premiership',
    'SP1': 'La Liga',
    'SP2': 'La Liga 2',
    'D1': 'Bundesliga',
    'D2': '2. Bundesliga',
    'I1': 'Serie A',
    'I2': 'Serie B',
    'F1': 'Ligue 1',
    'F2': 'Ligue 2',
    'N1': 'Eredivisie',
    'B1': 'Pro League',
    'P1': 'Primeira Liga',
    'T1': 'Super Lig',
    'G1': 'Super League Greece',
}

SEASONS = [
    '9394', '9495', '9596', '9697', '9798', '9899', '9900',
    '0001', '0102', '0203', '0304', '0405', '0506', '0607',
    '0708', '0809', '0910', '1011', '1112', '1213', '1314',
    '1415', '1516', '1617', '1718', '1819', '1920', '2021',
    '2122', '2223', '2324', '2425',
]

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{season}/{code}.csv'


def normalize_csv(raw_df, league_name, competition_type='club'):
    """Normalize a football-data.co.uk raw DataFrame to the unified schema."""
    col_map = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_score',
        'FTAG': 'away_score',
        'FTR': 'result',
    }
    df = raw_df.rename(columns=col_map).copy()
    required = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    df = df[df[required].notna().all(axis=1)].copy()

    keep = required + (['result'] if 'result' in df.columns else [])
    df = df[keep].copy()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    if 'result' not in df.columns:
        df['result'] = df.apply(
            lambda r: 'H' if r['home_score'] > r['away_score']
            else ('D' if r['home_score'] == r['away_score'] else 'A'),
            axis=1,
        )

    df['neutral'] = False
    df['tournament'] = league_name
    df['league'] = league_name
    df['competition_type'] = competition_type
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run tests — verificar que pasan**

```bash
pytest tests/test_ingest.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/test_ingest.py
git commit -m "feat: add ingest normalize_csv with tests"
```

---

## Task 3: Create src/ingest.py — download + build_clubs_dataset + __main__

**Files:**
- Modify: `src/ingest.py` (agregar funciones de descarga y combinación)
- Modify: `tests/test_ingest.py` (agregar tests de descarga y combinación)

- [ ] **Step 1: Agregar tests de download y combine a tests/test_ingest.py**

Agregar al final de `tests/test_ingest.py`:

```python
from unittest.mock import patch, MagicMock
from src.ingest import download_csv, build_clubs_dataset, combine_datasets
import os


def test_download_csv_returns_none_on_http_error(tmp_path):
    with patch('urllib.request.urlretrieve', side_effect=Exception('404')):
        result = download_csv('E0', '2425', str(tmp_path))
    assert result is None


def test_download_csv_skips_if_file_exists(tmp_path):
    dest = tmp_path / 'E0_2425.csv'
    dest.write_text('existing')
    with patch('urllib.request.urlretrieve') as mock_dl:
        result = download_csv('E0', '2425', str(tmp_path))
    mock_dl.assert_not_called()
    assert str(result) == str(dest)


def test_build_clubs_dataset_normalizes_multiple_csvs(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    # Write two tiny CSVs
    csv1 = raw_dir / 'E0_2324.csv'
    csv1.write_text('Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/2023,Arsenal,Chelsea,2,1,H\n')
    csv2 = raw_dir / 'SP1_2324.csv'
    csv2.write_text('Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n05/08/2023,Barcelona,Madrid,1,0,H\n')
    # Map just these two codes
    leagues = {'E0': 'Premier League', 'SP1': 'La Liga'}
    seasons = ['2324']
    result = build_clubs_dataset(str(raw_dir), leagues=leagues, seasons=seasons)
    assert len(result) == 2
    assert set(result['league'].tolist()) == {'Premier League', 'La Liga'}
    assert 'competition_type' in result.columns
    assert (result['competition_type'] == 'club').all()


def test_combine_datasets_adds_league_to_international(tmp_path):
    intl_path = tmp_path / 'results.csv'
    intl_df = pd.DataFrame({
        'date': ['2020-01-01'],
        'home_team': ['Brazil'],
        'away_team': ['Germany'],
        'home_score': [2],
        'away_score': [1],
        'neutral': [False],
        'tournament': ['Friendly'],
    })
    intl_df.to_csv(intl_path, index=False)

    clubs_df = pd.DataFrame({
        'date': pd.to_datetime(['2020-02-01']),
        'home_team': ['Arsenal'],
        'away_team': ['Chelsea'],
        'home_score': [1],
        'away_score': [0],
        'neutral': [False],
        'tournament': ['Premier League'],
        'league': ['Premier League'],
        'competition_type': ['club'],
        'result': ['H'],
    })

    combined = combine_datasets(clubs_df, str(intl_path))
    assert len(combined) == 2
    assert 'league' in combined.columns
    assert 'competition_type' in combined.columns
    assert set(combined['competition_type'].tolist()) == {'club', 'international'}
```

- [ ] **Step 2: Run tests — verificar que fallan**

```bash
pytest tests/test_ingest.py::test_download_csv_returns_none_on_http_error -v
```

Expected: `ImportError: cannot import name 'download_csv'`

- [ ] **Step 3: Agregar download_csv, build_clubs_dataset y combine_datasets a src/ingest.py**

Agregar después de `normalize_csv`:

```python
def download_csv(league_code, season, raw_dir):
    """Download a league/season CSV. Returns path if successful, None on error."""
    dest = Path(raw_dir) / f'{league_code}_{season}.csv'
    if dest.exists():
        return dest
    url = BASE_URL.format(season=season, code=league_code)
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception:
        return None


def build_clubs_dataset(raw_dir, leagues=None, seasons=None):
    """Download all league CSVs and return a unified normalized DataFrame."""
    if leagues is None:
        leagues = LEAGUES
    if seasons is None:
        seasons = SEASONS
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    frames = []
    for code, name in leagues.items():
        for season in seasons:
            path = Path(raw_dir) / f'{code}_{season}.csv'
            if not path.exists():
                path = download_csv(code, season, raw_dir)
            if path is None or not path.exists():
                continue
            try:
                raw = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
                normalized = normalize_csv(raw, name)
                frames.append(normalized)
            except Exception as exc:
                print(f"Warning: skipping {code} {season}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def combine_datasets(clubs_df, international_path='data/results.csv'):
    """Merge clubs DataFrame with existing international results.csv."""
    intl = pd.read_csv(international_path)
    intl['date'] = pd.to_datetime(intl['date'])
    intl['league'] = intl.get('tournament', pd.Series('International', index=intl.index)).fillna('International')
    intl['competition_type'] = 'international'
    combined = pd.concat([intl, clubs_df], ignore_index=True)
    return combined.sort_values('date').reset_index(drop=True)


if __name__ == '__main__':
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    print(f'Downloading club data to {raw_dir}...')
    clubs = build_clubs_dataset(raw_dir)
    print(f'Downloaded {len(clubs)} club matches.')
    clubs.to_csv('data/clubs.csv', index=False)
    print('Saved data/clubs.csv')
    combined = combine_datasets(clubs)
    combined.to_csv('data/all_matches.csv', index=False)
    print(f'Saved data/all_matches.csv ({len(combined)} total matches)')
```

- [ ] **Step 4: Run tests — verificar que pasan**

```bash
pytest tests/test_ingest.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Agregar data/raw/ al .gitignore**

Abrir `.gitignore` y agregar:
```
data/raw/
data/clubs.csv
data/all_matches.csv
```

- [ ] **Step 6: Commit**

```bash
git add src/ingest.py tests/test_ingest.py .gitignore
git commit -m "feat: add ingest download, combine, and __main__ with tests"
```

---

## Task 4: Extend src/preprocess.py — nuevas features (TDD)

**Files:**
- Modify: `src/preprocess.py`
- Modify: `tests/test_preprocess.py`

- [ ] **Step 1: Actualizar tests/test_preprocess.py**

Reemplazar el contenido completo de `tests/test_preprocess.py` con:

```python
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from src.preprocess import build_training_data, build_features_for_prediction, FEATURE_COLS


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01',
                 '2020-04-01', '2020-05-01', '2020-06-01'],
        'home_team': ['Brazil', 'Brazil', 'Brazil', 'Argentina', 'Brazil', 'Germany'],
        'away_team': ['Germany', 'France',  'Spain',  'Brazil',   'Italy',  'Brazil'],
        'home_score': [3, 2, 1, 0, 2, 1],
        'away_score': [1, 2, 0, 1, 0, 0],
        'neutral': [False] * 6,
        'tournament': ['Friendly'] * 6,
    })


@pytest.fixture
def sample_df_with_leagues():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01'],
        'home_team': ['Arsenal', 'Arsenal', 'Barcelona', 'Arsenal'],
        'away_team': ['Chelsea', 'Liverpool', 'Madrid', 'Man City'],
        'home_score': [2, 1, 3, 0],
        'away_score': [1, 2, 0, 1],
        'neutral': [False] * 4,
        'tournament': ['Premier League', 'Premier League', 'La Liga', 'Premier League'],
        'league': ['Premier League', 'Premier League', 'La Liga', 'Premier League'],
        'competition_type': ['club', 'club', 'club', 'club'],
    })


def test_build_training_data_returns_tuple(sample_df):
    result = build_training_data(sample_df)
    assert isinstance(result, tuple) and len(result) == 2


def test_build_training_data_returns_expected_columns(sample_df):
    df, le = build_training_data(sample_df)
    assert set(FEATURE_COLS + ['result']) == set(df.columns)


def test_build_training_data_returns_label_encoder(sample_df):
    df, le = build_training_data(sample_df)
    assert isinstance(le, LabelEncoder)


def test_build_training_data_length_matches_input(sample_df):
    df, _ = build_training_data(sample_df)
    assert len(df) == len(sample_df)


def test_build_training_data_result_home_win(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[0]['result'] == 'H'


def test_build_training_data_result_draw(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[1]['result'] == 'D'


def test_build_training_data_result_away_win(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[3]['result'] == 'A'


def test_build_training_data_first_match_stats_are_zero(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[0]['home_avg_goals_scored'] == 0.0
    assert df.iloc[0]['home_win_rate'] == 0.0


def test_build_training_data_second_match_uses_first(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[1]['home_avg_goals_scored'] == pytest.approx(3.0)


def test_build_training_data_has_new_feature_cols(sample_df):
    df, _ = build_training_data(sample_df)
    assert 'league_encoded' in df.columns
    assert 'is_international' in df.columns
    assert 'home_league_win_rate' in df.columns
    assert 'away_league_win_rate' in df.columns


def test_build_training_data_backward_compat_no_league_col(sample_df):
    # sample_df has no 'league' column — should not raise
    df, le = build_training_data(sample_df)
    assert len(df) == len(sample_df)
    assert 'league_encoded' in df.columns


def test_build_training_data_league_win_rate_per_league(sample_df_with_leagues):
    df, _ = build_training_data(sample_df_with_leagues)
    # Arsenal row 0: first Premier League match -> home_league_win_rate = 0
    assert df.iloc[0]['home_league_win_rate'] == 0.0
    # Arsenal row 1: prior PL home match was a win -> home_league_win_rate = 1.0
    assert df.iloc[1]['home_league_win_rate'] == pytest.approx(1.0)


def test_build_training_data_is_international_flag(sample_df_with_leagues):
    df, _ = build_training_data(sample_df_with_leagues)
    assert (df['is_international'] == 0).all()


def test_build_features_for_prediction_returns_all_keys(sample_df):
    df, le = build_training_data(sample_df)
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', label_encoder=le)
    assert set(features.keys()) == set(FEATURE_COLS)


def test_build_features_for_prediction_neutral_flag(sample_df):
    df, le = build_training_data(sample_df)
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', is_neutral=True, label_encoder=le)
    assert features['home_is_neutral'] == 1


def test_build_features_for_prediction_unknown_team_returns_zeros(sample_df):
    features = build_features_for_prediction(sample_df, 'Unknown', 'Brazil')
    assert features['home_avg_goals_scored'] == 0.0
    assert features['home_win_rate'] == 0.0


def test_build_features_for_prediction_league_encoded_zero_without_encoder(sample_df):
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', league='Friendly')
    assert features['league_encoded'] == 0


def test_build_features_for_prediction_is_international_from_param(sample_df):
    features = build_features_for_prediction(
        sample_df, 'Brazil', 'Germany', competition_type='international'
    )
    assert features['is_international'] == 1

    features_club = build_features_for_prediction(
        sample_df, 'Brazil', 'Germany', competition_type='club'
    )
    assert features_club['is_international'] == 0


def test_build_training_data_away_stats_rolling_correctness():
    df = pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01'],
        'home_team': ['Brazil', 'France'],
        'away_team': ['Germany', 'Germany'],
        'home_score': [3, 2],
        'away_score': [1, 2],
        'neutral': [False, False],
        'tournament': ['Friendly', 'Friendly'],
    })
    result, _ = build_training_data(df)
    assert result.iloc[0]['away_avg_goals_scored'] == 0.0
    assert result.iloc[1]['away_avg_goals_scored'] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests — verificar que fallan**

```bash
pytest tests/test_preprocess.py -v
```

Expected: múltiples FAILs — `build_training_data` returns un DataFrame, no una tupla; faltan las nuevas features.

- [ ] **Step 3: Actualizar FEATURE_COLS y build_training_data en src/preprocess.py**

Reemplazar el contenido completo de `src/preprocess.py`:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
]


def build_training_data(df, n=5):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if 'league' not in df.columns:
        df['league'] = df.get('tournament', pd.Series('International', index=df.index)).fillna('International')
    if 'competition_type' not in df.columns:
        df['competition_type'] = 'international'

    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['away_win'] = (df['away_score'] > df['home_score']).astype(int)

    def rolling_mean(group_col, val_col):
        return df.groupby(group_col)[val_col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean().fillna(0)
        )

    def rolling_mean_by_league(team_col, val_col):
        return df.groupby([team_col, 'league'])[val_col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean().fillna(0)
        )

    df['home_avg_goals_scored']   = rolling_mean('home_team', 'home_score')
    df['home_avg_goals_conceded'] = rolling_mean('home_team', 'away_score')
    df['home_win_rate']           = rolling_mean('home_team', 'home_win')
    df['away_avg_goals_scored']   = rolling_mean('away_team', 'away_score')
    df['away_avg_goals_conceded'] = rolling_mean('away_team', 'home_score')
    df['away_win_rate']           = rolling_mean('away_team', 'away_win')
    df['home_is_neutral']         = df['neutral'].astype(int)

    df['home_league_win_rate'] = rolling_mean_by_league('home_team', 'home_win')
    df['away_league_win_rate'] = rolling_mean_by_league('away_team', 'away_win')
    df['is_international'] = (df['competition_type'] == 'international').astype(int)

    le = LabelEncoder()
    df['league_encoded'] = le.fit_transform(df['league'].fillna('Unknown'))

    df['result'] = df.apply(
        lambda r: 'H' if r['home_score'] > r['away_score']
                  else ('D' if r['home_score'] == r['away_score'] else 'A'),
        axis=1,
    )

    return df[FEATURE_COLS + ['result']].copy(), le


def build_features_for_prediction(df, home_team, away_team, is_neutral=False, n=5,
                                   before_date=None, league=None, label_encoder=None,
                                   competition_type=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if before_date is not None:
        df = df[df['date'] < pd.Timestamp(before_date)]

    hm = df[df['home_team'] == home_team].tail(n)
    am = df[df['away_team'] == away_team].tail(n)

    def safe_mean(series):
        return 0.0 if len(series) == 0 or series.isna().all() else float(series.mean())

    if league and 'league' in df.columns:
        hm_l = df[(df['home_team'] == home_team) & (df['league'] == league)].tail(n)
        am_l = df[(df['away_team'] == away_team) & (df['league'] == league)].tail(n)
        home_league_win_rate = safe_mean((hm_l['home_score'] > hm_l['away_score']).astype(float))
        away_league_win_rate = safe_mean((am_l['away_score'] > am_l['home_score']).astype(float))
    else:
        home_league_win_rate = 0.0
        away_league_win_rate = 0.0

    if label_encoder is not None and league:
        try:
            league_enc = int(label_encoder.transform([league])[0])
        except ValueError:
            league_enc = 0
    else:
        league_enc = 0

    is_international = 1 if competition_type == 'international' else 0

    return {
        'home_avg_goals_scored':   safe_mean(hm['home_score']),
        'home_avg_goals_conceded': safe_mean(hm['away_score']),
        'home_win_rate':           safe_mean((hm['home_score'] > hm['away_score']).astype(float)),
        'away_avg_goals_scored':   safe_mean(am['away_score']),
        'away_avg_goals_conceded': safe_mean(am['home_score']),
        'away_win_rate':           safe_mean((am['away_score'] > am['home_score']).astype(float)),
        'home_is_neutral':         int(is_neutral),
        'league_encoded':          league_enc,
        'is_international':        is_international,
        'home_league_win_rate':    home_league_win_rate,
        'away_league_win_rate':    away_league_win_rate,
    }
```

- [ ] **Step 4: Run tests — verificar que pasan**

```bash
pytest tests/test_preprocess.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Verificar que los demás tests no se rompieron en cascada**

```bash
pytest tests/ -v --ignore=tests/test_train.py --ignore=tests/test_predict.py
```

Expected: `test_preprocess.py` y `test_ingest.py` en PASS. `test_stats.py` puede fallar — se arregla en Task 7. `test_train.py` y `test_predict.py` fallarán porque usan la tupla-nueva — se arreglan en tasks siguientes.

- [ ] **Step 6: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: extend preprocess with league features and LabelEncoder"
```

---

## Task 5: Migrate src/train.py to XGBoost (TDD)

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Actualizar tests/test_train.py**

Reemplazar el contenido completo de `tests/test_train.py`:

```python
import os
import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from src.train import train
from src.preprocess import FEATURE_COLS


@pytest.fixture
def sample_csv(tmp_path):
    dates = pd.date_range(start='2000-01-01', periods=20, freq='30D')
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d').tolist(),
        'home_team': ['Brazil', 'Argentina'] * 10,
        'away_team': ['Germany', 'France'] * 10,
        'home_score': [2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0],
        'away_score': [1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1],
        'neutral': [False] * 20,
        'tournament': ['Friendly'] * 20,
    })
    path = tmp_path / 'results.csv'
    df.to_csv(path, index=False)
    return str(path)


def test_train_saves_model_pkl(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert os.path.exists(model_path)


def test_train_artifact_contains_required_keys(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    artifact = joblib.load(model_path)
    assert 'model' in artifact
    assert artifact['feature_cols'] == FEATURE_COLS
    assert artifact['n'] == 5
    assert 'league_encoder' in artifact
    assert isinstance(artifact['league_encoder'], LabelEncoder)


def test_train_model_has_predict_proba(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    clf, _ = train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert hasattr(clf, 'predict_proba')


def test_train_returns_accuracy_in_valid_range(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    _, accuracy = train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert 0.0 <= accuracy <= 1.0


def test_train_uses_xgboost(sample_csv, tmp_path):
    from xgboost import XGBClassifier
    model_path = str(tmp_path / 'model.pkl')
    clf, _ = train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert isinstance(clf, XGBClassifier)
```

- [ ] **Step 2: Run tests — verificar que fallan**

```bash
pytest tests/test_train.py -v
```

Expected: `test_train_artifact_contains_required_keys` falla (no hay `league_encoder`), `test_train_uses_xgboost` falla (es RF).

- [ ] **Step 3: Actualizar src/train.py**

Reemplazar el contenido completo de `src/train.py`:

```python
import os
import pandas as pd
from xgboost import XGBClassifier
import joblib
from src.preprocess import build_training_data, FEATURE_COLS


def train(data_path='data/all_matches.csv', model_path='model.pkl', n_estimators=100, n=5):
    if not os.path.exists(data_path):
        data_path = 'data/results.csv'
    df = pd.read_csv(data_path)
    training_df, le = build_training_data(df, n=n)

    X = training_df[FEATURE_COLS]
    y = training_df['result']

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = XGBClassifier(n_estimators=n_estimators, random_state=42, eval_metric='mlogloss')
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")

    joblib.dump({'model': clf, 'feature_cols': FEATURE_COLS, 'n': n, 'league_encoder': le}, model_path)
    print(f"Model saved to {model_path}")
    return clf, accuracy


if __name__ == '__main__':
    train()
```

- [ ] **Step 4: Run tests — verificar que pasan**

```bash
pytest tests/test_train.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: migrate to XGBoost and save league_encoder in artifact"
```

---

## Task 6: Update src/predict.py — league parameter (TDD)

**Files:**
- Modify: `src/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Actualizar tests/test_predict.py**

Reemplazar el contenido completo de `tests/test_predict.py`:

```python
import pandas as pd
import joblib
import pytest
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from src.predict import predict_match, get_team_match_count
from src.preprocess import FEATURE_COLS


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01',
                 '2020-04-01', '2020-05-01'],
        'home_team': ['Brazil', 'Brazil', 'Brazil', 'Argentina', 'Brazil'],
        'away_team': ['Germany', 'France',  'Spain',  'Brazil',   'Italy'],
        'home_score': [3, 2, 1, 0, 2],
        'away_score': [1, 2, 0, 1, 0],
        'neutral': [False] * 5,
        'tournament': ['Friendly'] * 5,
        'league': ['International'] * 5,
        'competition_type': ['international'] * 5,
    })


@pytest.fixture
def mock_model_path(tmp_path):
    le = LabelEncoder()
    le.fit(['International', 'Premier League'])
    clf = XGBClassifier(n_estimators=2, random_state=42, eval_metric='mlogloss')
    X = pd.DataFrame([[0.5] * len(FEATURE_COLS)] * 6, columns=FEATURE_COLS)
    y = ['H', 'D', 'A', 'H', 'D', 'A']
    clf.fit(X, y)
    path = str(tmp_path / 'model.pkl')
    joblib.dump({'model': clf, 'feature_cols': FEATURE_COLS, 'n': 5, 'league_encoder': le}, path)
    return path


def test_predict_match_returns_required_keys(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path)
    assert {'result', 'result_label', 'probabilities', 'home_team', 'away_team'}.issubset(result.keys())


def test_predict_match_result_is_valid_class(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path)
    assert result['result'] in ['H', 'D', 'A']


def test_predict_match_probabilities_sum_to_one(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path)
    assert abs(sum(result['probabilities'].values()) - 1.0) < 1e-6


def test_predict_match_teams_stored_in_result(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path)
    assert result['home_team'] == 'Brazil'
    assert result['away_team'] == 'Germany'


def test_predict_match_with_league_param(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path,
                           league='International', competition_type='international')
    assert result['result'] in ['H', 'D', 'A']


def test_predict_match_unknown_league_does_not_raise(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path,
                           league='Unknown League', competition_type='club')
    assert result['result'] in ['H', 'D', 'A']


def test_get_team_match_count_counts_home_and_away(sample_df):
    assert get_team_match_count(sample_df, 'Brazil') == 5


def test_get_team_match_count_returns_zero_for_unknown(sample_df):
    assert get_team_match_count(sample_df, 'Unknown') == 0
```

- [ ] **Step 2: Run tests — verificar que fallan**

```bash
pytest tests/test_predict.py -v
```

Expected: `test_predict_match_with_league_param` falla — `predict_match` no acepta `league` ni `competition_type`.

- [ ] **Step 3: Actualizar src/predict.py**

Reemplazar el contenido completo de `src/predict.py`:

```python
import functools
import pandas as pd
import joblib
from src.preprocess import build_features_for_prediction, FEATURE_COLS

_RESULT_LABELS = {
    'H': '{home} wins',
    'D': 'Draw',
    'A': '{away} wins',
}


@functools.lru_cache(maxsize=None)
def _load_artifact(model_path):
    return joblib.load(model_path)


def predict_match(home_team, away_team, df, model_path='model.pkl', is_neutral=False,
                  league=None, competition_type=None):
    artifact = _load_artifact(model_path)
    clf = artifact['model']
    n = artifact.get('n', 5)
    feature_cols = artifact['feature_cols']
    label_encoder = artifact.get('league_encoder')

    features = build_features_for_prediction(
        df, home_team, away_team,
        is_neutral=is_neutral, n=n,
        league=league, label_encoder=label_encoder,
        competition_type=competition_type,
    )
    X = pd.DataFrame([features])[feature_cols]

    predicted = clf.predict(X)[0]
    probas = clf.predict_proba(X)[0]

    label = _RESULT_LABELS[predicted].replace('{home}', home_team).replace('{away}', away_team)

    return {
        'result': predicted,
        'result_label': label,
        'probabilities': {cls: float(p) for cls, p in zip(clf.classes_, probas)},
        'home_team': home_team,
        'away_team': away_team,
    }


def get_team_match_count(df, team):
    return int(((df['home_team'] == team) | (df['away_team'] == team)).sum())
```

- [ ] **Step 4: Run tests — verificar que pasan**

```bash
pytest tests/test_predict.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: add league and competition_type params to predict_match"
```

---

## Task 7: Add get_league_stats() to src/stats.py (TDD)

**Files:**
- Modify: `src/stats.py`
- Modify: `tests/test_stats.py`

- [ ] **Step 1: Agregar tests de get_league_stats al final de tests/test_stats.py**

Agregar al final del archivo `tests/test_stats.py` (no reemplazar el contenido existente):

```python
from src.stats import get_league_stats


@pytest.fixture
def league_df():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01',
                 '2020-04-01', '2020-05-01'],
        'home_team': ['Arsenal', 'Arsenal', 'Arsenal', 'Arsenal', 'Chelsea'],
        'away_team': ['Chelsea', 'Liverpool', 'Man City', 'Spurs', 'Arsenal'],
        'home_score': [2, 1, 0, 3, 1],
        'away_score': [1, 1, 2, 0, 0],
        'neutral': [False] * 5,
        'tournament': ['Premier League'] * 5,
        'league': ['Premier League'] * 5,
        'competition_type': ['club'] * 5,
    })


def test_get_league_stats_total_matches(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'Premier League')
    # Arsenal: 4 home + 1 away = 5
    assert stats['total_matches'] == 5


def test_get_league_stats_wins(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'Premier League')
    # Home wins: rows 0 (2-1), 3 (3-0) = 2. Away win: row 4 (1-0 Chelsea home, Arsenal away wins)
    assert stats['wins'] == 3


def test_get_league_stats_returns_league_name(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'Premier League')
    assert stats['league'] == 'Premier League'


def test_get_league_stats_unknown_team_returns_zeros(league_df):
    stats = get_league_stats(league_df, 'Unknown FC', 'Premier League')
    assert stats['total_matches'] == 0
    assert stats['wins'] == 0


def test_get_league_stats_unknown_league_returns_zeros(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'La Liga')
    assert stats['total_matches'] == 0


def test_get_league_stats_win_rate_is_ratio(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'Premier League')
    assert 0.0 <= stats['win_rate'] <= 1.0


def test_get_league_stats_no_league_col_falls_back(league_df):
    df_no_league = league_df.drop(columns=['league'])
    stats = get_league_stats(df_no_league, 'Arsenal', 'Premier League')
    assert stats['total_matches'] == 0
```

- [ ] **Step 2: Run tests — verificar que fallan**

```bash
pytest tests/test_stats.py::test_get_league_stats_total_matches -v
```

Expected: `ImportError: cannot import name 'get_league_stats'`

- [ ] **Step 3: Agregar get_league_stats a src/stats.py**

Agregar al final de `src/stats.py`:

```python
def get_league_stats(df, team, league, n_recent=10):
    if 'league' not in df.columns:
        return {
            'total_matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'win_rate': 0.0, 'goals_scored_per_game': 0.0,
            'goals_conceded_per_game': 0.0, 'form': [], 'league': league,
        }

    league_df = df[df['league'] == league].copy()
    all_m = _all_matches_for_team(league_df, team)

    if len(all_m) == 0:
        return {
            'total_matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'win_rate': 0.0, 'goals_scored_per_game': 0.0,
            'goals_conceded_per_game': 0.0, 'form': [], 'league': league,
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
        'goals_scored_per_game': round(float(all_m['goals_for'].mean()), 2),
        'goals_conceded_per_game': round(float(all_m['goals_against'].mean()), 2),
        'form': recent['result'].tolist(),
        'league': league,
    }
```

- [ ] **Step 4: Run tests — verificar que pasan**

```bash
pytest tests/test_stats.py -v
```

Expected: todos en PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stats.py tests/test_stats.py
git commit -m "feat: add get_league_stats filtered by competition"
```

---

## Task 8: Update app.py — competition selector, League tab, remove emojis

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Leer app.py completo para localizar todas las secciones a cambiar**

```bash
# En el proyecto, leer app.py línea por línea para identificar:
# 1. Todas las ocurrencias de emojis en strings
# 2. La sección de selectboxes de equipos (líneas ~183-206)
# 3. La definición de tabs (línea ~278)
# 4. Las secciones with tab1, tab2, tab3, tab4
```

- [ ] **Step 2: Actualizar imports en app.py**

Agregar `get_league_stats` al import de stats:

```python
from src.stats import (
    get_team_overall_stats,
    get_radar_stats,
    get_head_to_head,
    get_recent_matches,
    get_league_stats,
)
```

- [ ] **Step 3: Actualizar DATA_PATH para usar all_matches.csv si existe**

Reemplazar las constantes al inicio de app.py:

```python
DATA_PATH = 'data/all_matches.csv' if os.path.exists('data/all_matches.csv') else 'data/results.csv'
MODEL_PATH = 'model.pkl'
```

- [ ] **Step 4: Actualizar el header — eliminar emojis**

Reemplazar:
```python
st.markdown("# ⚽ Football Match Predictor")
st.caption("Modelo Random Forest  ·  49 000+ partidos internacionales  ·  1872–2026")
```
Con:
```python
st.markdown("# Football Match Predictor")
caption_text = "XGBoost model  ·  All competitions  ·  1872–2026"
if DATA_PATH == 'data/results.csv':
    caption_text += "  ·  *Run `python -m src.ingest` to add club leagues*"
st.caption(caption_text)
```

- [ ] **Step 5: Agregar selector de competición y lógica de filtrado de equipos**

Localizar la sección de equipos (después de `st.markdown("<div style='margin-bottom:1.2rem'></div>"...)`).

Reemplazar la lógica de `teams` y los selectboxes con:

```python
# Competition selector
WC_LABEL = 'FIFA World Cup'
if 'league' in df.columns:
    league_list = sorted([l for l in df['league'].dropna().unique() if l != WC_LABEL])
    has_wc = WC_LABEL in df['league'].values
    competition_type_map = (
        df.groupby('league')['competition_type'].first().to_dict()
        if 'competition_type' in df.columns else {}
    )
else:
    league_list = []
    has_wc = False
    competition_type_map = {}

if 'selected_league' not in st.session_state:
    st.session_state.selected_league = 'All'

st.markdown(
    '<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
    'letter-spacing:0.8px;color:#444;margin-bottom:8px">Competition</div>',
    unsafe_allow_html=True,
)

# World Cup featured card (shown only when WC data is available)
if has_wc:
    wc_border = '#00d4aa' if st.session_state.selected_league == WC_LABEL else '#1e2130'
    if st.button(
        f"**World Cup** — FIFA World Cup, all editions",
        key='wc_btn',
        use_container_width=True,
    ):
        st.session_state.selected_league = WC_LABEL
        st.rerun()

other_options = ['All'] + league_list
other_idx = other_options.index(st.session_state.selected_league) if st.session_state.selected_league in other_options else 0
chosen = st.selectbox("Other competitions", other_options, index=other_idx, label_visibility='collapsed')
if chosen != st.session_state.selected_league:
    st.session_state.selected_league = chosen
    st.rerun()

selected_league = st.session_state.selected_league

if selected_league == 'All' or 'league' not in df.columns:
    filtered_df = df
    league_param = None
    comp_type_param = None
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

opt_col, btn_col = st.columns([1, 3])
with opt_col:
    is_neutral = st.checkbox(
        "Neutral venue",
        value=True,
        help="Enable for World Cup, Copa America, or other neutral-venue tournaments",
    )
with btn_col:
    predict_btn = st.button("Predict Match", type="primary", use_container_width=True)
```

- [ ] **Step 6: Actualizar llamada a predict_match para pasar league y competition_type**

Localizar la llamada a `predict_match` dentro del bloque `if predict_btn:` y reemplazarla con:

```python
    prediction = predict_match(
        home_team, away_team, df, MODEL_PATH,
        is_neutral=is_neutral,
        league=league_param,
        competition_type=comp_type_param,
    )
```

- [ ] **Step 7: Agregar pestaña League y actualizar definición de tabs**

Localizar la línea que define los tabs:
```python
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯  Predicción", "📊  Estadísticas", "⚔️  Head-to-Head", "📋  Historial"]
    )
```

Reemplazar con:
```python
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Prediction", "Statistics", "Head-to-Head", "History", "League"]
    )
```

- [ ] **Step 8: Eliminar emojis en el contenido de los tabs**

Dentro de `tab1` (Predicción), reemplazar todas las ocurrencias de emojis en strings HTML y labels de métricas:

- `"🏠 {home_team}"` → `f"Home — {home_team}"`
- `"🤝 Empate"` → `"Draw"`
- `"✈️ {away_team}"` → `f"Away — {away_team}"`
- `'🌍 Cancha neutral'` → `'Neutral venue'`  
- `'⚽ Goles/PJ'` → `'Goals / Game'`
- `'🛡️ Recibidos/PJ'` → `'Conceded / Game'`
- `'🏆 Victorias'` → `'Win Rate'`
- `'📊 Partidos'` → `'Matches'`
- `'FORMA RECIENTE'` (sin emoji) → mantener
- `"📋 Comparación de métricas"` → `"Metric Comparison"`
- `"⚽ Goles anotados / PJ"` → `"Goals scored / Game"`
- `"🛡️ Goles recibidos / PJ"` → `"Goals conceded / Game"`
- `"🏆 % Victorias"` → `"Win rate"`
- `"🤝 % Empates"` → `"Draw rate"`
- `"❌ % Derrotas"` → `"Loss rate"`
- `"⚡ Partidos con gol"` → `"Scoring rate"`
- `"🔒 Porterías a cero"` → `"Clean sheet rate"`
- `"📊 Total partidos"` → `"Total matches"`

Hacer lo mismo para cualquier otra aparición de emojis en `tab3` (Head-to-Head) y `tab4` (Historial).

- [ ] **Step 9: Implementar el contenido de tab5 (League)**

Agregar al final del bloque `if predict_btn:`, después de `with tab4:`:

```python
    # ═══════════════ TAB 5 · LEAGUE ════════════════════════════════════════
    with tab5:
        if league_param is None:
            st.info("Select a specific competition above to see league statistics.")
        else:
            st.markdown(f"### {league_param}")
            lc1, lc2 = st.columns(2)
            for col, team in [(lc1, home_team), (lc2, away_team)]:
                with col:
                    lstats = get_league_stats(df, team, league_param)
                    st.markdown(f"**{team}**")
                    if lstats['total_matches'] == 0:
                        st.caption(f"No data for {team} in {league_param}.")
                    else:
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Matches", lstats['total_matches'])
                        m2.metric("Win Rate", f"{lstats['win_rate']*100:.0f}%")
                        m3.metric("Goals / Game", lstats['goals_scored_per_game'])
                        badges = " ".join(_badge(r) for r in lstats['form'][-5:]) or "—"
                        st.markdown(
                            f'<div style="font-size:0.62rem;color:#444;font-weight:700;'
                            f'text-transform:uppercase;letter-spacing:0.5px;margin:10px 0 6px">RECENT FORM</div>'
                            f'{badges}',
                            unsafe_allow_html=True,
                        )
```

- [ ] **Step 10: Lanzar la app y verificar manualmente**

```bash
streamlit run app.py
```

Verificar en el navegador:
1. No hay emojis visibles en ningún label, tab ni botón
2. El selector "Competition" aparece encima de Home/Away
3. Seleccionar "Premier League" filtra los equipos a solo equipos de esa liga (si `all_matches.csv` existe), o muestra todos si no existe
4. Seleccionar "FIFA World Cup" muestra solo selecciones nacionales
5. Al predecir, aparece la pestaña "League" con stats del equipo en la liga seleccionada
6. Si se selecciona "All", la pestaña League muestra el mensaje de selección

- [ ] **Step 11: Commit**

```bash
git add app.py
git commit -m "feat: add competition selector, League tab, and remove emojis from UI"
```

---

## Task 9: Reentrenar el modelo y verificar end-to-end

**Files:**
- Ejecutar ingesta y reentrenamiento

- [ ] **Step 1: Descargar datos de ligas (puede tomar varios minutos)**

```bash
python -m src.ingest
```

Expected: `Saved data/all_matches.csv (NNN total matches)` donde NNN >> 49000.

- [ ] **Step 2: Reentrenar el modelo con el dataset completo**

```bash
python -m src.train
```

Expected: `Test accuracy: 0.XXX` y `Model saved to model.pkl`.

- [ ] **Step 3: Run all tests**

```bash
pytest tests/ -v
```

Expected: todos en PASS.

- [ ] **Step 4: Lanzar app y verificar con datos reales**

```bash
streamlit run app.py
```

Verificar:
1. El caption muestra "XGBoost model  ·  All competitions  ·  1872–2026" (sin el aviso de ingest)
2. El selector Competition muestra todas las ligas descargadas
3. "FIFA World Cup" aparece como segunda opción después de "All"
4. Seleccionar Premier League filtra equipos ingleses
5. Predicción Arsenal vs Chelsea con "Premier League" seleccionado funciona correctamente
6. La pestaña League muestra estadísticas reales

- [ ] **Step 5: Commit final**

```bash
git add model.pkl data/clubs.csv data/all_matches.csv
git commit -m "feat: retrain model with all-leagues dataset"
```

> Nota: si `model.pkl` y los CSV están en `.gitignore`, omitir estos archivos del commit.
