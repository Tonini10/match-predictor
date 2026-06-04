# Match Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit web app that predicts football match results (Win/Draw/Win) and probabilities using a Random Forest classifier trained on Kaggle historical data.

**Architecture:** Feature engineering computes rolling team stats (last 5 matches) from a CSV dataset. A RandomForestClassifier is trained chronologically and serialized to `model.pkl`. Streamlit loads both the model and dataset to run live predictions via a dropdown UI.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, joblib, plotly, streamlit, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `data/results.csv` | Raw Kaggle dataset (user places manually) |
| `src/__init__.py` | Makes `src` a Python package |
| `src/preprocess.py` | Feature engineering: rolling team stats + label generation |
| `src/train.py` | Loads CSV, calls preprocess, trains + serializes model |
| `src/predict.py` | Loads model, builds prediction feature vector, returns result + probas |
| `app.py` | Streamlit UI: team dropdowns, predict button, bar chart |
| `tests/test_preprocess.py` | Unit tests for feature engineering |
| `tests/test_train.py` | Tests that training produces a valid serialized artifact |
| `tests/test_predict.py` | Tests for prediction output shape and correctness |
| `requirements.txt` | Python dependencies |
| `pytest.ini` | Sets pythonpath so `from src.x import y` works in tests |

---

## Task 1: Project Setup

**Files:**
- Create: `match-predictor/requirements.txt`
- Create: `match-predictor/pytest.ini`
- Create: `match-predictor/src/__init__.py`
- Create: `match-predictor/tests/__init__.py`
- Create: `match-predictor/data/.gitkeep`

- [ ] **Step 1: Create the project structure**

```
mkdir -p match-predictor/src match-predictor/tests match-predictor/data
touch match-predictor/src/__init__.py
touch match-predictor/tests/__init__.py
touch match-predictor/data/.gitkeep
```

- [ ] **Step 2: Write requirements.txt**

Create `match-predictor/requirements.txt`:

```
streamlit>=1.30.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
plotly>=5.18.0
pytest>=7.0.0
```

- [ ] **Step 3: Write pytest.ini**

Create `match-predictor/pytest.ini`:

```ini
[pytest]
pythonpath = .
```

- [ ] **Step 4: Install dependencies**

```bash
cd match-predictor
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt pytest.ini src/__init__.py tests/__init__.py data/.gitkeep
git commit -m "chore: project scaffolding"
```

---

## Task 2: Feature Engineering (preprocess.py) — TDD

**Files:**
- Create: `match-predictor/tests/test_preprocess.py`
- Create: `match-predictor/src/preprocess.py`

- [ ] **Step 1: Write the failing tests**

Create `match-predictor/tests/test_preprocess.py`:

```python
import pandas as pd
import pytest
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


def test_build_training_data_returns_expected_columns(sample_df):
    result = build_training_data(sample_df)
    assert set(FEATURE_COLS + ['result']).issubset(set(result.columns))


def test_build_training_data_length_matches_input(sample_df):
    result = build_training_data(sample_df)
    assert len(result) == len(sample_df)


def test_build_training_data_result_home_win(sample_df):
    result = build_training_data(sample_df)
    assert result.iloc[0]['result'] == 'H'  # Brazil 3-1 Germany


def test_build_training_data_result_draw(sample_df):
    result = build_training_data(sample_df)
    assert result.iloc[1]['result'] == 'D'  # Brazil 2-2 France


def test_build_training_data_result_away_win(sample_df):
    result = build_training_data(sample_df)
    assert result.iloc[3]['result'] == 'A'  # Argentina 0-1 Brazil


def test_build_training_data_first_match_stats_are_zero(sample_df):
    result = build_training_data(sample_df)
    assert result.iloc[0]['home_avg_goals_scored'] == 0.0
    assert result.iloc[0]['home_win_rate'] == 0.0


def test_build_training_data_second_match_uses_first(sample_df):
    # Row 1 is Brazil's second home match; prior home match (row 0) scored 3
    result = build_training_data(sample_df)
    assert result.iloc[1]['home_avg_goals_scored'] == pytest.approx(3.0)


def test_build_features_for_prediction_returns_all_keys(sample_df):
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany')
    assert set(features.keys()) == set(FEATURE_COLS)


def test_build_features_for_prediction_neutral_flag(sample_df):
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', is_neutral=True)
    assert features['home_is_neutral'] == 1


def test_build_features_for_prediction_unknown_team_returns_zeros(sample_df):
    features = build_features_for_prediction(sample_df, 'Unknown', 'Brazil')
    assert features['home_avg_goals_scored'] == 0.0
    assert features['home_win_rate'] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd match-predictor
pytest tests/test_preprocess.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.preprocess'`

- [ ] **Step 3: Implement src/preprocess.py**

Create `match-predictor/src/preprocess.py`:

```python
import pandas as pd

FEATURE_COLS = [
    'home_avg_goals_scored',
    'home_avg_goals_conceded',
    'home_win_rate',
    'away_avg_goals_scored',
    'away_avg_goals_conceded',
    'away_win_rate',
    'home_is_neutral',
]


def build_training_data(df, n=5):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['away_win'] = (df['away_score'] > df['home_score']).astype(int)

    def rolling_mean(group_col, val_col):
        return df.groupby(group_col)[val_col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean().fillna(0)
        )

    df['home_avg_goals_scored']   = rolling_mean('home_team', 'home_score')
    df['home_avg_goals_conceded'] = rolling_mean('home_team', 'away_score')
    df['home_win_rate']           = rolling_mean('home_team', 'home_win')
    df['away_avg_goals_scored']   = rolling_mean('away_team', 'away_score')
    df['away_avg_goals_conceded'] = rolling_mean('away_team', 'home_score')
    df['away_win_rate']           = rolling_mean('away_team', 'away_win')
    df['home_is_neutral']         = df['neutral'].astype(int)

    df['result'] = df.apply(
        lambda r: 'H' if r['home_score'] > r['away_score']
                  else ('D' if r['home_score'] == r['away_score'] else 'A'),
        axis=1,
    )

    return df[FEATURE_COLS + ['result']].copy()


def build_features_for_prediction(df, home_team, away_team, is_neutral=False, n=5):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    hm = df[df['home_team'] == home_team].tail(n)
    am = df[df['away_team'] == away_team].tail(n)

    def safe_mean(series):
        return float(series.mean()) if len(series) > 0 else 0.0

    return {
        'home_avg_goals_scored':   safe_mean(hm['home_score']),
        'home_avg_goals_conceded': safe_mean(hm['away_score']),
        'home_win_rate':           safe_mean((hm['home_score'] > hm['away_score']).astype(float)),
        'away_avg_goals_scored':   safe_mean(am['away_score']),
        'away_avg_goals_conceded': safe_mean(am['home_score']),
        'away_win_rate':           safe_mean((am['away_score'] > am['home_score']).astype(float)),
        'home_is_neutral':         int(is_neutral),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_preprocess.py -v
```

Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: feature engineering with rolling team stats"
```

---

## Task 3: Model Training (train.py) — TDD

**Files:**
- Create: `match-predictor/tests/test_train.py`
- Create: `match-predictor/src/train.py`

- [ ] **Step 1: Write the failing tests**

Create `match-predictor/tests/test_train.py`:

```python
import os
import joblib
import pandas as pd
import pytest
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


def test_train_artifact_contains_model_and_feature_cols(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    artifact = joblib.load(model_path)
    assert 'model' in artifact
    assert artifact['feature_cols'] == FEATURE_COLS


def test_train_model_has_predict_proba(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    clf, _ = train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert hasattr(clf, 'predict_proba')


def test_train_returns_accuracy_in_valid_range(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    _, accuracy = train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert 0.0 <= accuracy <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.train'`

- [ ] **Step 3: Implement src/train.py**

Create `match-predictor/src/train.py`:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.preprocess import build_training_data, FEATURE_COLS


def train(data_path='data/results.csv', model_path='model.pkl', n_estimators=100):
    df = pd.read_csv(data_path)
    training_df = build_training_data(df)

    X = training_df[FEATURE_COLS]
    y = training_df['result']

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")

    joblib.dump({'model': clf, 'feature_cols': FEATURE_COLS}, model_path)
    print(f"Model saved to {model_path}")
    return clf, accuracy


if __name__ == '__main__':
    train()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: random forest training with chronological split"
```

---

## Task 4: Prediction Inference (predict.py) — TDD

**Files:**
- Create: `match-predictor/tests/test_predict.py`
- Create: `match-predictor/src/predict.py`

- [ ] **Step 1: Write the failing tests**

Create `match-predictor/tests/test_predict.py`:

```python
import pandas as pd
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
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
    })


@pytest.fixture
def mock_model_path(tmp_path):
    clf = RandomForestClassifier(n_estimators=2, random_state=42)
    X = pd.DataFrame([[0.5] * len(FEATURE_COLS)] * 6, columns=FEATURE_COLS)
    y = ['H', 'D', 'A', 'H', 'D', 'A']
    clf.fit(X, y)
    path = str(tmp_path / 'model.pkl')
    joblib.dump({'model': clf, 'feature_cols': FEATURE_COLS}, path)
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


def test_get_team_match_count_counts_home_and_away(sample_df):
    # Brazil: home in rows 0,1,2,4 and away in row 3 = 5 total
    assert get_team_match_count(sample_df, 'Brazil') == 5


def test_get_team_match_count_returns_zero_for_unknown(sample_df):
    assert get_team_match_count(sample_df, 'Unknown') == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_predict.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.predict'`

- [ ] **Step 3: Implement src/predict.py**

Create `match-predictor/src/predict.py`:

```python
import pandas as pd
import joblib
from src.preprocess import build_features_for_prediction, FEATURE_COLS

_RESULT_LABELS = {
    'H': '{home} gana',
    'D': 'Empate',
    'A': '{away} gana',
}


def predict_match(home_team, away_team, df, model_path='model.pkl'):
    artifact = joblib.load(model_path)
    clf = artifact['model']

    features = build_features_for_prediction(df, home_team, away_team)
    X = pd.DataFrame([features])[FEATURE_COLS]

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

- [ ] **Step 4: Run all tests to verify everything passes**

```bash
pytest -v
```

Expected: `19 passed`

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: prediction inference with probabilities"
```

---

## Task 5: Streamlit UI (app.py)

**Files:**
- Create: `match-predictor/app.py`

- [ ] **Step 1: Implement app.py**

Create `match-predictor/app.py`:

```python
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
default_away = teams.index('Argentina') if 'Argentina' in teams else 1

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

        prediction = predict_match(home_team, away_team, df, MODEL_PATH)

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
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: streamlit UI with team dropdowns and probability chart"
```

---

## Task 6: Download Dataset, Train, and Run

- [ ] **Step 1: Download the dataset from Kaggle**

Go to: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Download `results.csv` and place it at `match-predictor/data/results.csv`.

- [ ] **Step 2: Train the model**

```bash
cd match-predictor
python -m src.train
```

Expected output:
```
Test accuracy: 0.XXX
Model saved to model.pkl
```

Accuracy around 0.48–0.55 is normal for football prediction (the sport is inherently unpredictable).

- [ ] **Step 3: Launch the app**

```bash
streamlit run app.py
```

Expected: browser opens at `http://localhost:8501` showing the predictor UI.

- [ ] **Step 4: Verify the golden path**

1. Select two different teams from the dropdowns
2. Click "Predecir"
3. Verify: result label appears, bar chart shows 3 bars summing to ~100%

- [ ] **Step 5: Verify edge case — same team**

1. Set both dropdowns to the same team
2. Click "Predecir"
3. Verify: red error message "Selecciona dos equipos diferentes."

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "feat: complete match predictor with trained model"
```
