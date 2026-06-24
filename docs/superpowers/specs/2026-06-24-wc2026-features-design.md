# Mundial 2026 + Features + Over/Under + UI — Design Spec

**Date:** 2026-06-24  
**Scope:** Four coordinated enhancements to the Football Match Predictor

---

## 1. World Cup 2026 Data (football-data.org API)

### Goal
Automatically pull live FIFA World Cup 2026 match results into `data/all_matches.csv` so predictions reflect current tournament form.

### Architecture

**New file: `src/ingest_api.py`**

Responsibilities:
- `fetch_wc_matches(api_key) → pd.DataFrame` — calls `GET https://api.football-data.org/v4/competitions/WC/matches`, filters out unplayed matches (no fullTime score), normalizes to unified schema
- `normalize_api_match(match_dict) → dict` — maps football-data.org JSON fields to internal schema columns

Normalized schema (same as `all_matches.csv`):

| Column | Source | Notes |
|--------|--------|-------|
| `date` | `utcDate` | parse to datetime |
| `home_team` | `homeTeam.name` | |
| `away_team` | `awayTeam.name` | |
| `home_score` | `score.fullTime.home` | int |
| `away_score` | `score.fullTime.away` | int |
| `tournament` | `"FIFA World Cup 2026"` | constant |
| `league` | `"FIFA World Cup 2026"` | constant |
| `competition_type` | `"international"` | constant |
| `neutral` | `True` | all WC matches at neutral venue |
| stat columns | `NaN` | free tier has no shot/corner data |

**Integration with `src/ingest.py` `__main__`:**

After building `combined = combine_datasets(clubs, ...)`, check for `FOOTBALL_DATA_API_KEY` env var:
```python
api_key = os.environ.get('FOOTBALL_DATA_API_KEY')
if api_key:
    from src.ingest_api import fetch_wc_matches
    wc = fetch_wc_matches(api_key)
    combined = pd.concat([combined, wc], ignore_index=True).sort_values('date').reset_index(drop=True)
    combined = combined.drop_duplicates(subset=['date','home_team','away_team'], keep='last')
```

**API key setup (user instruction):**
1. Registrarse gratis en https://www.football-data.org/client/register
2. El API key llega al email registrado
3. Set env var: `$env:FOOTBALL_DATA_API_KEY="your_key"` (PowerShell) or add to `.env`
4. Run `python -m src.ingest data/raw`

### Error handling
- If API call fails (network error, invalid key, rate limit): print warning and continue without WC data — never crashes the ingest pipeline
- Rate limit: free tier = 10 req/min; one call per ingest run is well within limits

### Tests (`tests/test_ingest_api.py`)
- `test_fetch_wc_matches_normalizes_response` — mock `urllib.request.urlopen`, verify schema columns
- `test_fetch_wc_matches_skips_unplayed` — match with `score.fullTime.home = null` is excluded
- `test_fetch_wc_matches_api_error_returns_empty` — network error → empty DataFrame, no exception

---

## 2. New Model Features + Second Model (Over/Under 2.5)

### New features in `src/preprocess.py`

Six new features added to `FEATURE_COLS` (appended at end):

| Feature | Formula | Notes |
|---------|---------|-------|
| `home_weighted_form` | `ewm(span=5, min_periods=1)` on `home_win` with `shift(1)` | Recent wins count more |
| `away_weighted_form` | same, on `away_win` | |
| `home_conversion_rate` | rolling mean of `home_score / home_shots_on_target` (last 5) | 0 when shots_on_target=0 |
| `away_conversion_rate` | rolling mean of `away_score / away_shots_on_target` (last 5) | |
| `home_def_solidity` | rolling mean of `away_score / away_shots_on_target` (last 5) | Goals conceded per shot on target faced |
| `away_def_solidity` | rolling mean of `home_score / home_shots_on_target` (last 5) | |

Implementation in `build_training_data`:
```python
# Weighted form (ewm)
df['home_weighted_form'] = df.groupby('home_team')['home_win'].transform(
    lambda x: x.shift(1).ewm(span=5, min_periods=1).mean().fillna(0)
)
# Conversion rate
df['home_conv_per_match'] = df['home_score'] / df['home_shots_on_target'].replace(0, float('nan'))
df['home_conversion_rate'] = df.groupby('home_team')['home_conv_per_match'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(0)
)
```

Same pattern in `build_features_for_prediction` using `stat_avg()` helper already present.

When `home_shots_on_target` column is absent (international matches without stats): all 6 new features default to 0.0 — same fallback as existing stat features.

### Second model: `model_ou.pkl`

**Target:** `over_2.5 = (home_score + away_score) > 2.5` → binary (1=over, 0=under)

**Changes to `src/train.py`:**
- After training the result model, train a second `XGBClassifier` on the same `X_train`/`X_test` with `y_ou` target
- Save as `model_ou.pkl`
- Print: `Test accuracy (result): 0.XXX` and `Test accuracy (over/under): 0.XXX`
- Both models use identical hyperparameters (same grid/defaults)

**Changes to `src/predict.py`:**
- `predict_match()` loads `model_ou.pkl` alongside `model.pkl`
- Returns additional key: `over_under_prob` (float, probability of over 2.5)
- If `model_ou.pkl` doesn't exist (first run before retrain): `over_under_prob = None`

### Tests
- `tests/test_preprocess.py` — 3 new cases: weighted form > simple win_rate for recent streak; conversion_rate correct when shots_on_target present; all 6 features = 0.0 when stat columns missing
- `tests/test_predict.py` — `over_under_prob` key present in result; value between 0 and 1

---

## 3. Combined Betting Recommendation

### New function in `src/betting.py`

```python
def recommend_combined(result_probs, ou_prob):
    """
    Returns:
      result_rec: output of existing recommend()
      ou_rec: {'market': 'Over 2.5'|'Under 2.5'|None, 'prob': float|None}
      combined_label: human-readable string
    """
```

Over/under thresholds:
- `ou_prob >= 0.60` → recommend Over 2.5
- `ou_prob <= 0.40` → recommend Under 2.5
- `0.40 < ou_prob < 0.60` → no recommendation (too close to call)

`combined_label` examples (all UI text in Spanish):
- `"Local gana (1)  ·  Más de 2.5 goles"` (both confident)
- `"Doble oportunidad 1X  ·  Menos de 2.5 goles"`
- `"Local gana (1)  ·  Goles inciertos"`
- `"Sin recomendación  ·  Más de 2.5 goles"`

All labels, card headers, metric names, captions, and expander titles in `app.py` in Spanish.

### Changes to `app.py`

**Odds expander** — add two new inputs after the existing three:
```
[ Home (1) ] [ Draw (X) ] [ Away (2) ] [ Over 2.5 ] [ Under 2.5 ]
```

**Bet recommendation card** (Tab 1, after probability bars):

```
┌─────────────────────────────────────────────────────────┐
│  BET RECOMMENDATION                                      │
│  ┌──────────────────────┐  ┌──────────────────────────┐ │
│  │ RESULTADO            │  │ GOLES                    │ │
│  │ Local gana (1)       │  │ Over 2.5                 │ │
│  │ HIGH CONFIDENCE      │  │ 68% probabilidad         │ │
│  └──────────────────────┘  └──────────────────────────┘ │
│  → Apostar: Local gana (1)  ·  Over 2.5                 │
│                                                          │
│  [EV Home +12%] [EV Over +8%]  ← only if odds entered  │
└─────────────────────────────────────────────────────────┘
```

`over_under_prob` displayed as metric: `"Over 2.5"` with `68.3%` value in Tab 1 alongside the existing H/D/A metrics (becomes a 4th metric column or second row).

### Tests (`tests/test_betting.py`)
- `test_recommend_combined_both_confident` — result high + ou >= 0.55
- `test_recommend_combined_no_result_has_ou` — result None + ou recommendation
- `test_recommend_combined_grey_zone_ou` — ou between 0.40–0.60 → ou_rec market is None
- `test_recommend_combined_ou_none_skipped` — `ou_prob=None` → ou_rec is None, no crash

---

## 4. UI — Comparative Bars + Sparklines

### New function in `src/stats.py`

```python
def get_stat_sparklines(df, team, n=5) -> dict[str, list[float]]:
    """
    Returns last n match values per stat for the team (home or away perspective).
    Chronological order (oldest first). Empty list if column missing.
    Keys: 'shots', 'shots_on_target', 'corners', 'yellow', 'red'
    """
```

Logic: same filtering as `get_recent_performance` but returns per-match values instead of averages.

### Changes to `app.py` — "Recent performance" expander

Replace the current `st.dataframe` table with two Plotly charts:

**Chart 1 — Grouped bar (height ~220px):**
- X axis: `['Shots', 'Shots on Target', 'Corners', 'Yellow Cards']`
- Two bar groups: home team (`#00d4aa`) vs away team (`#f43f5e`)
- Values: averages from `get_recent_performance()` (already computed)
- Only rendered if at least one team has non-None values

**Chart 2 — 2×2 sparkline subplots (height ~200px):**
- Subplot titles: Shots, Shots on Target, Corners, Yellow Cards
- Each subplot: two lines (home `#00d4aa`, away `#f43f5e`) over last 5 matches
- X axis: match index 1–5 (oldest to newest), no labels
- Y axis: minimal, no grid clutter
- Data from `get_stat_sparklines()` for each team
- If a team has < 2 data points for a stat, that line is omitted from the subplot

**Fallback:** if all stats are None for both teams, show only the caption ("stats available for club leagues only").

### Tests (`tests/test_stats.py`)
- `test_get_stat_sparklines_returns_chronological_order`
- `test_get_stat_sparklines_empty_when_column_missing`
- `test_get_stat_sparklines_handles_nan_values` — NaN entries excluded from list

---

## File Change Summary

| File | Action |
|------|--------|
| `src/ingest_api.py` | **Create** |
| `src/ingest.py` | Modify `__main__` to call API if key present |
| `src/preprocess.py` | Add 6 features + `FEATURE_COLS` entries |
| `src/train.py` | Train + save `model_ou.pkl` |
| `src/predict.py` | Load `model_ou.pkl`, return `over_under_prob` |
| `src/betting.py` | Add `recommend_combined()` |
| `src/stats.py` | Add `get_stat_sparklines()` |
| `app.py` | Over/under metric, combined rec card, sparkline charts, odds inputs |
| `tests/test_ingest_api.py` | **Create** |
| `tests/test_preprocess.py` | Add 3 cases |
| `tests/test_predict.py` | Add 2 cases |
| `tests/test_betting.py` | Add 4 cases |
| `tests/test_stats.py` | Add 3 cases |

## Dependency / Order

1. `ingest_api.py` + `ingest.py` update → regenerate `all_matches.csv`
2. `preprocess.py` new features → tests pass
3. `train.py` second model → `model_ou.pkl` generated
4. `predict.py` loads both models
5. `betting.py` `recommend_combined`
6. `stats.py` `get_stat_sparklines`
7. `app.py` UI (depends on all above)
