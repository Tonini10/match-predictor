# Ratings de Jugadores FIFA como Features del Modelo — Design Spec

**Fecha:** 2026-06-09
**Fuente de datos:** Dataset FIFA/EA FC de Kaggle
**Impacto esperado:** +3-6 puntos de accuracy sobre el 47.7% actual

## Objetivo

Incorporar el rating promedio de los jugadores de cada equipo como features del modelo de predicción, usando datos del dataset FIFA/EA FC disponible en Kaggle.

---

## Fuente de Datos

**Dataset:** "EA Sports FC 25 Complete Player Dataset" o equivalente de Kaggle.
URL de referencia: `https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset`

Columnas relevantes:
```
short_name, long_name, club_name, nationality_name,
overall, potential, value_eur,
pace, shooting, passing, dribbling, defending, physic,
player_positions
```

**Archivo:** `data/players.csv` — descargado manualmente de Kaggle, commiteado en el repo (con Git LFS si supera 50 MB, directamente si no).

---

## Arquitectura

### Nuevo archivo: `src/player_features.py`

Módulo con responsabilidad única: calcular features de jugadores por equipo a partir de `players.csv`.

```
src/player_features.py
    └── get_team_player_features(players_df, team_name) → dict
```

**Mapeo de nombres de equipo:** Los nombres en `players.csv` (ej. `"Manchester City"`) pueden diferir de los de `results.csv` / `all_matches.csv` (ej. `"Man City"`). Se implementa un diccionario de normalización `TEAM_NAME_MAP` en `player_features.py`.

### Features calculadas

| Feature | Cálculo | Fallback si no hay datos |
|---|---|---|
| `home_team_rating` | `mean(overall)` de todos los jugadores del equipo local | `0.0` |
| `away_team_rating` | `mean(overall)` del equipo visitante | `0.0` |
| `home_team_attack` | `(mean(shooting) + mean(dribbling)) / 2` sobre jugadores con posición FW o MF | `0.0` |
| `away_team_attack` | Ídem visitante | `0.0` |
| `home_team_defense` | `(mean(defending) + mean(physic)) / 2` sobre jugadores con posición DF o GK | `0.0` |
| `away_team_defense` | Ídem visitante | `0.0` |
| `rating_diff` | `home_team_rating - away_team_rating` | `0.0` |

Todas las features se normalizan al rango [0, 1] dividiendo entre 100 (ratings FIFA van de 0 a 100).

### FEATURE_COLS actualizado

```python
FEATURE_COLS = [
    # Existentes (11)
    'home_avg_goals_scored', 'home_avg_goals_conceded', 'home_win_rate',
    'away_avg_goals_scored', 'away_avg_goals_conceded', 'away_win_rate',
    'home_is_neutral', 'league_encoded', 'is_international',
    'home_league_win_rate', 'away_league_win_rate',
    # Nuevas (7)
    'home_team_rating', 'away_team_rating',
    'home_team_attack', 'away_team_attack',
    'home_team_defense', 'away_team_defense',
    'rating_diff',
]
```

---

## Cambios por archivo

### `src/player_features.py` (crear)

- `load_players(path='data/players.csv') → pd.DataFrame | None` — carga CSV, retorna None si no existe
- `get_team_player_features(players_df, team_name) → dict` — calcula las 7 features para un equipo; retorna dict de ceros si el equipo no está en el dataset
- `TEAM_NAME_MAP: dict` — diccionario de normalización de nombres (ej. `"Man City"` → `"Manchester City"`)

### `src/preprocess.py`

- `FEATURE_COLS` crece de 11 a 18 features
- `build_training_data(df, n=5, players_df=None)` — nuevo parámetro opcional `players_df`; si se pasa, llama a `get_team_player_features` para cada partido
- `build_features_for_prediction(..., players_df=None)` — ídem; si no hay players_df, features de jugadores = 0.0

### `src/train.py`

- Carga `players_df = load_players()` al inicio
- Lo pasa a `build_training_data(df, n=n, players_df=players_df)`
- Lo guarda en el artifact: `{'model': clf, ..., 'players_df': players_df}`

### `src/predict.py`

- Carga `players_df` del artifact (`artifact.get('players_df')`)
- Lo pasa a `build_features_for_prediction(..., players_df=players_df)`

### `app.py`

- Sin cambios de UI — las features de jugadores son internas al modelo

---

## Compatibilidad hacia atrás

- Si `data/players.csv` no existe → `load_players()` retorna `None` → todas las features de jugadores = `0.0` → el modelo sigue funcionando con las 18 features pero las 7 nuevas valen 0
- El modelo **debe** reentrenarse después de agregar `players.csv` para que las nuevas features tengan peso real
- El modelo anterior (sin features de jugadores) no es compatible con el nuevo `FEATURE_COLS` de 18 features — se requiere reentrenar

---

## Testing

| Archivo | Tests |
|---|---|
| `tests/test_player_features.py` (crear) | `test_load_players_returns_none_if_missing`, `test_get_team_features_returns_dict_with_all_keys`, `test_get_team_features_returns_zeros_for_unknown_team`, `test_team_name_map_normalizes_known_alias` |
| `tests/test_preprocess.py` | Extender: `test_build_training_data_with_players_df_adds_features`, `test_build_training_data_without_players_df_features_are_zero` |
| `tests/test_train.py` | Extender: `test_train_artifact_contains_players_df` |
| `tests/test_predict.py` | Extender: `test_predict_match_uses_players_df_from_artifact` |

---

## Orden de implementación

1. `src/player_features.py` + `tests/test_player_features.py`
2. `src/preprocess.py` — FEATURE_COLS y parámetro `players_df`
3. `src/train.py` — carga y guarda `players_df`
4. `src/predict.py` — usa `players_df` del artifact
5. Descargar `data/players.csv` de Kaggle y reentrenar
6. Verificar mejora de accuracy

---

## Limitaciones conocidas

- **Cobertura:** Los equipos internacionales (selecciones) tienen nombres distintos en FIFA vs el dataset de partidos internacionales. El `TEAM_NAME_MAP` cubrirá los principales (Argentina, Brazil, France, etc.) pero equipos pequeños quedarán con features = 0.
- **Temporalidad:** Los ratings FIFA son de una temporada específica. Partidos históricos usarán ratings actuales (no los de la época del partido). Esto es una aproximación aceptable para el scope actual.
- **Dataset size:** `players.csv` típicamente pesa 5-15 MB — no requiere Git LFS.
