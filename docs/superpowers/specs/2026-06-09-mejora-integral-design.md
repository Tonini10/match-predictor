# Mejora Integral — Match Predictor

**Fecha:** 2026-06-09
**Enfoque elegido:** Balanceado (Opción B)

## Objetivo

Expandir el proyecto de predicción de partidos internacionales a todas las ligas disponibles (clubes + internacional), mejorar el modelo con nuevas features y XGBoost, y actualizar la UI para que sea profesional, sin emojis, con filtro por competición y sección destacada para el Mundial.

---

## Sección 1 — Pipeline de Datos

### Fuente

**football-data.co.uk** — CSVs gratuitos por liga/temporada, sin API key, compatibles con el pipeline actual. Se descargan manualmente y se actualizan por temporada.

### Nuevos archivos

| Archivo | Descripción |
|---|---|
| `src/ingest.py` | Descarga, normaliza y unifica CSVs de todas las ligas |
| `data/clubs.csv` | Dataset normalizado de partidos de clubes (generado) |
| `data/all_matches.csv` | Dataset combinado: `results.csv` + `clubs.csv` (generado) |
| `data/raw/` | CSVs originales descargados — ignorados por git |

### Schema unificado

El dataset `all_matches.csv` extiende el schema actual con dos columnas nuevas:

```
date, home_team, away_team, home_score, away_score, tournament,
neutral, league, competition_type
```

- `league` — nombre de la competición: `"Premier League"`, `"La Liga"`, `"International"`, `"FIFA World Cup"`, etc.
- `competition_type` — `"club"` o `"international"`

### Flujo de ingesta

```
python -m src.ingest
```

1. Descarga CSVs de football-data.co.uk por liga/temporada
2. Normaliza columnas al schema unificado (renombra `FTHG`→`home_score`, etc.)
3. Agrega columnas `league` y `competition_type`
4. Combina con `results.csv` existente → produce `all_matches.csv`
5. El dataset internacional original (`results.csv`) no se modifica

### Manejo de errores en datos

- Si `all_matches.csv` no existe al arrancar la app → usa `results.csv` + `model.pkl` existente sin cambios, muestra aviso en UI indicando que hay que ejecutar `src/ingest.py` y reentrenar para activar ligas de clubes. El modelo reentrenado con `all_matches.csv` no es compatible con `results.csv` y viceversa — siempre deben estar sincronizados.
- Columnas faltantes en un CSV de liga → se omite esa liga con warning en consola

---

## Sección 2 — Modelo Predictivo

### Cambio de algoritmo

`RandomForestClassifier` → `XGBClassifier` (XGBoost). La interfaz es compatible con scikit-learn; el cambio es mínimo en `src/train.py`. Se agrega `xgboost` a `requirements.txt`.

### Features nuevas

| Feature | Descripción |
|---|---|
| `league_encoded` | Liga codificada como entero (label encoding, guardado en artifact) |
| `competition_type` | `0` = club, `1` = internacional |
| `home_league_win_rate` | Win rate del equipo local en los últimos N partidos **de esta liga** |
| `away_league_win_rate` | Win rate del equipo visitante en los últimos N partidos **de esta liga** |

Las features existentes se mantienen sin cambios.

### Cambios por archivo

- **`src/preprocess.py`** — agrega label encoding de `league`, calcula rolling stats filtradas por liga además de las globales, actualiza `FEATURE_COLS`
- **`src/train.py`** — reemplaza `RandomForestClassifier` por `XGBClassifier`, guarda el `LabelEncoder` de liga en el artifact `model.pkl`
- **`src/predict.py`** — recibe `league` como parámetro, aplica el encoder guardado para construir features correctas
- **`model.pkl`** — no es backwards compatible con el modelo actual; se requiere reentrenar tras la ingesta

### Compatibilidad

`predict_match()` recibe `league` como parámetro opcional. Si no se pasa (o el equipo no tiene historial en esa liga), las features de liga se calculan en cero y la predicción usa solo historial global — sin error.

---

## Sección 3 — UI / UX

### Principios de diseño

- Sin emojis en ningún elemento de la interfaz
- Labels en inglés, tipografía limpia con Inter
- Nombres de equipos: fuente grande (1.1rem), bold, con la liga en gris sutil debajo
- Estilo oscuro profesional existente se mantiene

### Selector de competición

Encima del selector de equipos. Implementado con pills/chips horizontales. La competición `"World Cup"` aparece como tarjeta destacada separada (estilo FEATURED con borde teal) por encima del resto de las ligas.

Al seleccionar una competición:
- El dropdown de equipos filtra automáticamente solo los equipos con partidos en esa liga
- El parámetro `league` se pasa a `predict_match()`

### Selector de equipos

- Label "Home" / "Away" en texto uppercase pequeño
- Nombre del equipo en tipografía grande y bold
- Nombre de la liga del partido mostrado en gris debajo del equipo

### Nueva pestaña — League

Quinta pestaña en el resultado. Muestra estadísticas del equipo filtradas por la competición seleccionada:

- **World Cup:** títulos, partidos en dataset, win rate en Mundial, goles por partido en Mundial
- **Ligas de club:** partidos en dataset en esa liga, win rate local, goles por partido
- **Internacional general:** mismas métricas sobre partidos internacionales

### Cambios en app.py

- Nuevo selector de competición con card destacada para World Cup
- Equipos filtrados según liga seleccionada
- `predict_match()` recibe `league`
- Nueva pestaña "League" con stats por competición
- Pestaña "History" filtra partidos por liga seleccionada
- Todos los emojis existentes en la UI se eliminan

---

## Sección 4 — Testing

### Nuevos tests

| Archivo | Qué cubre |
|---|---|
| `tests/test_ingest.py` | Normalización de CSVs, manejo de columnas faltantes, ligas desconocidas, combinación con dataset internacional |

### Tests extendidos

| Archivo | Extensión |
|---|---|
| `tests/test_preprocess.py` | Nuevas features: `league_encoded`, `competition_type`, rolling stats por liga |
| `tests/test_train.py` | XGBoost, nuevo set de features, artifact contiene LabelEncoder |
| `tests/test_predict.py` | `predict_match()` con parámetro `league`, fallback cuando no hay historial en liga |

### Manejo de errores en runtime

- Equipo con menos de 5 partidos en la liga seleccionada → warning en UI (patrón existente con `get_team_match_count`, extendido por liga)
- Equipo sin historial en la liga → features de liga en cero, predicción usa historial global, sin error

---

## Orden de implementación

1. `src/ingest.py` + tests → produce `all_matches.csv`
2. `src/preprocess.py` — nuevas features + label encoding
3. `src/train.py` — migración a XGBoost
4. `src/predict.py` — parámetro `league`
5. `src/stats.py` — stats filtradas por liga
6. `app.py` — selector de competición, filtro de equipos, nueva pestaña League, eliminación de emojis
7. Tests completos

---

## Dependencias nuevas

```
xgboost>=2.0.0
```
