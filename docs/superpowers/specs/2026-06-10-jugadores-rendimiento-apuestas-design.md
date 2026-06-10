# Jugadores, Rendimiento y Recomendación de Apuesta — Design Spec

**Fecha:** 2026-06-10
**Estado:** Aprobado

---

## Overview

Ampliar match-predictor en tres frentes:

1. **Plantillas y jugadores** — activar las features de calidad de plantilla (ratings FIFA) que el pipeline ya tiene cableadas pero sin datos, y mostrar plantillas en la UI.
2. **Rendimiento reciente** — incorporar las estadísticas de partido que ya existen en `data/raw/` (tiros, tiros al arco, córners, tarjetas) como features de forma del equipo y como sección visual.
3. **Recomendación de apuesta** — tras cada predicción, sugerir el mercado más conveniente según las probabilidades del modelo, con detección opcional de value bets ingresando cuotas manualmente.

Enfoque elegido: **incremental** sobre el pipeline actual (sin refactor de feature store).

---

## 1. Datos de jugadores (plantillas)

- **Nuevo archivo:** `data/players.csv` — dataset FIFA de Kaggle (*FIFA 23 complete player dataset*, archivo `players_22.csv` o equivalente). Columnas requeridas: `short_name`, `player_positions`, `overall`, `age`, `club_name`, `nationality_name`, `shooting`, `pace`, `attacking_finishing`.
- **Colocación manual**, igual que `results.csv`. Si falta, la app funciona con el comportamiento actual y muestra un aviso con instrucciones de descarga.
- Con el archivo presente, las features existentes `team_rating`, `team_attack`, `team_defense` (en `src/player_features.py`) dejan de valer cero y aportan señal real al modelo.
- **Ampliación de `src/player_features.py`:**
  - `get_team_squad(players_df, team_name, n=11)` → DataFrame con el top *n* de jugadores por `overall`: nombre, posición, edad, rating general y **atributos ofensivos individuales** (`shooting`, `attacking_finishing`, `pace`) que reflejan la capacidad de remate de cada jugador. Busca por `club_name` y si no por `nationality_name` (reutilizando `TEAM_NAME_MAP`). Devuelve DataFrame vacío si el equipo no está.

## 2. Rendimiento reciente (stats de partido)

- **`src/ingest.py`:** conservar columnas de stats de los CSVs de football-data.co.uk al construir `all_matches.csv`:
  - `HS`/`AS` → tiros, `HST`/`AST` → tiros al arco, `HC`/`AC` → córners (tiros de esquina), `HY`/`AY` → tarjetas amarillas, `HR`/`AR` → tarjetas rojas (amarillas y rojas **separadas**).
  - Columnas resultantes: `home_shots`, `away_shots`, `home_shots_on_target`, `away_shots_on_target`, `home_corners`, `away_corners`, `home_yellow`, `away_yellow`, `home_red`, `away_red`. `NaN` para partidos sin datos (internacionales).
- **`src/preprocess.py`:** promedios móviles de los últimos 5 partidos por equipo → **10 features nuevas** del modelo:
  - `home_avg_shots`, `home_avg_shots_on_target`, `home_avg_corners`, `home_avg_yellow`, `home_avg_red`
  - `away_avg_shots`, `away_avg_shots_on_target`, `away_avg_corners`, `away_avg_yellow`, `away_avg_red`
  - Imputación a `0.0` cuando no hay datos (partidos internacionales). XGBoost tolera esta imputación.
- **Reentrenar** el modelo y comparar precisión en el split de test contra el modelo actual; reportar la comparación.

## 3. Recomendación de apuesta — `src/betting.py` (nuevo)

### Recomendación base (siempre, tras predecir)

Entrada: dict de probabilidades `{H, D, A}`. Lógica por umbrales:

| Condición | Recomendación | Confianza |
|---|---|---|
| prob. máx ≥ 60% | Apostar al resultado más probable (1, X o 2) | Alta |
| 50% ≤ prob. máx < 60% | Apostar al resultado más probable | Media |
| 40% ≤ prob. máx < 50% | Doble oportunidad (1X o X2) con los dos resultados más probables | Media |
| prob. máx < 40% | **No apostar** — partido impredecible | — |

### Value bet (opcional)

- 3 inputs numéricos para cuotas decimales de la casa del usuario: local (1), empate (X), visitante (2).
- Por mercado: `EV = prob_modelo × cuota − 1`. Se marca con EV positivo (verde) cuando `EV > 0`.
- Si no se ingresan cuotas, la sección de value bet no se calcula.

### Interfaz del módulo

- `recommend(probs: dict) -> dict` — `{market, label, confidence}` según la tabla de umbrales.
- `expected_values(probs: dict, odds: dict) -> dict` — EV por mercado para las cuotas ingresadas.

**Disclaimer visible en la UI:** herramienta estadística; no garantiza resultados ni constituye asesoría financiera.

## 4. UI (`app.py`)

Debajo del resultado de la predicción, en este orden:

1. **Recomendación de apuesta** — tarjeta con la sugerencia (mercado + confianza) y 3 campos opcionales de cuotas; al ingresarlas se muestra el EV por mercado con los positivos en verde. Disclaimer al pie.
2. **Plantillas** (expander) — dos tablas lado a lado con el top 11 de cada equipo: nombre, posición, edad, rating general y atributos ofensivos por jugador (shooting, finishing, pace). Solo si `players.csv` existe y hay datos del equipo.
3. **Comparativa de plantillas** (expander) — barras agrupadas: rating general, ataque y defensa de ambos equipos.
4. **Rendimiento reciente** (expander) — tabla comparativa de promedios de los últimos 5 partidos: tiros, tiros al arco, tiros de esquina, amarillas y rojas. "Sin datos" para equipos sin stats (selecciones).

## 5. Manejo de errores

- `players.csv` ausente → aviso con instrucciones; secciones de plantilla/comparativa ocultas; features de jugadores a cero (comportamiento actual).
- Equipo sin stats de partido → sección de rendimiento muestra "sin datos"; features a 0.
- Cuotas inválidas (≤ 1.0 o vacías) → no se calcula EV para ese mercado.
- `all_matches.csv` generado con versión anterior de `ingest.py` (sin columnas de stats) → preprocess trata las columnas faltantes como `NaN`/0 sin romper.

## 6. Testing

- **`tests/test_betting.py`** (nuevo): umbrales de recomendación (los 4 casos de la tabla), cálculo de EV (positivo, negativo, cuota inválida).
- **`tests/test_preprocess.py`** (ampliar): rolling stats de últimos 5 partidos (incluyendo amarillas y rojas separadas), imputación a 0 sin datos, compatibilidad con CSV sin columnas de stats.
- **`tests/test_player_features.py`** (ampliar): `get_team_squad` — top-n correcto, atributos ofensivos presentes, equipo inexistente → DataFrame vacío, mapeo de nombres.
- Los 79 tests existentes deben seguir pasando.

## Out of Scope

- APIs en vivo de cuotas o alineaciones.
- Stats reales de rendimiento individual por temporada (FBref/Transfermarkt).
- Mercados de apuesta adicionales (over/under, ambos anotan, hándicap).
- Actualización automática del dataset de jugadores.
