# Deploy en Hugging Face Spaces — Design Spec

**Fecha:** 2026-06-09
**Plataforma:** Hugging Face Spaces (Streamlit)
**Estrategia de datos:** Git LFS para archivos pesados

## Objetivo

Hacer la app de predicción de partidos accesible públicamente sin costo, usando Hugging Face Spaces con Git LFS para servir el modelo y los datos directamente desde el repositorio.

---

## Arquitectura

Hugging Face Spaces ejecuta la app como un contenedor con Python + Streamlit. El Space tiene su propio repositorio Git (en hf.co) que se puede sincronizar con el repo de GitHub. Los archivos pesados se gestionan con Git LFS, que HF soporta nativamente.

```
GitHub repo (código fuente)
    └── sincronizado manualmente o via CI
HF Space repo (deployment)
    ├── app.py
    ├── src/
    ├── requirements.txt
    ├── README.md          ← metadatos del Space (title, sdk, etc.)
    ├── model.pkl          ← Git LFS (~284 MB)
    └── data/
        ├── results.csv    ← Git LFS (~4 MB)
        └── all_matches.csv ← Git LFS (~50 MB)
```

---

## Archivos a crear / modificar

| Acción | Archivo | Descripción |
|--------|---------|-------------|
| Create | `README.md` (en el Space) | Metadatos HF: title, emoji, sdk=streamlit, app_file=app.py |
| Create | `.gitattributes` | Rutas de LFS: `*.pkl`, `data/*.csv` |
| No change | `app.py`, `src/`, `tests/` | Sin cambios de código |
| No change | `requirements.txt` | Ya tiene todas las dependencias |

---

## Proceso de setup

### Paso 1: Crear el Space en Hugging Face

1. Ir a huggingface.co → New Space
2. Nombre: `match-predictor`
3. SDK: Streamlit
4. Visibilidad: Public
5. Clonar el repo del Space localmente:
   ```bash
   git clone https://huggingface.co/spaces/<username>/match-predictor
   ```

### Paso 2: Configurar Git LFS

```bash
cd match-predictor
git lfs install
git lfs track "*.pkl"
git lfs track "data/*.csv"
git add .gitattributes
```

### Paso 3: Copiar el código del proyecto

Copiar todos los archivos del proyecto local al directorio del Space (excepto `data/raw/`, `.superpowers/`, `venv/`).

### Paso 4: Crear README.md con metadatos del Space

```yaml
---
title: Football Match Predictor
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---
```

### Paso 5: Commit y push

```bash
git add .
git commit -m "feat: initial deploy to Hugging Face Spaces"
git push
```

HF despliega automáticamente en ~3-5 minutos.

---

## Actualizaciones futuras

Para actualizar el modelo después de reentrenar:
```bash
cd <hf-space-dir>
cp ../match-predictor/model.pkl .
git add model.pkl
git commit -m "chore: update model"
git push
```

HF redespliega en ~2 minutos.

---

## Limitaciones conocidas

- **RAM:** HF Spaces gratuito tiene 16 GB RAM — suficiente para el modelo y los datos.
- **CPU:** 2 vCPU gratis — predicciones toman <1 segundo.
- **Sleep:** Los Spaces gratuitos se "duermen" tras 48h sin actividad. El primer request después tarda ~30s en despertar.
- **Storage:** Git LFS gratuito en HF tiene límite de 10 GB — bien dentro del límite.

---

## Sin cambios al código

`app.py` ya detecta `data/all_matches.csv` con `os.path.exists()` y cae back a `data/results.csv`. No se necesita ningún cambio de código para el deploy.
