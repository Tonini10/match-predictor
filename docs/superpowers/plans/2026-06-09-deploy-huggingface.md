# Deploy en Hugging Face Spaces — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publicar la app en Hugging Face Spaces para acceso público, usando Git LFS para el modelo y los datos.

**Architecture:** El Space en HF es un repo Git independiente que contiene el mismo código que el proyecto local más archivos de configuración HF. Los archivos pesados (model.pkl, CSVs) se gestionan con Git LFS. No se necesitan cambios de código.

**Tech Stack:** Hugging Face Spaces, Git LFS, Streamlit, Python

---

## File Map

| Acción | Archivo | Descripción |
|--------|---------|-------------|
| Create | `.gitattributes` | Configuración de Git LFS para *.pkl y data/*.csv |
| Create | `README.md` | Metadatos del Space (title, sdk, app_file) |

---

## Task 1: Configurar Git LFS en el proyecto local

**Files:**
- Create: `.gitattributes`

- [ ] **Step 1: Instalar Git LFS si no está instalado**

```bash
git lfs install
```

Expected: `Git LFS initialized.`

- [ ] **Step 2: Crear .gitattributes con tracking de archivos pesados**

Crear el archivo `.gitattributes` en la raíz del proyecto con este contenido:

```
*.pkl filter=lfs diff=lfs merge=lfs -text
data/*.csv filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 3: Commitear .gitattributes**

```bash
git add .gitattributes
git commit -m "chore: configure Git LFS for model and data files"
```

---

## Task 2: Crear README.md con metadatos del Space

**Files:**
- Create: `README.md`

- [ ] **Step 1: Crear README.md en la raíz del proyecto**

```markdown
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

# Football Match Predictor

Predicts football match outcomes using XGBoost trained on 234,000+ matches from international competitions and club leagues worldwide.

## Features
- Predict home win / draw / away win with probability breakdown
- Filter by competition (World Cup, Premier League, La Liga, and 15+ more)
- Head-to-head history and team statistics
- League-specific performance stats
```

- [ ] **Step 2: Commitear README.md**

```bash
git add README.md
git commit -m "docs: add Hugging Face Space metadata and README"
```

---

## Task 3: Crear el Space en Hugging Face y hacer push

**Files:** ninguno (operación de Git remoto)

- [ ] **Step 1: Crear el Space en huggingface.co**

1. Ir a https://huggingface.co/new-space
2. Completar:
   - **Owner:** tu usuario de HF
   - **Space name:** `match-predictor`
   - **License:** MIT
   - **SDK:** Streamlit
   - **Visibility:** Public
3. Click "Create Space"
4. HF mostrará la URL del repo: `https://huggingface.co/spaces/<username>/match-predictor`

- [ ] **Step 2: Agregar el Space como remote**

```bash
git remote add hf https://huggingface.co/spaces/<username>/match-predictor
```

Reemplazar `<username>` con tu usuario de Hugging Face.

- [ ] **Step 3: Push al Space (incluye archivos LFS)**

```bash
git push hf master
```

Este push puede tomar varios minutos porque sube model.pkl (~284 MB) y all_matches.csv (~50 MB) via LFS.

Expected: push exitoso, HF inicia el build automáticamente.

- [ ] **Step 4: Verificar el deploy**

1. Ir a `https://huggingface.co/spaces/<username>/match-predictor`
2. Esperar ~3-5 minutos a que el build termine (indicador verde)
3. Hacer una predicción de prueba: seleccionar "Premier League", elegir dos equipos, click "Predict Match"
4. Verificar que aparece resultado con probabilidades y la pestaña League

- [ ] **Step 5: Push también a GitHub para mantener sync**

```bash
git push origin master
```

---

## Task 4: Actualizar el modelo en el futuro

Para cada vez que se reentrene el modelo localmente y se quiera actualizar en producción:

- [ ] **Step 1: Reentrenar localmente**

```bash
python -m src.ingest   # si hay datos nuevos
python -m src.train
```

- [ ] **Step 2: Push del modelo actualizado al Space**

```bash
git add model.pkl
git commit -m "chore: update model"
git push hf master
git push origin master
```

HF redespliega automáticamente en ~2 minutos.
