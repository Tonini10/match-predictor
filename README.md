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
- Recent performance: shots, shots on target, corners and cards over the last 5 matches
- Bet recommendation (1X2 / double chance / no bet) with optional value-bet
  detection from your bookmaker's odds. Statistical tool only — no guarantees.
- Squad tables and team-quality comparison (requires the optional player dataset below)

## Player data (optional)

Download the [FIFA 23 complete player dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)
from Kaggle (the `players_22.csv` or newer file) and save it as `data/players.csv`.
This enables:

- Squad-quality features in the model (`team_rating`, `team_attack`, `team_defense`)
- Squad tables and team comparison charts in the app

After adding the file, retrain: `python -m src.train`
