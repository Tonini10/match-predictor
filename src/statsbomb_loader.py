# -*- coding: utf-8 -*-
"""
statsbomb_loader.py
Carga y cachea datos de StatsBomb Open Data usando statsbombpy.
"""
import warnings
import functools

import pandas as pd

warnings.filterwarnings('ignore')

# Competiciones disponibles confirmadas
COMPETITIONS = {
    'WC 2022': {'competition_id': 43, 'season_id': 106},
    'WC 2018': {'competition_id': 43, 'season_id': 3},
}


def _sb():
    """Importa statsbombpy.sb con warnings suprimidos."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from statsbombpy import sb
        return sb


@functools.lru_cache(maxsize=4)
def get_matches(competition_key):
    """Retorna DataFrame de partidos para una clave de competición del dict COMPETITIONS.

    Args:
        competition_key: clave en COMPETITIONS, ej. 'WC 2022'

    Returns:
        pd.DataFrame con columnas típicas de StatsBomb matches.
    """
    if competition_key not in COMPETITIONS:
        raise ValueError(f"Competición desconocida: {competition_key}. "
                         f"Opciones: {list(COMPETITIONS.keys())}")
    cfg = COMPETITIONS[competition_key]
    sb = _sb()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        matches = sb.matches(
            competition_id=cfg['competition_id'],
            season_id=cfg['season_id'],
        )
    return matches


@functools.lru_cache(maxsize=200)
def get_events(match_id):
    """Retorna DataFrame de eventos para un match_id específico.

    Args:
        match_id: ID numérico del partido.

    Returns:
        pd.DataFrame con todos los eventos del partido.
    """
    sb = _sb()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        events = sb.events(match_id=match_id)
    return events


def get_team_shots(competition_key, team_name):
    """Retorna todos los eventos de tiro de un equipo en una competición.

    Args:
        competition_key: clave en COMPETITIONS.
        team_name: nombre del equipo tal como aparece en StatsBomb.

    Returns:
        pd.DataFrame con columnas:
            x, y          — coordenada del tiro (StatsBomb: 0-120 x, 0-80 y)
            end_x, end_y  — coordenada final del tiro
            xg            — expected goals (shot.statsbomb_xg)
            outcome       — resultado del tiro (Goal, Saved, Blocked, Off T, Post, Wayward)
            player        — nombre del jugador
            minute        — minuto del partido
            match_id      — ID del partido
            opponent      — nombre del rival
        Retorna DataFrame vacío si no se encuentran datos.
    """
    try:
        matches = get_matches(competition_key)
    except Exception:
        return pd.DataFrame()

    # Normaliza nombres de columnas para buscar el equipo
    home_col = 'home_team'
    away_col = 'away_team'

    # Algunos DataFrames de StatsBomb anidan los nombres de equipo
    def _extract_team_name(val):
        if isinstance(val, dict):
            return val.get('home_team_name', val.get('away_team_name', str(val)))
        return str(val)

    rows = []
    for _, match in matches.iterrows():
        # Detecta si el equipo juega en este partido
        home_raw = match.get(home_col, '')
        away_raw = match.get(away_col, '')
        home_name = _extract_team_name(home_raw) if not isinstance(home_raw, str) else home_raw
        away_name = _extract_team_name(away_raw) if not isinstance(away_raw, str) else away_raw

        # Detecta también las columnas anidadas que StatsBomb usa a veces
        if home_name == home_raw:
            home_name = match.get('home_team_name', home_raw)
        if away_name == away_raw:
            away_name = match.get('away_team_name', away_raw)

        if team_name not in (home_name, away_name):
            continue

        opponent = away_name if home_name == team_name else home_name
        mid = match['match_id']

        try:
            events = get_events(mid)
        except Exception:
            continue

        try:
            shots = events[
                (events['type'].apply(
                    lambda v: (v.get('name') if isinstance(v, dict) else v) == 'Shot'
                    if not isinstance(v, str) else v == 'Shot'
                )) &
                (events['team'].apply(
                    lambda v: (v.get('name') if isinstance(v, dict) else v) == team_name
                    if not isinstance(v, str) else v == team_name
                ))
            ].copy()
        except Exception:
            continue

        if shots.empty:
            continue

        for _, shot in shots.iterrows():
            try:
                loc = shot.get('location', None)
                if isinstance(loc, list) and len(loc) >= 2:
                    x, y = loc[0], loc[1]
                else:
                    continue

                end_loc = shot.get('shot_end_location', [None, None])
                end_x = end_loc[0] if (isinstance(end_loc, list) and len(end_loc) >= 1) else None
                end_y = end_loc[1] if (isinstance(end_loc, list) and len(end_loc) >= 2) else None

                xg = shot.get('shot_statsbomb_xg', None)

                outcome_raw = shot.get('shot_outcome', 'Unknown')
                outcome = str(outcome_raw) if outcome_raw else 'Unknown'

                player_raw = shot.get('player', 'Unknown')
                player = str(player_raw) if player_raw else 'Unknown'

                minute = shot.get('minute', None)

                rows.append({
                    'x': x,
                    'y': y,
                    'end_x': end_x,
                    'end_y': end_y,
                    'xg': xg,
                    'outcome': outcome,
                    'player': player,
                    'minute': minute,
                    'match_id': mid,
                    'opponent': opponent,
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(columns=['x', 'y', 'end_x', 'end_y', 'xg',
                                     'outcome', 'player', 'minute', 'match_id', 'opponent'])

    return pd.DataFrame(rows)


def get_team_match_stats(competition_key, team_name):
    """Retorna estadísticas resumen por partido de un equipo.

    Args:
        competition_key: clave en COMPETITIONS.
        team_name: nombre del equipo.

    Returns:
        pd.DataFrame con columnas:
            date, opponent, goals_for, goals_against,
            shots, shots_on_target, xg_for, xg_against, result (W/D/L)
    """
    try:
        matches = get_matches(competition_key)
    except Exception:
        return pd.DataFrame()

    rows = []

    for _, match in matches.iterrows():
        home_raw = match.get('home_team', '')
        away_raw = match.get('away_team', '')
        home_name = match.get('home_team_name', home_raw if isinstance(home_raw, str) else
                              home_raw.get('home_team_name', str(home_raw))
                              if isinstance(home_raw, dict) else str(home_raw))
        away_name = match.get('away_team_name', away_raw if isinstance(away_raw, str) else
                              away_raw.get('away_team_name', str(away_raw))
                              if isinstance(away_raw, dict) else str(away_raw))

        is_home = (home_name == team_name)
        is_away = (away_name == team_name)
        if not is_home and not is_away:
            continue

        opponent = away_name if is_home else home_name
        mid = match['match_id']
        date = match.get('match_date', match.get('date', ''))

        try:
            home_score = int(match.get('home_score', 0) or 0)
            away_score = int(match.get('away_score', 0) or 0)
        except (ValueError, TypeError):
            home_score = away_score = 0

        goals_for = home_score if is_home else away_score
        goals_against = away_score if is_home else home_score

        if goals_for > goals_against:
            result = 'W'
        elif goals_for == goals_against:
            result = 'D'
        else:
            result = 'L'

        shots = 0
        shots_on_target = 0
        xg_for = 0.0
        xg_against = 0.0

        try:
            events = get_events(mid)

            def _is_team(val):
                if isinstance(val, dict):
                    return val.get('name', '') == team_name
                return str(val) == team_name

            def _is_opponent(val):
                if isinstance(val, dict):
                    return val.get('name', '') == opponent
                return str(val) == opponent

            def _is_shot(val):
                if isinstance(val, dict):
                    return val.get('name', '') == 'Shot'
                return str(val) == 'Shot'

            team_mask = events['team'].apply(_is_team)
            opp_mask = events['team'].apply(_is_opponent)
            shot_mask = events['type'].apply(_is_shot)

            team_shots = events[team_mask & shot_mask]
            opp_shots = events[opp_mask & shot_mask]

            shots = len(team_shots)

            def _count_on_target(shot_row):
                outcome = str(shot_row.get('shot_outcome', '') or '')
                return outcome in ('Goal', 'Saved')

            def _get_xg(shot_row):
                return float(shot_row.get('shot_statsbomb_xg', 0.0) or 0.0)

            shots_on_target = sum(1 for _, r in team_shots.iterrows() if _count_on_target(r))
            xg_for = sum(_get_xg(r) for _, r in team_shots.iterrows())
            xg_against = sum(_get_xg(r) for _, r in opp_shots.iterrows())

        except Exception:
            pass

        rows.append({
            'date': str(date),
            'opponent': opponent,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'shots': shots,
            'shots_on_target': shots_on_target,
            'xg_for': round(xg_for, 3),
            'xg_against': round(xg_against, 3),
            'result': result,
        })

    if not rows:
        return pd.DataFrame(columns=['date', 'opponent', 'goals_for', 'goals_against',
                                     'shots', 'shots_on_target', 'xg_for', 'xg_against', 'result'])

    df = pd.DataFrame(rows)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_team_list(competition_key):
    """Retorna lista ordenada de equipos participantes en la competición.

    Args:
        competition_key: clave en COMPETITIONS.

    Returns:
        list[str] con nombres de equipos ordenados alfabéticamente.
    """
    try:
        matches = get_matches(competition_key)
    except Exception:
        return []

    teams = set()
    for _, match in matches.iterrows():
        for col in ('home_team', 'away_team', 'home_team_name', 'away_team_name'):
            val = match.get(col)
            if val is None:
                continue
            if isinstance(val, dict):
                name = val.get('home_team_name', val.get('away_team_name'))
                if name:
                    teams.add(str(name))
            elif isinstance(val, str) and val:
                teams.add(val)

    return sorted(teams)
