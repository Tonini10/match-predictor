import pandas as pd


def _all_matches_for_team(df, team):
    """Return unified DataFrame of all matches for a team with goals_for/against and result."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])  # exclude future/unplayed matches
    df = df.sort_values('date').reset_index(drop=True)

    home = df[df['home_team'] == team].copy()
    home['goals_for'] = home['home_score']
    home['goals_against'] = home['away_score']
    home['result'] = home.apply(
        lambda r: 'W' if r['home_score'] > r['away_score']
        else ('D' if r['home_score'] == r['away_score'] else 'L'), axis=1
    )

    away = df[df['away_team'] == team].copy()
    away['goals_for'] = away['away_score']
    away['goals_against'] = away['home_score']
    away['result'] = away.apply(
        lambda r: 'W' if r['away_score'] > r['home_score']
        else ('D' if r['away_score'] == r['home_score'] else 'L'), axis=1
    )

    combined = pd.concat([home, away]).sort_values('date').reset_index(drop=True)
    return combined


def get_team_overall_stats(df, team, n_recent=10):
    all_m = _all_matches_for_team(df, team)

    if len(all_m) == 0:
        return {
            'total_matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'win_rate': 0.0, 'draw_rate': 0.0, 'loss_rate': 0.0,
            'goals_scored_per_game': 0.0, 'goals_conceded_per_game': 0.0,
            'clean_sheet_rate': 0.0, 'scoring_rate': 0.0, 'form': [],
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
        'draw_rate': round(draws / total, 3),
        'loss_rate': round(losses / total, 3),
        'goals_scored_per_game': round(float(all_m['goals_for'].mean()), 2),
        'goals_conceded_per_game': round(float(all_m['goals_against'].mean()), 2),
        'clean_sheet_rate': round(float((all_m['goals_against'] == 0).mean()), 3),
        'scoring_rate': round(float((all_m['goals_for'] > 0).mean()), 3),
        'form': recent['result'].tolist(),
    }


def get_radar_stats(df, home_team, away_team):
    hs = get_team_overall_stats(df, home_team, n_recent=10)
    as_ = get_team_overall_stats(df, away_team, n_recent=10)

    def norm_higher(h, a):
        denom = max(h, a, 1e-9)
        return round(h / denom, 3), round(a / denom, 3)

    def norm_lower(h, a):
        # Use global average (1.5 goals/game) as anchor so similar teams don't collapse to 0
        anchor = max(h, a, 1.5)
        return round(max(0.0, 1 - h / anchor), 3), round(max(0.0, 1 - a / anchor), 3)

    home_form = sum(1 for r in hs['form'] if r == 'W') / max(len(hs['form']), 1)
    away_form = sum(1 for r in as_['form'] if r == 'W') / max(len(as_['form']), 1)

    att_h, att_a = norm_higher(hs['goals_scored_per_game'], as_['goals_scored_per_game'])
    def_h, def_a = norm_lower(hs['goals_conceded_per_game'], as_['goals_conceded_per_game'])
    frm_h, frm_a = norm_higher(home_form, away_form)
    con_h, con_a = norm_higher(1 - hs['loss_rate'], 1 - as_['loss_rate'])
    eff_h, eff_a = norm_higher(hs['scoring_rate'], as_['scoring_rate'])
    sol_h, sol_a = norm_higher(hs['clean_sheet_rate'], as_['clean_sheet_rate'])

    dimensions = ['Attack', 'Defense', 'Form', 'Consistency', 'Efficiency', 'Solidity']
    return {
        'dimensions': dimensions,
        'home': [att_h, def_h, frm_h, con_h, eff_h, sol_h],
        'away': [att_a, def_a, frm_a, con_a, eff_a, sol_a],
    }


def get_head_to_head(df, home_team, away_team, last_n=5):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df = df.sort_values('date')

    h2h = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
    ].copy()

    if len(h2h) == 0:
        return {
            'home_wins': 0, 'draws': 0, 'away_wins': 0,
            'home_goals_total': 0, 'away_goals_total': 0,
            'avg_goals_per_game': 0.0, 'last_matches': [],
        }

    home_wins = draws = away_wins = 0
    home_goals = away_goals = 0

    for _, row in h2h.iterrows():
        if row['home_team'] == home_team:
            home_goals += row['home_score']
            away_goals += row['away_score']
            if row['home_score'] > row['away_score']:
                home_wins += 1
            elif row['home_score'] == row['away_score']:
                draws += 1
            else:
                away_wins += 1
        else:
            home_goals += row['away_score']
            away_goals += row['home_score']
            if row['away_score'] > row['home_score']:
                home_wins += 1
            elif row['away_score'] == row['home_score']:
                draws += 1
            else:
                away_wins += 1

    total_goals = home_goals + away_goals
    last_matches = []
    for _, row in h2h.tail(last_n).iterrows():
        last_matches.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': int(row['home_score']),
            'away_score': int(row['away_score']),
            'tournament': str(row.get('tournament', 'N/A')),
        })
    last_matches.reverse()  # most recent first

    return {
        'home_wins': home_wins,
        'draws': draws,
        'away_wins': away_wins,
        'home_goals_total': home_goals,
        'away_goals_total': away_goals,
        'avg_goals_per_game': round(total_goals / len(h2h), 2),
        'last_matches': last_matches,
    }


def get_recent_matches(df, team, n=10):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df = df.sort_values('date')

    home = df[df['home_team'] == team].copy()
    home['opponent'] = home['away_team']
    home['home_away'] = 'Home'
    home['goals_for'] = home['home_score']
    home['goals_against'] = home['away_score']
    home['result'] = home.apply(
        lambda r: 'W' if r['home_score'] > r['away_score']
        else ('D' if r['home_score'] == r['away_score'] else 'L'), axis=1
    )

    away = df[df['away_team'] == team].copy()
    away['opponent'] = away['home_team']
    away['home_away'] = 'Away'
    away['goals_for'] = away['away_score']
    away['goals_against'] = away['home_score']
    away['result'] = away.apply(
        lambda r: 'W' if r['away_score'] > r['home_score']
        else ('D' if r['away_score'] == r['home_score'] else 'L'), axis=1
    )

    combined = pd.concat([home, away]).sort_values('date')
    cols = ['date', 'opponent', 'home_away', 'goals_for', 'goals_against', 'result', 'tournament']
    if n is None:
        recent = combined[cols].copy()
    else:
        recent = combined[cols].tail(n).copy()
    return recent.sort_values('date', ascending=False).reset_index(drop=True)


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
