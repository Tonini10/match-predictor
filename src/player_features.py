from pathlib import Path
import pandas as pd

TEAM_NAME_MAP = {
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Spurs': 'Tottenham Hotspur',
    'Newcastle': 'Newcastle United',
    'West Ham': 'West Ham United',
    'Leicester': 'Leicester City',
    'Brighton': 'Brighton & Hove Albion',
    'Wolves': 'Wolverhampton Wanderers',
    'Nottm Forest': 'Nottingham Forest',
    'Atletico Madrid': 'Atlético de Madrid',
    'Betis': 'Real Betis',
    'Celta Vigo': 'Celta de Vigo',
    'USA': 'United States',
    'Korea Republic': 'Korea Republic',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
}

_ATTACK_RE = r'ST|CF|LW|RW|CAM|CM|LM|RM'
_DEFEND_RE = r'CB|LB|RB|LWB|RWB|GK|CDM'


def load_players(path='data/players.csv'):
    """Load FIFA players CSV. Returns None if file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False)


def get_team_player_features(players_df, team_name):
    """Return dict with team_rating, team_attack, team_defense (normalized 0-1).
    Returns zeros if team not found in dataset.
    """
    zeros = {'team_rating': 0.0, 'team_attack': 0.0, 'team_defense': 0.0}
    if players_df is None or len(players_df) == 0:
        return zeros

    normalized = TEAM_NAME_MAP.get(team_name, team_name)

    team = players_df[players_df['club_name'] == normalized]
    if len(team) == 0:
        team = players_df[players_df['nationality_name'] == normalized]
    if len(team) == 0:
        return zeros

    rating = float(team['overall'].mean()) / 100.0

    attackers = team[team['player_positions'].str.contains(_ATTACK_RE, na=False, regex=True)]
    if len(attackers) > 0:
        attack = (float(attackers['shooting'].mean()) + float(attackers['dribbling'].mean())) / 2.0 / 100.0
    else:
        attack = rating

    defenders = team[team['player_positions'].str.contains(_DEFEND_RE, na=False, regex=True)]
    if len(defenders) > 0:
        defense = (float(defenders['defending'].mean()) + float(defenders['physic'].mean())) / 2.0 / 100.0
    else:
        defense = rating

    return {'team_rating': rating, 'team_attack': attack, 'team_defense': defense}


SQUAD_COLS = ['short_name', 'player_positions', 'age', 'overall',
              'shooting', 'attacking_finishing', 'pace']


def get_team_squad(players_df, team_name, n=11):
    """Top n players by overall rating for a club or national team.
    Returns an empty DataFrame when the team or the dataset is missing.
    """
    empty = pd.DataFrame(columns=SQUAD_COLS)
    if players_df is None or len(players_df) == 0:
        return empty

    normalized = TEAM_NAME_MAP.get(team_name, team_name)
    team = players_df[players_df['club_name'] == normalized]
    if len(team) == 0:
        team = players_df[players_df['nationality_name'] == normalized]
    if len(team) == 0:
        return empty

    cols = [c for c in SQUAD_COLS if c in team.columns]
    return team.nlargest(n, 'overall')[cols].reset_index(drop=True)
