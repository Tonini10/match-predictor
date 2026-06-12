from pathlib import Path
import pandas as pd

TEAM_NAME_MAP = {
    'AZ Alkmaar': 'AZ',
    'Accrington': 'Accrington Stanley',
    'Ad. Demirspor': 'Adana Demirspor',
    'Alaves': 'Deportivo Alavés',
    'Almeria': 'Almería',
    'Amiens': 'Amiens SC',
    'Angers': 'Angers SCO',
    'Ankaragucu': 'Ankaragücü',
    'Ath Bilbao': 'Athletic Club',
    'Ath Madrid': 'Atlético Madrid',
    'Atletico Madrid': 'Atlético Madrid',
    'Augsburg': 'FC Augsburg',
    'Barcelona': 'FC Barcelona',
    'Bari': 'Bari 1908',
    'Bayern Munich': 'FC Bayern München',
    'Besiktas': 'Beşiktaş',
    'Betis': 'Real Betis',
    'Birmingham': 'Birmingham City',
    'Blackburn': 'Blackburn Rovers',
    'Bochum': 'VfL Bochum 1848',
    'Bolton': 'Bolton Wanderers',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Bournemouth': 'AFC Bournemouth',
    'Bradford': 'Bradford City',
    'Brighton': 'Brighton & Hove Albion',
    'Bristol Rvs': 'Bristol Rovers',
    'Burton': 'Burton Albion',
    'Buyuksehyr': 'İstanbul Başakşehir',
    'Cadiz': 'Cádiz',
    'Cambridge': 'Cambridge United',
    'Cambuur': 'SC Cambuur',
    'Cardiff': 'Cardiff City',
    'Carlisle': 'Carlisle United',
    'Cartagena': 'FC Cartagena',
    'Celta': 'Celta de Vigo',
    'Celta Vigo': 'Celta de Vigo',
    'Charlton': 'Charlton Athletic',
    'Cheltenham': 'Cheltenham Town',
    'Colchester': 'Colchester United',
    'Coventry': 'Coventry City',
    'Crewe': 'Crewe Alexandra',
    'Darmstadt': 'Darmstadt 98',
    'Derby': 'Derby County',
    'Doncaster': 'Doncaster Rovers',
    'Dundee Utd': 'Dundee United',
    'Eibar': 'SD Eibar',
    'Espanol': 'Espanyol',
    'Eupen': 'AS Eupen',
    'Exeter': 'Exeter City',
    'FC Koln': 'FC Köln',
    'Famalicao': 'Famalicão',
    'Fenerbahce': 'Fenerbahçe',
    'Forest Green': 'Forest Green Rovers',
    'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
    'Freiburg': 'Freiburg II',
    'Gaziantep': 'Gaziantep F.K.',
    'Grenoble': 'Grenoble Foot 38',
    'Greuther Furth': 'SpVgg Greuther Fürth',
    'Groningen': 'FC Groningen',
    'Hamburg': 'Hamburger SV',
    'Hannover': 'Hannover 96',
    'Heerenveen': 'SC Heerenveen',
    'Hertha': 'Hertha BSC',
    'Hoffenheim': 'TSG Hoffenheim',
    'Huddersfield': 'Huddersfield Town',
    'Hull': 'Hull City',
    'Ibiza': 'UD Ibiza',
    'Ipswich': 'Ipswich Town',
    'Istanbulspor': 'İstanbulspor',
    'Karlsruhe': 'Karlsruher SC',
    'Kasimpasa': 'Kasımpaşa',
    'Korea Republic': 'Korea Republic',
    'Leeds': 'Leeds United',
    'Leganes': 'Leganés',
    'Leicester': 'Leicester City',
    'Lincoln': 'Lincoln City',
    'Luton': 'Luton Town',
    'Mainz': 'FSV Mainz 05',
    'Malaga': 'Málaga',
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Mansfield': 'Mansfield Town',
    'Maritimo': 'Marítimo',
    'Mirandes': 'Mirandés',
    'Newcastle': 'Newcastle United',
    'Nimes': 'Nîmes',
    'Northampton': 'Northampton Town',
    'Norwich': 'Norwich City',
    'Nottm Forest': 'Nottingham Forest',
    'Nurnberg': 'Nürnberg',
    'Oostende': 'KV Oostende',
    'Osnabruck': 'Osnabrück',
    'Oviedo': 'Real Oviedo',
    'Oxford': 'Oxford United',
    'Pacos Ferreira': 'Paços de Ferreira',
    'Paris FC': 'Paris',
    'Pau FC': 'Pau',
    'Peterboro': 'Peterborough United',
    'Plymouth': 'Plymouth Argyle',
    'Preston': 'Preston North End',
    'QPR': 'Queens Park Rangers',
    'Rotherham': 'Rotherham United',
    'Sheffield United': 'Sheffield United',
    'Sheffield Weds': 'Sheffield Wednesday',
    'Shrewsbury': 'Shrewsbury Town',
    'Sociedad': 'Real Sociedad',
    'Spal': 'SPAL',
    'Spurs': 'Tottenham Hotspur',
    'St Etienne': 'Saint-Étienne',
    'St Johnstone': 'St. Johnstone',
    'St Mirren': 'St. Mirren',
    'St Pauli': 'St. Pauli',
    'St Truiden': 'Sint-Truiden',
    'Standard': 'Standard Liège',
    'Stockport': 'Stockport County',
    'Stoke': 'Stoke City',
    'Stuttgart': 'VfB Stuttgart',
    'Sudtirol': 'Südtirol',
    'Swansea': 'Swansea City',
    'Swindon': 'Swindon Town',
    'Tottenham': 'Tottenham Hotspur',
    'Tranmere': 'Tranmere Rovers',
    'Twente': 'FC Twente',
    'USA': 'United States',
    'Umraniyespor': 'Ümraniyespor',
    'Union Berlin': 'FC Union Berlin',
    'Utrecht': 'FC Utrecht',
    'Valladolid': 'Real Valladolid',
    'Volendam': 'FC Volendam',
    'Wehen': 'Wehen Wiesbaden',
    'West Brom': 'West Bromwich Albion',
    'West Ham': 'West Ham United',
    'Wigan': 'Wigan Athletic',
    'Wolfsburg': 'VfL Wolfsburg',
    'Wolves': 'Wolverhampton Wanderers',
    'Wycombe': 'Wycombe Wanderers',
    'Zaragoza': 'Real Zaragoza',
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
