from pathlib import Path
import pandas as pd

TEAM_NAME_MAP = {
    'AZ Alkmaar': 'AZ Alkmaar',
    'Accrington': 'Accrington Stanley',
    'Alaves': 'Deportivo Alavés',
    'Albacete': 'Albacete Balompié',
    'Almeria': 'UD Almería',
    'Amiens': 'Amiens SC',
    'Angers': 'Angers SCO',
    'Annecy': 'FC Annecy',
    'Arouca': 'FC Arouca',
    'Ath Bilbao': 'Athletic Club',
    'Ath Madrid': 'Atlético Madrid',
    'Atletico Madrid': 'Atlético Madrid',
    'Augsburg': 'FC Augsburg',
    'Barcelona': 'FC Barcelona',
    'Bari': 'SSC Bari',
    'Bayern Munich': 'FC Bayern München',
    'Besiktas': 'Beşiktaş JK',
    'Betis': 'Real Betis Balompié',
    'Birmingham': 'Birmingham City',
    'Blackburn': 'Blackburn Rovers',
    'Blackpool': 'Blackpool FC',
    'Bochum': 'VfL Bochum 1848',
    'Bolton': 'Bolton Wanderers',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Boulogne': "US Boulogne Cote d'Opale",
    'Bournemouth': 'AFC Bournemouth',
    'Bradford': 'Bradford City',
    'Brighton': 'Brighton & Hove Albion',
    'Bristol Rvs': 'Bristol Rovers',
    'Burgos': 'Burgos CF',
    'Burton': 'Burton Albion',
    'Buyuksehyr': 'Medipol Başakşehir FK',
    'Cadiz': 'Cádiz CF',
    'Cambridge': 'Cambridge United',
    'Cardiff': 'Cardiff City',
    'Castellon': 'CD Castellón',
    'Celta': 'RC Celta',
    'Celta Vigo': 'RC Celta',
    'Cercle Brugge': 'Cercle Brugge KSV',
    'Cesena': 'Cesena FC',
    'Charlton': 'Charlton Athletic',
    'Cheltenham': 'Cheltenham Town',
    'Clermont': 'Clermont Foot 63',
    'Club Brugge': 'Club Brugge KV',
    'Colchester': 'Colchester United',
    'Cordoba': 'Córdoba CF',
    'Coventry': 'Coventry City',
    'Crewe': 'Crewe Alexandra',
    'Darmstadt': 'SV Darmstadt 98',
    'Derby': 'Derby County',
    'Doncaster': 'Doncaster Rovers',
    'Dundee': 'Dundee FC',
    'Dundee United': 'Dundee United FC',
    'Dundee Utd': 'Dundee United FC',
    'Eibar': 'SD Eibar',
    'Elche': 'Elche CF',
    'Elversberg': 'SV Elversberg',
    'Erzgebirge Aue': 'FC Erzgebirge Aue',
    'Espanol': 'RCD Espanyol',
    'Estrela': 'Estrela da Amadora',
    'Exeter': 'Exeter City',
    'Eyupspor': 'Eyüpspor',
    'FC Koln': '1. FC Köln',
    'Famalicao': 'Famalicão',
    'Fenerbahce': 'Fenerbahçe SK',
    'Fortuna Dusseldorf': 'Fortuna Düsseldorf',
    'Freiburg': 'SC Freiburg',
    'Fulham': 'Fulham FC',
    'Galatasaray': 'Galatasaray SK',
    'Gaziantep': 'Gaziantep FK',
    'Genclerbirligi': 'Gençlerbirliği SK',
    'Getafe': 'Getafe CF',
    'Gil Vicente': 'Gil Vicente FC',
    'Girona': 'Girona FC',
    'Goztep': 'Göztepe SK',
    'Granada': 'Granada CF',
    'Grenoble': 'Grenoble Foot 38',
    'Greuther Furth': 'SpVgg Greuther Fürth',
    'Groningen': 'FC Groningen',
    'Hamburg': 'Hamburger SV',
    'Hannover': 'Hannover 96',
    'Hansa Rostock': 'FC Hansa Rostock',
    'Heerenveen': 'SC Heerenveen',
    'Heracles': 'Heracles Almelo',
    'Hertha': 'Hertha BSC',
    'Hoffenheim': 'TSG 1899 Hoffenheim',
    'Huddersfield': 'Huddersfield Town',
    'Huesca': 'SD Huesca',
    'Hull': 'Hull City',
    'Ingolstadt': 'FC Ingolstadt 04',
    'Ipswich': 'Ipswich Town',
    'Juve Stabia': 'SS Juve Stabia',
    'Karlsruhe': 'Karlsruher SC',
    'Kasimpasa': 'Kasımpaşa SK',
    'Korea Republic': 'Korea Republic',
    'La Coruna': 'RC Deportivo de La Coruña',
    'Las Palmas': 'UD Las Palmas',
    'Le Havre': 'Le Havre AC',
    'Le Mans': 'Le Mans FC',
    'Leeds': 'Leeds United',
    'Leganes': 'CD Leganés',
    'Leicester': 'Leicester City',
    'Lens': 'RC Lens',
    'Levante': 'Levante UD',
    'Lille': 'Lille OSC',
    'Lincoln': 'Lincoln City',
    'Livingston': 'Livingston FC',
    'Lorient': 'FC Lorient',
    'Luton': 'Luton Town',
    'Mainz': '1. FSV Mainz 05',
    'Malaga': 'Málaga CF',
    'Mallorca': 'RCD Mallorca',
    'Man City': 'Manchester City',
    'Man United': 'Manchester United',
    'Mansfield': 'Mansfield Town',
    'Metz': 'FC Metz',
    'Milan': 'AC Milan',
    'Millwall': 'Millwall FC',
    'Mirandes': 'CD Mirandés',
    'Monaco': 'AS Monaco',
    'Montpellier': 'Montpellier HSC',
    'Moreirense': 'Moreirense FC',
    'Motherwell': 'Motherwell FC',
    'Nacional': 'CD Nacional',
    'Nancy': 'AS Nancy Lorraine',
    'Nantes': 'FC Nantes',
    'Newcastle': 'Newcastle United',
    'Northampton': 'Northampton Town',
    'Norwich': 'Norwich City',
    'Nottm Forest': 'Nottingham Forest',
    'Nurnberg': '1. FC Nürnberg',
    'Oldham': 'Oldham Athletic',
    'Osnabruck': 'VfL Osnabrück',
    'Oviedo': 'Real Oviedo',
    'Oxford': 'Oxford United',
    'Paderborn': 'SC Paderborn 07',
    'Palermo': 'Palermo FC',
    'Panathinaikos': 'Panathinaikos FC',
    'Paris FC': 'Paris FC',
    'Pau FC': 'Pau FC',
    'Peterboro': 'Peterborough United',
    'Plymouth': 'Plymouth Argyle',
    'Porto': 'FC Porto',
    'Preston': 'Preston North End',
    'QPR': 'Queens Park Rangers',
    'RAAL La Louviere': 'RAAL La Louvière',
    'Rangers': 'Rangers FC',
    'Reading': 'Reading FC',
    'Red Star': 'Red Star FC',
    'Rio Ave': 'Rio Ave FC',
    'Rodez': 'Rodez Aveyron Football',
    'Rotherham': 'Rotherham United',
    'Schalke 04': 'FC Schalke 04',
    'Sevilla': 'Sevilla FC',
    'Sheffield United': 'Sheffield United',
    'Sheffield Weds': 'Sheffield Wednesday',
    'Shrewsbury': 'Shrewsbury Town',
    'Sociedad': 'Real Sociedad',
    'Spurs': 'Tottenham Hotspur',
    'St Mirren': 'St. Mirren',
    'St Pauli': 'FC St. Pauli',
    'St Truiden': 'Sint-Truidense VV',
    'Standard': 'Standard de Liège',
    'Stockport': 'Stockport County',
    'Stoke': 'Stoke City',
    'Strasbourg': 'RC Strasbourg Alsace',
    'Stuttgart': 'VfB Stuttgart',
    'Sudtirol': 'Südtirol',
    'Swansea': 'Swansea City',
    'Swindon': 'Swindon Town',
    'Tondela': 'CD Tondela',
    'Tottenham': 'Tottenham Hotspur',
    'Toulouse': 'Toulouse FC',
    'Tranmere': 'Tranmere Rovers',
    'Twente': 'FC Twente',
    'USA': 'United States',
    'Union Berlin': '1. FC Union Berlin',
    'Utrecht': 'FC Utrecht',
    'Valencia': 'Valencia CF',
    'Valladolid': 'Real Valladolid CF',
    'Villarreal': 'Villarreal CF',
    'Volendam': 'FC Volendam',
    'Wehen': 'SV Wehen Wiesbaden',
    'Werder Bremen': 'SV Werder Bremen',
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
