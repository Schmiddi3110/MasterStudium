import requests
import time
import random
import os
import pandas as pd
from datetime import datetime

api_endpoint = 'http://api.clubelo.com/'
clubs = ['Bayern', 'Dortmund', 'Leverkusen', 'Schalke', 'Gladbach', 'Mainz', 'Hannover', 'Freiburg', 'Stuttgart', 'Hamburg', 'Frankfurt', 'Wolfsburg', 'Nuernberg', 'Werder', 'Hertha', 'Duesseldorf', 'Augsburg', 'Hoffenheim', 'Fuerth', 'Braunschweig', 'Koeln', 'UnionBerlin', 'Paderborn', 'Ingolstadt','Bochum', 'Heidenheim', 'Darmstadt', 'RBLeipzig', 'Bielefeld' ]

clubelo_mapping={
    'Bayern': 'Bayern Munich',
    'Dortmund': 'Dortmund',
    'Leverkusen': 'Leverkusen',
    'Schalke': 'Schalke 04',
    'Gladbach': "M'gladbach",
    'Mainz': 'Mainz',
    'Hannover': 'Hannover',
    'Freiburg': 'Freiburg',
    'Stuttgart': 'Stuttgart',
    'Hamburg': 'Hamburg',
    'Frankfurt': 'Ein Frankfurt',
    'Wolfsburg': 'Wolfsburg',
    'Nuernberg': 'Nurnberg',
    'Werder': 'Werder Bremen',
    'Hertha': 'Hertha',
    'Duesseldorf': 'Fortuna Dusseldorf',
    'Augsburg': 'Augsburg',
    'Hoffenheim': 'Hoffenheim',
    'Fuerth': 'Greuther Furth',
    'Braunschweig': 'Braunschweig',
    'Koeln': 'FC Koln',
    'UnionBerlin': 'Union Berlin',
    'Paderborn': 'Paderborn',
    'Ingolstadt': 'Ingolstadt',
    'Bochum': 'Bochum',
    'Heidenheim': 'Heidenheim',
    'Darmstadt': 'Darmstadt',
    'RBLeipzig': 'RB Leipzig',
    'Bielefeld': 'Bielefeld'
}

def crawl_clubelo():
    """
    Crawl ClubElo ratings data for each club in the 'clubs' dictionary.

    This function sends requests to the ClubElo API for each team, retrieves the data, and saves it as a CSV file.
    """
    for team in clubs:
        # Construct the API endpoint URL for the team
        api_url = api_endpoint + team

        response = requests.get(api_url)

        # Check if the request was successful
        if response.status_code == 200:
            with open(f'data/clubelo/{team}.csv', 'wb') as data_file:
                data_file.write(response.content)

            time.sleep(random.randint(60, 80))
        else:
            print(f"Failed to retrieve data for {team}. Status code: {response.status_code}")
            time.sleep(random.randint(60, 80))


def add_elo_to_features(data_dir, all_features):
    """
    Add ClubElo ratings to a DataFrame.

    Parameters:
    data_dir (str): The directory where ClubElo ratings CSV files are stored.
    all_features (DataFrame): A DataFrame containing features, including HomeTeam, AwayTeam, Date, etc.

    Returns:
    None: Modifies the all_features DataFrame in place by adding new columns for Elo ratings for home team (EloH)
                and away team (EloA).
    """

    clubelo_dir = data_dir + '/clubelo'
    for filename in os.listdir(clubelo_dir):
        if os.path.isfile(os.path.join(clubelo_dir, filename)):
            file = os.path.join(clubelo_dir, filename)
            df = pd.read_csv(clubelo_dir + "/" + filename)
            df = df[(df['To'] >= '2013-07-01') & (df['To'] <= '2024-12-31')]

            for index, row in all_features[(all_features["HomeTeam"] == clubelo_mapping[str(filename[:-4])])].iterrows():               
                date = row['Date']
                elo = df[df['To'] == str(date)[:10]]['Elo']
                all_features.loc[((all_features["HomeTeam"] == clubelo_mapping[str(filename[:-4])]) & (all_features['Date'] == date)), 'EloH'] = elo[elo.index[0]]

            for index, row in all_features[(all_features["AwayTeam"] == clubelo_mapping[str(filename[:-4])])].iterrows():              
                date = row['Date']
                elo = df[df['To'] == str(date)[:10]]['Elo']
                all_features.loc[((all_features["AwayTeam"] == clubelo_mapping[str(filename[:-4])]) & (
                            all_features['Date'] == date)), 'EloA'] = elo[elo.index[0]]
