
import time
import random
from collections import deque

import pandas as pd
import numpy as np

seasons = ['2013-2014', '2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2020-2021',
           '2021-2022', '2022-2023', '2023-2024']

clubs = {
    'Bayern Munich': ['054efa67', 'Bayern-Munich'],
    'Augsburg': ['0cdc4311', 'Augsburg'],
    'Leverkusen': ['c7a9f859', 'Bayer-Leverkusen'],
    'Bochum': ['b42c6323', 'Bochum'],
    'Darmstadt 98': ['6a6967fc', 'Darmstadt-98'],
    'Dortmund': ['add600ae', 'Dortmund'],
    'Eint Frankfurt': ['f0ac8ee6', 'Eintracht-Frankfurt'],
    'Freiburg': ['a486e511', 'Freiburg'],
    'Heidenheim': ['18d9d2a7', 'Heidenheim'],
    'Hoffenheim': ['033ea6b8', 'Hoffenheim'],
    'Köln': ['bc357bf7', 'Koln'],
    'Mainz 05': ['a224b06a', 'Mainz-05'],
    "M'Gladbach": ['32f3ee20', 'Monchengladbach'],
    'RB Leipzig': ['acbb6a5b', 'RB-Leipzig'],
    'Union Berlin': ['7a41008f', 'Union-Berlin'],
    'Werder Bremen': ['62add3bf', 'Werder-Bremen'],
    'Wolfsburg': ['4eaa11d7', 'Wolfsburg'],
    'Ingolstadt 04': ['12eb2039', 'Ingolstadt-04'],
    'Hamburger SV': ['26790c6a', 'Hamburger-SV'],
    'Stuttgart': ['598bc722', 'Stuttgart'],
    'Schalke 04': ['c539e393', 'Schalke-04'],
    'Hertha BSC': ['2818f8bc', 'Hertha-BSC'],
    'Arminia': ['247c4b67', 'Arminia'],
    'Greuther Fürth': ['12192a4c', 'Greuther-Furth'],
    'Düsseldorf': ['b1278397', 'Dusseldorf'],
    'Paderborn 07': ['d9f93f02', 'Paderborn-07'],
    'Hannover 96': ['60b5e41f', 'Hannover-96'],
    'Nürnberg': ['6f2c108c', 'Nurnberg'],
    'Braunschweig': ['8107958d', 'Eintracht-Braunschweig']
}

# Problem: Datensätze benennen Vereine teils unterschiedlich z.B Eint Frankfurt vs Ein Frankfurt. KEY: Vereinsname in fbref: Value Vereinsname Datensatz
fbref_mapping = {
    'Bayern Munich': 'Bayern Munich',
    'Augsburg': 'Augsburg',
    'Braunschweig': 'Braunschweig',
    'Hannover 96': 'Hannover',
    'Hertha BSC': 'Hertha',
    'Hoffenheim': 'Hoffenheim',
    'Leverkusen': 'Leverkusen',
    'Mainz 05': 'Mainz',
    'Schalke 04': 'Schalke 04',
    'Eint Frankfurt': 'Ein Frankfurt',
    'Freiburg': 'Freiburg',
    'Hamburger SV': 'Hamburg',
    "M'Gladbach": "M'gladbach",
    'Stuttgart': 'Stuttgart',
    'Werder Bremen': 'Werder Bremen',
    'Wolfsburg': 'Wolfsburg',
    'Dortmund': 'Dortmund',
    'Nürnberg': 'Nurnberg',
    'Köln': 'FC Koln',
    'Paderborn 07': 'Paderborn',
    'Darmstadt 98': 'Darmstadt',
    'Ingolstadt 04': 'Ingolstadt',
    'RB Leipzig': 'RB Leipzig',
    'Düsseldorf': 'Fortuna Dusseldorf',
    'Union Berlin': 'Union Berlin',
    'Arminia': 'Bielefeld',
    'Bochum': 'Bochum',
    'Greuther Fürth': 'Greuther Furth',
    'Heidenheim': 'Heidenheim'
}


def crawl_fbref_stats():
    """
    Crawl football statistics from FBref for multiple seasons and clubs.    

    Returns:
    dict: A dictionary containing crawled statistics for each season and club combination.        
    """
    fbref_data = {}
    for season in seasons:
        clubs_url = 'https://fbref.com/de/wettbewerbe/20/' + season + '/Statistiken-Bundesliga-' + season
        df = pd.read_html(clubs_url)[0]
        clubs_list = df['Verein'].to_list()

        time.sleep(1)
        for club in clubs_list:
            stats_url = "https://www.fbref.com/de/mannschaften/" + clubs[club][
                0] + "/" + season + "/spielprotokoll/c20/schedule/" + clubs[club][1] + "-Punkte-und-Eckdata-Bundesliga"

            df_stats = pd.read_html(stats_url)[1]
            filtered_df = df_stats[df_stats['Wett'] == 'Bundesliga'].dropna(subset=['Ergebnis'])

            # old seasons dont include xG/xGA scores
            if 'xG' not in filtered_df.columns:
                filtered_df['xG'] = np.nan
                filtered_df['xGA'] = np.nan

            fbref_data[season + club] = filtered_df
            time.sleep(random.randint(2, 10))

    return fbref_data


def crawl_fbref_wages():
    """
    Crawl football club wages statistics from FBref for multiple seasons and clubs.

    Returns:
    dict: A dictionary containing total club wages for each season and club        
    """
    fbref_wages = {}
    for season in seasons:
        clubs_url = 'https://fbref.com/de/wettbewerbe/20/' + season + '/Statistiken-Bundesliga-' + season
        df = pd.read_html(clubs_url)[0]
        clubs_list = df['Verein'].to_list()

        time.sleep(1)
        for club in clubs_list:
            wages_url = "https://fbref.com/de/mannschaften/" + clubs[club][0] + "/" + season + "/wages/" + clubs[club][
                1] + "-Angaben-zu-Gehaltern"
            df_stats = pd.read_html(wages_url)[0]
            df_stats = df_stats.dropna(subset=['Jahresgehälter'])
            try:
                df_stats['Jahresgehälter'] = df_stats['Jahresgehälter'].str.extract(r'€ (\d+(?:\.\d+)+)')
                df_stats['Jahresgehälter'] = df_stats['Jahresgehälter'].str.replace(".", "")
                df_stats['Jahresgehälter'] = pd.to_numeric(df_stats['Jahresgehälter'])
            except:
                print(season + club)
                continue
            fbref_wages[season + club] = df_stats['Jahresgehälter'].sum()
            time.sleep(random.randint(2, 10))

    return fbref_wages


def add_crawl_to_df(fbref_data, all_features):
    """
    Add crawled football statistics from FBref to a DataFrame.

    Parameters:
    fbref_data (dict): A dictionary containing crawled statistics for each season and club combination.                            
    all_features (DataFrame): A DataFrame containing features, including HomeTeam, AwayTeam, Date, etc.

    Returns:
    None: Modifies the all_features DataFrame in place by adding new columns for expected goals (xGH, xGA)
            and ball possession (BBH, BBA) for both home and away teams.
    """

    # add Columns to all_features
    all_features['xGH'] = np.nan  # expected Goals HomeTeam
    all_features['xGA'] = np.nan  # expected Goals AwayTeam
    all_features['BBH'] = np.nan  # Ballpossesion HomeTeam
    all_features['BBA'] = np.nan  # Ballpossession AwayTeam

    # add features
    xG_history_dict = {fbref_mapping[str(key)[9:]]: deque(maxlen=10) for key in fbref_data}
    bb_history_dict = {fbref_mapping[str(key)[9:]]: deque(maxlen=10) for key in fbref_data}

    for key in fbref_data.keys():
        fbref_data[key]['Datum'] = pd.to_datetime(fbref_data[key]['Datum'], format='mixed', dayfirst=True)
        for index, row in fbref_data[key].iterrows():
            try:
                # is the team playing home or away?
                selected_row = all_features[((all_features["HomeTeam"] == fbref_mapping[str(key)[9:]]) | (
                (all_features["AwayTeam"] == fbref_mapping[str(key)[9:]]))) & (
                                                        all_features['Date'] == str(row['Datum'])[0:10])].index[0]
                mask = all_features.loc[selected_row, ['HomeTeam', 'AwayTeam']].isin([fbref_mapping[str(key)[9:]]])
                column_name = mask.index[mask].tolist()
                xG_history_dict[fbref_mapping[str(key)[9:]]].append(row['xG'])
                bb_history_dict[fbref_mapping[str(key)[9:]]].append(row['Besitz'])

                if column_name[0] == "HomeTeam":
                    all_features.at[selected_row, 'xGH'] = round(
                        sum(xG_history_dict[fbref_mapping[str(key)[9:]]]) / len(
                            xG_history_dict[fbref_mapping[str(key)[9:]]]), ndigits=2)
                    all_features.at[selected_row, 'BBH'] = round(
                        sum(bb_history_dict[fbref_mapping[str(key)[9:]]]) / len(
                            bb_history_dict[fbref_mapping[str(key)[9:]]]), ndigits=2)

                else:
                    all_features.at[selected_row, 'xGA'] = round(
                        sum(xG_history_dict[fbref_mapping[str(key)[9:]]]) / len(
                            xG_history_dict[fbref_mapping[str(key)[9:]]]), ndigits=2)
                    all_features.at[selected_row, 'BBA'] = round(
                        sum(bb_history_dict[fbref_mapping[str(key)[9:]]]) / len(
                            bb_history_dict[fbref_mapping[str(key)[9:]]]), ndigits=2)

            except:
                print(str(row['Datum'])[0:10], key)
                continue


def add_wages_to_df(wages, all_features):
    """
    Add club wages data to a DataFrame

    Parameters:
    wages (dict): A dictionary containing total club wages for each season and club combination.        
    all_features (DataFrame): A DataFrame containing features, including HomeTeam, AwayTeam, Date, etc.

    Returns:
    None: Modifies the all_features DataFrame in place by adding new columns for total wages for home team (TWH)
            and total wages for away team (TWA).
    """
    all_features["TWH"] = np.nan
    all_features["TWA"] = np.nan
    for key, value in wages.items():
        years = str(key)[:9].split('-')
        club = fbref_mapping[str(key)[9:]]
        all_features.loc[
            ((all_features["Date"] >= years[0] + '-07-01') & (all_features['Date'] <= years[1] + '-07-01')) & (
                    all_features['HomeTeam'] == club), 'TWH'] = wages[key]
        all_features.loc[
            ((all_features["Date"] >= years[0] + '-07-01') & (all_features['Date'] <= years[1] + '-07-01')) & (
                    all_features['AwayTeam'] == club), 'TWA'] = wages[key]
