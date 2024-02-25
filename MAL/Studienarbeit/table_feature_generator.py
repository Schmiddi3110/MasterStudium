import pandas as pd
from tqdm.auto import tqdm
from functools import reduce

english_name_mappings_rev = {
    "Hamburg": "Hamburger SV",
    "Augsburg": "FC Augsburg",
    "Leverkusen": "Bayer Leverkusen",
    "Hannover": "Hannover 96",
    "Nurnberg": "1. FC Nürnberg",
    "Ingolstadt": "FC Ingolstadt 04",
    "RB Leipzig": "RB Leipzig",
    "FC Koln": "1. FC Köln",
    "Paderborn": "SC Paderborn 07",
    "Bayern Munich": "Bayern München",
    "Bielefeld": "Arminia Bielefeld",
    "Bochum": "VfL Bochum",
    "Werder Bremen": "Werder Bremen",
    "Braunschweig": "Eintracht Braunschweig",
    "Stuttgart": "VfB Stuttgart",
    "Dortmund": "Borussia Dortmund",
    "Mainz": "1. FSV Mainz 05",
    "Freiburg": "SC Freiburg",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Darmstadt": "SV Darmstadt 98",
    "Greuther Furth": "SpVgg Greuther Fürth",
    "Hoffenheim": "1899 Hoffenheim",
    "Wolfsburg": "VfL Wolfsburg",
    "Union Berlin": "1. FC Union Berlin",
    "Hertha": "Hertha BSC",
    "Schalke 04": "FC Schalke 04",
    "Fortuna Dusseldorf": "Fortuna Düsseldorf",
    "M'gladbach": "Bor. Mönchengladbach",
    "Heidenheim": "1. FC Heidenheim 1846"
}

class TableFeatureGenerator:
    """
    Wrapper Class for generation of table features.
    """
    def __init__(self):
        return

    def __extract_last_seasons(self, season_lst, n):
        """
        Extracts the last n seasons from season_lst
        Parameters:
            season_lst: All seasons that n should be extracted
            n: Amount of last seasons that should be extracted
        Returns:
            The last n elements of season_lst
        """
        return season_lst[:2] if len(season_lst) > n else season_lst

    def get_curr_features(self, all_games):
        """
        Get current placing and points for every game. Placing and points are extracted from the previous matchday, 
        as the corresponding tables always contain tables and points after the matchday.
        Parameters:
            all_games: Dataframe with all games that the current placings and points should be extracted for
        Returns:
            all_games: Dataframe with all games and the additional placing/points.
        """
        pb = tqdm(total=all_games.shape[0], desc="Adding current table features: ", unit="match", position=0, leave=True)
        
        tables_path = 'data/Spieltagtabellen/gesamt/'
        for index, row in all_games.iterrows():
            #Get current match_day and season
            match_day_str = row['MatchDay']
            season = match_day_str.split('_')[0]
            match_day = match_day_str.split('_')[1]
            prev_match_day = str(int(match_day) - 1)
            #If it is the first Matchday, take the results from the season before
            if prev_match_day == '0':
                season_lst = season.split('-')
                prev_year = str(int(season_lst[0]) - 1)
                post_year = str(int(season_lst[1]) - 1)
                prev_season = prev_year + '-' + post_year
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']
                table_path = tables_path + prev_season + '/' + prev_season + '_' + '34' + '.csv'
                table = pd.read_csv(table_path)
    
                #Only take stats from the included games, tables before that shall not be included. This is only the case for the first 9 lines 
                if prev_year == '2012':
                    #Maybe use random numbers here
                    all_games.loc[index, 'CPLH'] = 0
                    all_games.loc[index, 'CPH'] = 0
                    all_games.loc[index, 'CPLA'] = 0
                    all_games.loc[index, 'CPA'] = 0
                else:
                    #In case the Team did not play in the first Bundesliga the season before, assign default values
                    if english_name_mappings_rev[home_team] not in table['Mannschaft'].values:
                        all_games.loc[index, 'CPLH'] = 18
                        all_games.loc[index, 'CPH'] = 0
                    else:
                        all_games.loc[index, 'CPLH'] = table.loc[table['Mannschaft'] == english_name_mappings_rev[home_team], '#'].values[0]
                        all_games.loc[index, 'CPH'] = table.loc[table['Mannschaft'] == english_name_mappings_rev[home_team], 'Pkt.'].values[0]
                        
                    if english_name_mappings_rev[away_team] not in table['Mannschaft'].values:
                        all_games.loc[index, 'CPLA'] = 18
                        all_games.loc[index, 'CPA'] = 0
                    else:
                        all_games.loc[index, 'CPLA'] = table.loc[table['Mannschaft'] == english_name_mappings_rev[away_team], '#'].values[0] 
                        all_games.loc[index, 'CPA'] = table.loc[table['Mannschaft'] == english_name_mappings_rev[away_team], 'Pkt.'].values[0]
            else:
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']
                table_path = tables_path + season + '/' + season + '_' + prev_match_day + '.csv'
                table = pd.read_csv(table_path)
                all_games.loc[index, 'CPLH'] = table.loc[table['Mannschaft'] == english_name_mappings_rev[home_team], '#'].values[0]
                all_games.loc[index, 'CPLA'] = table.loc[table['Mannschaft'] == english_name_mappings_rev[away_team], '#'].values[0]
                all_games.loc[index, 'CPH'] = table.loc[table['Mannschaft'] == english_name_mappings_rev[home_team], 'Pkt.'].values[0]
                all_games.loc[index, 'CPA'] = table.loc[table['Mannschaft'] == english_name_mappings_rev[away_team], 'Pkt.'].values[0]
            pb.update()
        return all_games
    
    def calc_avg_table_features(self, all_games):
        """
        Calculate average placing and points over the last 2 seasons (plus the current one). 
        If the team has only played in the season the current game takes place in, the average equals the current placing/points 
        as its a seasonal average.
        Parameters:
            all_games: Dataframe with all games the average seasonal placing and points should be calculated for
        Returns:
            all_games: Dataframe with all games and the additional placing/point features
        """
        pb = tqdm(total=all_games.shape[0], desc="Calculating average table features: ", unit="match", position=0, leave=True)
        tables_path = 'data/Spieltagtabellen/gesamt/'
        for index, row in all_games.iterrows():
            season_count_home = 1
            season_count_away = 1
            season_points_home = row['CPH']
            season_points_away = row['CPA']
            season_placing_home = row['CPLH']
            season_placing_away = row['CPLA']
    
            match_day_str = row['MatchDay']
            season = match_day_str.split('_')[0]
            prev_year = int(season.split('-')[0]) 
            post_year = int(season.split('-')[1])
    
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
    
            #Add history points starting from season 2014/2015, as 2013/2014 is the "youngest" season in the dataset
            if prev_year > 2013:
                #Retrieve all previous season until the first one in the dataset
                years_list = list(range((prev_year - 1), 2012, -1))
                #Get the last 2 seasons the corresponding teams have played in 
                last_seasons_home = []
                last_seasons_away = []
                #Extract all seasons that the home team and away team participated in
                for year in years_list:
                    prev_year = year
                    post_year = year + 1
                    table_path = tables_path + str(prev_year) + '-' + str(post_year) + '/' + str(prev_year) + '-' + str(post_year) + '_' + '34' + '.csv'
                    table = pd.read_csv(table_path)
                    if english_name_mappings_rev[home_team] in table['Mannschaft'].values:
                        last_seasons_home.append(prev_year)
                    if english_name_mappings_rev[away_team] in table['Mannschaft'].values:
                        last_seasons_away.append(prev_year)

                #Get the last 2 seasons the teams have played in 
                last_2_seasons_home = self.__extract_last_seasons(last_seasons_home, 2)
                last_2_seasons_away = self.__extract_last_seasons(last_seasons_away, 2)

                #Iterate over the last 2 seasons and get the values. If no seasons were played in the past, this will be skipped
                for year in last_2_seasons_home:
                    prev_year = year
                    post_year = year + 1
                    table_path = tables_path + str(prev_year) + '-' + str(post_year) + '/' + str(prev_year) + '-' + str(post_year) + '_' + '34' + '.csv'
                    table = pd.read_csv(table_path)

                    season_count_home += 1
                    home_points = table.loc[table['Mannschaft'] == english_name_mappings_rev[home_team], 'Pkt.'].values[0]
                    home_placing = table.loc[table['Mannschaft'] == english_name_mappings_rev[home_team], '#'].values[0]
                    season_points_home += home_points
                    season_placing_home += home_placing

                #Same as above, just for the away team seasons
                for year in last_2_seasons_away:
                    prev_year = year
                    post_year = year + 1
                    table_path = tables_path + str(prev_year) + '-' + str(post_year) + '/' + str(prev_year) + '-' + str(post_year) + '_' + '34' + '.csv'
                    table = pd.read_csv(table_path)

                    season_count_away += 1
                    away_points = table.loc[table['Mannschaft'] == english_name_mappings_rev[away_team], 'Pkt.'].values[0]
                    away_placing = table.loc[table['Mannschaft'] == english_name_mappings_rev[away_team], '#'].values[0]
                    season_points_away += away_points
                    season_placing_away += away_placing
    
            #Calculate averages
            all_games.loc[index, 'AVPSH'] = round((season_points_home / season_count_home), 2)
            all_games.loc[index, 'AVPSA'] = round((season_points_away / season_count_away), 2)
            all_games.loc[index, 'AVPLSH'] = round((season_placing_home / season_count_home), 2)
            all_games.loc[index, 'AVPLSA'] = round((season_placing_away / season_count_away), 2)
    
            pb.update()
        return all_games

    def calc_features(self, all_games):
        """
        Pipes all games through table feature calculation.
        Parameters:
            all_games: Dataframe with all games that the table features should be calculated for.
        Returns:
            all_games: Dataframe with all games and the additional table features.
        """
        all_games = self.get_curr_features(all_games)
        all_games = self.calc_avg_table_features(all_games)
        return all_games
    