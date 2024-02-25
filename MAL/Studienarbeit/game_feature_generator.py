import pandas as pd
from tqdm.auto import tqdm
from functools import reduce

class GameFeatureGenerator:
    """
    Wrapper Class for generation features from the game dataset. 
    """
    def __init__(self):
        return

    def __extract_last_games(self, data, n):
        """
        Extract the last n rows of data.
        Parameters:
            data: The dataframe the rows should be extracted from.
            n: How many rows should be extracted.
        Returns:
            Last n extracted rows of data.
        """
        return data.tail(n) if len(data) > n else data

    def calc_game_features(self, all_games):
        """
        Calculate game related features. 
        Paramters:
            all_games: Dataframe which contains all games these features should be calculated for and taken into account.
        Returns:
            all_games: Dataframe with additional game features.
        """
        pb = tqdm(total=all_games.shape[0], desc="Calculating game features: ", unit="match", position=0, leave=True)
        
        for index, row in all_games.iterrows():
            home_team_wr = 0
            away_team_wr = 0
            home_team_wr_home = 0
            away_team_wr_away = 0
            no_goals_conc_rate_home = 0
            no_goals_conc_rate_away = 0
            avg_goals_home = 0
            avg_goals_away = 0
            avg_goals_conc_home = 0
            avg_goals_conc_away = 0
            
            #Extract home/away team and their corresponding games / home/away games. For avg. features, only take the last 10 games of this history.
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            history = all_games.iloc[0:(index), :]
            home_team_games = self.__extract_last_games(history[(history['HomeTeam'] == home_team) | (history['AwayTeam'] == home_team)], 10)
            away_team_games = self.__extract_last_games(history[(history['HomeTeam'] == away_team) | (history['AwayTeam'] == away_team)], 10)
            home_team_games_home = self.__extract_last_games(history[(history['HomeTeam'] == home_team)], 5)
            away_team_games_away = self.__extract_last_games(history[(history['AwayTeam'] == away_team)], 5)
            home_team_games_away = self.__extract_last_games(history[(history['AwayTeam'] == home_team)], 5)
            away_team_games_home = self.__extract_last_games(history[(history['HomeTeam'] == away_team)], 5)
                
            #Winnrates overall and home/away
            home_team_wins = home_team_games[(home_team_games['HomeTeam'] == home_team) & (home_team_games['FTR'] == 'H') | 
                (home_team_games['AwayTeam'] == home_team) & (home_team_games['FTR'] == 'A')]
            away_team_wins = away_team_games[(away_team_games['HomeTeam'] == away_team) & (away_team_games['FTR'] == 'H') | 
                (away_team_games['AwayTeam'] == away_team) & (away_team_games['FTR'] == 'A')]
            home_team_home_wins = home_team_games_home[home_team_games_home['FTR'] == 'H']
            away_team_away_wins = away_team_games_away[away_team_games_away['FTR'] == 'A']
    
            #Goal features
            no_goals_conc_home_home = home_team_games_home[home_team_games_home['FTAG'] == 0]
            no_goals_conc_home_away = home_team_games_away[home_team_games_away['FTHG'] == 0]
            no_goals_conc_away_home = away_team_games_home[away_team_games_home['FTAG'] == 0]
            no_goals_conc_away_away = away_team_games_away[away_team_games_away['FTHG'] == 0]
            home_team_goals_home = home_team_games_home['FTHG'].sum()
            home_team_goals_away = home_team_games_away['FTAG'].sum()
            away_team_goals_home = away_team_games_home['FTHG'].sum()
            away_team_goals_away = away_team_games_away['FTAG'].sum()
            home_team_goals_conc_home = home_team_games_home['FTAG'].sum()
            home_team_goals_conc_away = home_team_games_away['FTHG'].sum()
            away_team_goals_conc_home = away_team_games_home['FTAG'].sum()
            away_team_goals_conc_away = away_team_games_away['FTHG'].sum()
    
            #At the start of the dataset, there are 0 games for each team. This is caught here. In those cases, the avg values will be NaN
            try:
                home_team_wr = round(home_team_wins.shape[0] / home_team_games.shape[0], 2)
                away_team_wr = round(away_team_wins.shape[0] / away_team_games.shape[0], 2)
            except:
                pass

            try:
                no_goals_conc_rate_home = round(((no_goals_conc_home_home.shape[0] + no_goals_conc_home_away.shape[0]) / (home_team_games_home.shape[0] + home_team_games_away.shape[0])), 2)
                no_goals_conc_rate_away = round(((no_goals_conc_away_home.shape[0] + no_goals_conc_away_away.shape[0]) / (away_team_games_home.shape[0] + away_team_games_away.shape[0])), 2)
                avg_goals_home = round((home_team_goals_home + home_team_goals_away) / (home_team_games_home.shape[0] + home_team_games_away.shape[0]), 2)
                avg_goals_away = round((away_team_goals_home + away_team_goals_away) / (away_team_games_home.shape[0] + away_team_games_away.shape[0]), 2)
                avg_goals_conc_home = round((home_team_goals_conc_home + home_team_goals_conc_away) / (home_team_games_home.shape[0] + home_team_games_away.shape[0]), 2)
                avg_goals_conc_away = round((away_team_goals_conc_home + away_team_goals_conc_away) / (away_team_games_home.shape[0] + away_team_games_away.shape[0]), 2)
            except:
                pass
    
            try:
                home_team_wr_home = round(home_team_home_wins.shape[0] / home_team_games_home.shape[0], 2)
                away_team_wr_away = round(away_team_away_wins.shape[0] / away_team_games_away.shape[0], 2)
            except:
                pass
                
            #Assign the features
            all_games.loc[index, 'WROH'] = home_team_wr
            all_games.loc[index, 'WROA'] = away_team_wr
            all_games.loc[index, 'WRHH'] = home_team_wr_home
            all_games.loc[index, 'WRAA'] = away_team_wr_away
            all_games.loc[index, 'NGCH'] = no_goals_conc_rate_home
            all_games.loc[index, 'NGCA'] = no_goals_conc_rate_away
            all_games.loc[index, 'AVGH'] = avg_goals_home
            all_games.loc[index, 'AVGA'] = avg_goals_away
            all_games.loc[index, 'AVGCH'] = avg_goals_conc_home
            all_games.loc[index, 'AVGCA'] = avg_goals_conc_away

            pb.update()
        return all_games

    def calc_match_spec_win_rates(self, all_games):
        """
        Calculate the winrates of each specific matchup for the last 4 matches the teams from that specific matchup have played. 
        Parameters:
            all_games: Dataframe which contains all games that should be taken into account.
        Returns:
            all_games: Dataframe with the additional winrates for each specific matchup.
        """
        home_win_rate = 0
        away_win_rate = 0
        draw_rate = 0
        pb = tqdm(total=all_games.shape[0], desc="Calculating match specific win rates: ", unit="match", position=0, leave=True)
        
        for index, row in all_games.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            history = all_games.iloc[0:(index), :]

            #Extract the last 4 matchups of the 2 teams and home wins, away wins and draws
            last_matchups = self.__extract_last_games(history[((history['HomeTeam'] == home_team) & (history['AwayTeam'] == away_team)) |
                ((history['HomeTeam'] == away_team) & (history['AwayTeam'] == home_team))
            ], 4)
            home_wins = last_matchups[((last_matchups['HomeTeam'] == home_team) & (last_matchups['FTR'] == 'H')) |
                ((last_matchups['AwayTeam'] == home_team) & (last_matchups['FTR'] == 'A'))
            ]
            away_wins = last_matchups[((last_matchups['HomeTeam'] == away_team) & (last_matchups['FTR'] == 'H')) |
                ((last_matchups['AwayTeam'] == away_team) & (last_matchups['FTR'] == 'A'))
            ]
            draws = last_matchups[last_matchups['FTR'] == 'D']
    
            #Until the first match was played, last_matchups will be empty. In that case, the rates will be NaN.
            try:
                home_win_rate = round(home_wins.shape[0] / last_matchups.shape[0], 2)
                away_win_rate = round(away_wins.shape[0] / last_matchups.shape[0], 2)
                draw_rate = round((1 - home_win_rate - away_win_rate), 2)
            except:
                pass

            #Assign the winrates
            all_games.loc[index, 'SMWRH'] = home_win_rate
            all_games.loc[index, 'SMWRA'] = away_win_rate
            all_games.loc[index, 'SMDR'] = draw_rate

            pb.update()
        return all_games

    def calc_features(self, all_games):
        """
        Pipes the all_games dataframe through the feature calculation methods.
        Parameters:
            all_games: All games that the features should be calculated for.
        Returns:
            all_games: Dataframe with all the additional game features.
        """
        all_games = self.calc_game_features(all_games)
        all_games = self.calc_match_spec_win_rates(all_games)
        return all_games
