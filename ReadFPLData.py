import pandas as pd
import numpy as np
import os


def ReadPlayerData(path, year, shift_cols, keep_cols):
    '''Browse all player folders in a given path and return gameweek data for each player in the season.

    Parameters:
        path (string) - path where the all player folders for the given season are located
        playerID (DataFrame) - maps Player names with FPL ID
        understatData (DataFrame) - understat data for all players in a given season

    Returns:
        playerSeasonData (DataFrame) - Contains data for each gameweek for all players in a given season'''

    players = os.listdir(path)
    playerSeasonData_df = pd.DataFrame()
    prevYear = str(year-1) + '/' + str(year)[2:]
    for player in players:
        if os.path.isfile(path + player + "/gw.csv"):
            playerPath = path + player + "/gw.csv" 
            df = pd.read_csv(playerPath)
            #Do not add players who haven't played at all the entire season (remove =0 for only training Data)
            if(df['minutes'].sum()>=0):
                df['label'] = df['total_points']
                df[shift_cols] = df[shift_cols].shift(1)
                if os.path.isfile(path + player + "/history.csv"):
                    historyPath = path + player + "/history.csv"
                    history_df = pd.read_csv(historyPath)
                    history_df = history_df.loc[history_df.season_name == prevYear]
                    history_df = history_df.reset_index()
                    history_df = history_df[shift_cols]
                    history_df = history_df.apply(pd.to_numeric)
                    if not history_df.empty:
                        matches = history_df.loc[0,'minutes']/90
                        df.loc[0, shift_cols] = history_df.loc[0,shift_cols]/matches
                playerSeasonData_df = pd.concat([playerSeasonData_df,df], ignore_index=True)
    playerSeasonData_df = playerSeasonData_df[keep_cols] 
    return playerSeasonData_df



def MapPlayerIDtoNameAndTeam(path, playerData_df):
    playerRaw_df = pd.read_csv(path + "players_raw.csv")
    playerRaw_df['name'] = playerRaw_df['first_name'] + " " + playerRaw_df['second_name']
    playerRaw_df['id'] = playerRaw_df.id.astype(int)

    playerData_df['player_name'] = playerData_df.element.map(playerRaw_df.set_index('id').name)
    playerData_df['web_name'] = playerData_df.element.map(playerRaw_df.set_index('id').web_name)
    playerData_df['element_type'] = playerData_df.element.map(playerRaw_df.set_index('id').element_type)

    #Map player to player team
    teams_df = pd.read_csv(path + "teams.csv" )
    playerData_df['player_team'] = playerData_df.element.map(playerRaw_df.set_index('id').team)
    playerData_df['player_team'] = playerData_df.player_team.map(teams_df.set_index('id').name)
    playerData_df['opponent_team'] = playerData_df.opponent_team.map(teams_df.set_index('id').name)

    return playerData_df

def ReadPlayerHistory(path, year, keepCols):
    players = os.listdir(path)
    playerHistory_df = pd.DataFrame()
    prevYear = str(year-1) + '/' + str(year)[2:]
    for player in players:
        if os.path.isfile(path + player + "/history.csv"):
            historyPath = path + player + "/history.csv"
            history_df = pd.read_csv(historyPath)
            history_df = history_df.loc[history_df.season_name == prevYear]
            if not history_df.empty:
                history_df = history_df.reset_index()
                elementCode = history_df.loc[0,'element_code']
                history_df = history_df[keepCols]
                history_df = history_df.apply(pd.to_numeric)
                matches = history_df.loc[0,'minutes']/90
                if matches == 0:
                    matches = 1
                history_df[keepCols] = history_df[keepCols]/matches
                history_df.loc[0,'element_code'] = elementCode
                playerHistory_df = pd.concat([playerHistory_df,history_df], ignore_index=True)
    return playerHistory_df


def ReadPlayerGameweekHistory(path, keepCols):
    players = os.listdir(path)
    playerHistory_df = pd.DataFrame()
    for player in players:
        if os.path.isfile(path + player + "/gw.csv"):
            historyPath = path + player + "/gw.csv"
            history_df = pd.read_csv(historyPath)
            history_df = history_df.tail(1)
            if not history_df.empty:
                history_df = history_df.reset_index()
                element = history_df.loc[0,'element']
                history_df = history_df[keepCols]
                history_df = history_df.apply(pd.to_numeric)
                history_df.loc[0,'element'] = element
                playerHistory_df = pd.concat([playerHistory_df,history_df], ignore_index=True)
    return playerHistory_df

def ReadFixtures(path, round, playerData_df):
    fixtures_df = pd.read_csv(path + "fixtures.csv")
    fixtures_df = fixtures_df.loc[fixtures_df.event == round]
    fixtures_df = fixtures_df.reset_index()

    playerData_df['h_team'] = np.nan
    playerData_df['a_team'] = np.nan
    playerData_df['opponent_team'] = np.nan
    playerData_df['was_home'] = np.nan
   
    teamsChecked = ()
    for ind in fixtures_df.index:
        home = fixtures_df['team_h'][ind]
        away = fixtures_df['team_a'][ind]

        if home in teamsChecked:
            dTeam = playerData_df['player_team'] == home
            dTeam_df = playerData_df[dTeam]
            dTeam_df.loc[dTeam_df.player_team == home, "opponent_team"] = away
            dTeam_df.loc[dTeam_df.player_team == home, "was_home"] = 1
            playerData_df = pd.concat([playerData_df,dTeam_df])
        else:
            playerData_df.loc[playerData_df.player_team == home, "opponent_team"] = away
            playerData_df.loc[playerData_df.player_team == home, "was_home"] = 1
            teamsChecked = teamsChecked + (home,)

        if away in teamsChecked:
            dTeam = playerData_df['player_team'] == away
            dTeam_df = playerData_df[dTeam]
            dTeam_df.loc[dTeam_df.player_team == away, "opponent_team"] = home
            dTeam_df.loc[dTeam_df.player_team == away, "was_home"] = 0
            playerData_df = pd.concat([playerData_df,dTeam_df])
        else:
            playerData_df.loc[playerData_df.player_team == away, "opponent_team"] = home
            playerData_df.loc[playerData_df.player_team == away, "was_home"] = 0
            teamsChecked = teamsChecked + (away,)

    return playerData_df