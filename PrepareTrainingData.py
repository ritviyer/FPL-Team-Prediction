import pandas as pd
import numpy as np
import os
import asyncio
pd.options.mode.chained_assignment = None
import ReadFPLData as rfd
import FetchUnderstatData as fud
import MapUnderstatToFPL as muf
import CalculatingFunctions as cf


#dataPath = "./current year/"
#trainingData_df = pd.DataFrame()
#round = 18

#seasons = os.listdir(dataPath)
#for season in seasons:
#    year = int(season[0:4])
#    seasonPath = dataPath + season + "/players/"
    
#    epl_shift_cols = ['assists','bonus','bps','clean_sheets','creativity','goals_conceded','goals_scored',\
#    'ict_index','influence','minutes','red_cards','saves','threat','yellow_cards','total_points']
    
#    epl_keep_cols = ['element','round','assists','bonus','bps','clean_sheets','value','creativity',\
#    'total_points','goals_conceded','goals_scored','ict_index','influence','minutes','opponent_team',\
#    'red_cards','saves','selected','threat','transfers_balance','was_home','yellow_cards','label']
    
#    playerData_df = rfd.ReadPlayerData(seasonPath,year,epl_shift_cols,epl_keep_cols)

#    #Map player ID to Player Name and Team
#    playerData_df = rfd.MapPlayerIDtoNameAndTeam(dataPath + season + "/", playerData_df)
#    playerData_df = playerData_df.groupby(['element_type']).apply(cf.GetNanValues)

    
#    #Map FPL and Understat data
#    #playerData_df = playerData_df.groupby(['player_team']).apply(muf.GetUnderstatID, yr = year)
#    #playerData_df = playerData_df.reset_index(drop=True)
#    understatIDLocation = './prediction/understatID.csv'
#    understatFile = pd.read_csv(understatIDLocation)
#    playerData_df['understat_id'] = playerData_df.element.map(understatFile.set_index('element').understat_id)
#    indexNames = playerData_df[ playerData_df['understat_id'].isna() ].index
#    playerData_df.drop(indexNames , inplace=True)
#    playerData_df['understat_id'] = playerData_df.understat_id.astype(int)


#    playerData_df['h_team'] = np.where(playerData_df['was_home'] == True, playerData_df['player_team'], playerData_df['opponent_team'])
#    playerData_df['a_team'] = np.where(playerData_df['was_home'] == True, playerData_df['opponent_team'], playerData_df['player_team'])

#    #playerData_df = playerData_df.groupby(['element']).apply(muf.understatPlayerStats, yr=str(year))
#    #playerData_df = playerData_df.reset_index(drop=True)
#    #playerData_df = playerData_df.fillna(0)

#    #playerData_df = playerData_df.groupby(['player_team']).apply(muf.understatTeamStats, yr=year, group = 'player_team')
#    #playerData_df = playerData_df.reset_index(drop=True)
#    #playerData_df = playerData_df.groupby(['opponent_team']).apply(muf.understatTeamStats, yr=year, group = 'opponent_team')
#    #playerData_df = playerData_df.reset_index(drop=True)

#    #playerData_df = playerData_df.sort_values(["player_name", "round"])

#    ##Remove wrongly mapped players
#    #removePlayerIndex = playerData_df[playerData_df['h_team'] == playerData_df['a_team']].index
#    #removePlayers = playerData_df.loc[removePlayerIndex,'player_name']
#    #removePlayers = removePlayers.tolist()
#    #playerData_df = playerData_df[~playerData_df.player_name.isin(removePlayers)]

#    ##Get Historic Understat Player Data
#    #un_cols = ['shots','xG','xA','key_passes','npg','npxG','xGChain','xGBuildup']
#    #playerData_df = playerData_df.groupby(['element']).apply(cf.shiftRows, colNames = un_cols)
#    #playerData_df = playerData_df.reset_index(drop=True)

#    #playerData_df = playerData_df.groupby(['element']).apply(muf.understatPlayerHistoricStats, yr=year, keepCols=un_cols)
#    #playerData_df = playerData_df.reset_index(drop=True)
#    #playerData_df = playerData_df.groupby(['player_team']).apply(muf.understatTeamHistoricStats, yr=year, group = 'player_team')
#    #playerData_df = playerData_df.reset_index(drop=True)
#    #playerData_df = playerData_df.groupby(['opponent_team']).apply(muf.understatTeamHistoricStats, yr=year, group = 'opponent_team')
#    #playerData_df = playerData_df.reset_index(drop=True)

#    #playerData_df = playerData_df.sort_values(["player_name", "round"])
    
#    #playerData_df = playerData_df.groupby(['element_type']).apply(cf.GetNanValues)
#    playerData_df['was_home'] = np.where(playerData_df['was_home'] == True, 1, 0)

#    playerData_df = cf.PercentSelected(playerData_df, year)
#    playerData_df['round'] = np.where(playerData_df['round'] > 38, playerData_df['round'] - 9, playerData_df['round'])

#    #playerData_df['value'] = playerData_df.value.astype(float)
#    #playerData_df['value'] = playerData_df['value']/10

#    #avg90 = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored', 'ict_index',\
#    #'influence', 'saves', 'threat', 'shots', 'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup']

#    #avg = ['minutes', 'total_points', 'xG_player_team', 'xGA_player_team', 'npxG_player_team', 'npxGA_player_team', 'deep_player_team', 'deep_allowed_player_team',\
#    #'scored_player_team', 'missed_player_team', 'npxGD_player_team', 'ppda_att_player_team', 'ppda_def_player_team', 'ppda_allowed_att_player_team',\
#    #'ppda_allowed_def_player_team', 'xG_opponent_team', 'xGA_opponent_team', 'npxG_opponent_team', 'npxGA_opponent_team', 'deep_opponent_team',\
#    #'deep_allowed_opponent_team', 'scored_opponent_team', 'missed_opponent_team', 'npxGD_opponent_team', 'ppda_att_opponent_team', 'ppda_def_opponent_team',\
#    #'ppda_allowed_att_opponent_team', 'ppda_allowed_def_opponent_team']


#    #playerData_df = playerData_df.groupby(['element']).apply(cf.RunningAverage90, prev = 3, params=avg90)
#    #playerData_df = playerData_df.groupby(['element']).apply(cf.RunningAverage90, prev = 5, params=avg90)
#    #playerData_df = playerData_df.groupby(['element']).apply(cf.RunningAverage90, prev = 38, params=avg90)

#    #playerData_df = playerData_df.groupby(['element']).apply(cf.RunningAverage, prev = 3, params=avg)
#    #playerData_df = playerData_df.groupby(['element']).apply(cf.RunningAverage, prev = 5, params=avg)
#    #playerData_df = playerData_df.groupby(['element']).apply(cf.RunningAverage, prev = 38, params=avg)

#    trainingData_df = pd.concat([trainingData_df,playerData_df], ignore_index=True)
#    #playerData_df.to_csv(dataPath + season + '/cleaned_players.csv', encoding='utf-8', index = False)


#trainingData_df.to_csv('./prediction/Gameweeks/2020 Training Data/round'+str(round-1)+'Training.csv', encoding='utf-8', index = False)


def TrainLastWeekData(round,year,dataPath):
    trainingData_df = pd.DataFrame()
    seasonPath = dataPath + "players/"
    
    epl_shift_cols = ['assists','bonus','bps','clean_sheets','creativity','goals_conceded','goals_scored',\
    'ict_index','influence','minutes','red_cards','saves','threat','yellow_cards','total_points']
    
    epl_keep_cols = ['element','round','assists','bonus','bps','clean_sheets','value','creativity',\
    'total_points','goals_conceded','goals_scored','ict_index','influence','minutes','opponent_team',\
    'red_cards','saves','selected','threat','transfers_balance','was_home','yellow_cards','label']
    
    playerData_df = rfd.ReadPlayerData(seasonPath,year,epl_shift_cols,epl_keep_cols)

    #Map player ID to Player Name and Team
    playerData_df = rfd.MapPlayerIDtoNameAndTeam(dataPath, playerData_df)
    playerData_df = playerData_df.groupby(['element_type']).apply(cf.GetNanValues)

    
    #Map FPL and Understat data
    #playerData_df = playerData_df.groupby(['player_team']).apply(muf.GetUnderstatID, yr = year)
    #playerData_df = playerData_df.reset_index(drop=True)
    understatIDLocation = './prediction/understatID.csv'
    understatFile = pd.read_csv(understatIDLocation)
    playerData_df['understat_id'] = playerData_df.element.map(understatFile.set_index('element').understat_id)
    indexNames = playerData_df[ playerData_df['understat_id'].isna() ].index
    playerData_df.drop(indexNames , inplace=True)
    playerData_df['understat_id'] = playerData_df.understat_id.astype(int)


    playerData_df['h_team'] = np.where(playerData_df['was_home'] == True, playerData_df['player_team'], playerData_df['opponent_team'])
    playerData_df['a_team'] = np.where(playerData_df['was_home'] == True, playerData_df['opponent_team'], playerData_df['player_team'])

    playerData_df['was_home'] = np.where(playerData_df['was_home'] == True, 1, 0)

    playerData_df = cf.PercentSelected(playerData_df, year)
    playerData_df['round'] = np.where(playerData_df['round'] > 38, playerData_df['round'] - 9, playerData_df['round'])

    trainingData_df = pd.concat([trainingData_df,playerData_df], ignore_index=True)
    trainingData_df.to_csv('./prediction/Gameweeks/2020 Training Data/round'+str(round-1)+'Training.csv', encoding='utf-8', index = False)