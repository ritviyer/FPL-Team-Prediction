import pandas as pd
import numpy as np
import FetchFPLData as ffd
import ReadFPLData as rfd
import CalculatingFunctions as cf
import MapUnderstatToFPL as muf


pd.options.mode.chained_assignment = None

round = 4
year = 2020

path = './current year/2020-21/'
savePredectionDataTo = './prediction/Gameweeks/1/next_four/'

ffd.GetPlayerData(path)
#ffd.GetFixtures(path)

'''players_raw_df = pd.read_csv(path + 'players_raw.csv')
teams_df = pd.read_csv(path + "teams.csv" )
#ffd.GetPlayerHistoricData(path + 'players/', players_raw_df)

epl_keepCols = ['assists','bonus','bps','clean_sheets','creativity','goals_conceded','goals_scored',\
    'ict_index','influence','minutes','red_cards','saves','threat','yellow_cards','total_points']

playerHistory_df = rfd.ReadPlayerHistory(path + 'players/', year, epl_keepCols)
players_raw_df[epl_keepCols] = np.nan
for col in epl_keepCols:
    players_raw_df[col] = players_raw_df.code.map(playerHistory_df.set_index('element_code')[col])


players_raw_df['player_name'] = players_raw_df['first_name'] + " " + players_raw_df['second_name']
players_raw_df['player_team'] = players_raw_df['team']

players_raw_df = rfd.ReadFixtures(path, round, players_raw_df)

players_raw_df['player_team'] = players_raw_df.player_team.map(teams_df.set_index('id').name)
players_raw_df['opponent_team'] = players_raw_df.opponent_team.map(teams_df.set_index('id').name)

players_raw_df['element'] = players_raw_df['id']
players_raw_df['value'] = players_raw_df['now_cost']
players_raw_df['round'] = np.nan
players_raw_df['round'] = players_raw_df['round'].fillna(round)
players_raw_df['transfers_balance'] = players_raw_df['transfers_in'] - players_raw_df['transfers_out'] 


epl_keep_cols = ['element','round','assists','bonus','bps','clean_sheets','value','creativity',\
    'total_points','goals_conceded','goals_scored','ict_index','influence','minutes','opponent_team',\
    'red_cards','saves','selected_by_percent','threat','transfers_balance','was_home','yellow_cards',\
    'player_name','web_name','element_type','player_team']

players_raw_df = players_raw_df[epl_keep_cols]


indexNames = players_raw_df[ players_raw_df['opponent_team'].isna() ].index
players_raw_df.drop(indexNames , inplace=True)

players_raw_df = players_raw_df.groupby(['element_type']).apply(cf.GetNanValues)

understatIDLocation = './prediction/understatID.csv'
understatFile = pd.read_csv(understatIDLocation)
players_raw_df['understat_id'] = players_raw_df.element.map(understatFile.set_index('element').understat_id)

indexNames = players_raw_df[ players_raw_df['understat_id'].isna() ].index
players_raw_df.drop(indexNames , inplace=True)
players_raw_df['understat_id'] = players_raw_df.understat_id.astype(int)


#players_raw_df = players_raw_df.groupby(['player_team']).apply(muf.GetUnderstatID, yr = year)
#players_raw_df = players_raw_df.reset_index(drop=True)


players_raw_df['h_team'] = np.where(players_raw_df['was_home'] == 1, players_raw_df['player_team'], players_raw_df['opponent_team'])
players_raw_df['a_team'] = np.where(players_raw_df['was_home'] == 1, players_raw_df['opponent_team'], players_raw_df['player_team'])


un_cols = ['shots','xG','xA','key_passes','npg','npxG','xGChain','xGBuildup']
players_raw_df = players_raw_df.groupby(['element']).apply(muf.understatPlayerHistoricStats, yr=year, keepCols=un_cols)
players_raw_df = players_raw_df.reset_index(drop=True)

players_raw_df = players_raw_df.groupby(['player_team']).apply(muf.understatTeamHistoricStats, yr=year, group = 'player_team')
players_raw_df = players_raw_df.reset_index(drop=True)
players_raw_df = players_raw_df.groupby(['opponent_team']).apply(muf.understatTeamHistoricStats, yr=year, group = 'opponent_team')
players_raw_df = players_raw_df.reset_index(drop=True)
players_raw_df = players_raw_df.sort_values(["player_name", "round"])

players_raw_df = players_raw_df.groupby(['element_type']).apply(cf.GetNanValues)

players_raw_df['value'] = players_raw_df.value.astype(float)
players_raw_df['value'] = players_raw_df['value']/10

players_raw_df.to_csv(savePredectionDataTo + 'predictionData' + str(round) + '.csv', encoding='utf-8', index = False)'''


'''merged_df = pd.DataFrame()
for i in range(1,5):
    players_raw_df = pd.read_csv(savePredectionDataTo + 'predictionData' + str(i) + '.csv')
    merged_df = pd.concat([merged_df,players_raw_df], ignore_index=True)
merged_df.to_csv(savePredectionDataTo + 'predictionData.csv', encoding='utf-8', index = False)'''



'''avg90 = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored', 'ict_index',\
'influence', 'saves', 'threat', 'shots', 'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup']

avg = ['minutes', 'total_points', 'xG_player_team', 'xGA_player_team', 'npxG_player_team', 'npxGA_player_team', 'deep_player_team', 'deep_allowed_player_team',\
'scored_player_team', 'missed_player_team', 'npxGD_player_team', 'ppda_att_player_team', 'ppda_def_player_team', 'ppda_allowed_att_player_team',\
'ppda_allowed_def_player_team', 'xG_opponent_team', 'xGA_opponent_team', 'npxG_opponent_team', 'npxGA_opponent_team', 'deep_opponent_team',\
'deep_allowed_opponent_team', 'scored_opponent_team', 'missed_opponent_team', 'npxGD_opponent_team', 'ppda_att_opponent_team', 'ppda_def_opponent_team',\
'ppda_allowed_att_opponent_team', 'ppda_allowed_def_opponent_team']


players_raw_df = players_raw_df.groupby(['element']).apply(cf.RunningAverage90, prev = 3, params=avg90)
players_raw_df = players_raw_df.groupby(['element']).apply(cf.RunningAverage90, prev = 5, params=avg90)
players_raw_df = players_raw_df.groupby(['element']).apply(cf.RunningAverage90, prev = 38, params=avg90)

players_raw_df = players_raw_df.groupby(['element']).apply(cf.RunningAverage, prev = 3, params=avg)
players_raw_df = players_raw_df.groupby(['element']).apply(cf.RunningAverage, prev = 5, params=avg)
players_raw_df = players_raw_df.groupby(['element']).apply(cf.RunningAverage, prev = 38, params=avg)

players_raw_df.to_csv(savePredectionDataTo + 'predictionDataFinal.csv', encoding='utf-8', index = False)'''