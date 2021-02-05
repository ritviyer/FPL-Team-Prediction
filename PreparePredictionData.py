import pandas as pd
import numpy as np
import FetchFPLData as ffd
import ReadFPLData as rfd
import CalculatingFunctions as cf
import MapUnderstatToFPL as muf
import PrepareTrainingData as ptd
import os


pd.options.mode.chained_assignment = None

thisRound = 23
year = 2020

path = './current year/2020-21/'
savePredectionDataTo = './prediction/Gameweeks/'+str(thisRound)+'/next_games/'
os.makedirs(savePredectionDataTo,exist_ok=True)

ffd.GetPlayerData(path)
ffd.GetFixtures(path)
#ffd.GetTeams(path)

players_raw_df = pd.read_csv(path + 'players_raw.csv')
teams_df = pd.read_csv(path + "teams.csv" )

ffd.GetPlayerGameweekData(path + 'players/', players_raw_df)
ffd.GetPlayerHistoricData(path + 'players/', players_raw_df)

for round in range(thisRound,thisRound+4):
    players_raw_df = pd.read_csv(path + 'players_raw.csv')
    teams_df = pd.read_csv(path + "teams.csv" )
    epl_keepCols = ['assists','bonus','bps','clean_sheets','creativity','goals_conceded','goals_scored',\
        'ict_index','influence','minutes','red_cards','saves','threat','yellow_cards','total_points']

    playerGWHistory_df = rfd.ReadPlayerGameweekHistory(path + 'players/', epl_keepCols)
    playerHistory_df = rfd.ReadPlayerHistory(path + 'players/', year, epl_keepCols)
    players_raw_df[epl_keepCols] = np.nan

    for col in epl_keepCols:
        players_raw_df[col] = players_raw_df.id.map(playerGWHistory_df.set_index('element')[col])

    for col in epl_keepCols:
        players_raw_df[col] = np.where(players_raw_df[col].isna(), players_raw_df.code.map(playerHistory_df.set_index('element_code')[col]), players_raw_df[col])


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


    ##players_raw_df = players_raw_df.groupby(['player_team']).apply(muf.GetUnderstatID, yr = year)
    ##players_raw_df = players_raw_df.reset_index(drop=True)


    players_raw_df['h_team'] = np.where(players_raw_df['was_home'] == 1, players_raw_df['player_team'], players_raw_df['opponent_team'])
    players_raw_df['a_team'] = np.where(players_raw_df['was_home'] == 1, players_raw_df['opponent_team'], players_raw_df['player_team'])

    name = 'predictionData'+ str(round-thisRound+1)
    players_raw_df.to_csv(savePredectionDataTo + name + '.csv', encoding='utf-8', index = False)

ptd.TrainLastWeekData(thisRound,year,path)
###############################################################################################################################################3

#Uncomment below section only for 1st round of each gameweek calculation
players_raw_df = pd.read_csv(savePredectionDataTo + 'predictionData1.csv')
playerData_df = pd.read_csv('./prediction/Gameweeks/2020 Training Data/round'+str(thisRound-1)+'Training.csv')
playerData_df = playerData_df.drop(columns=['label'])
r2 = playerData_df[playerData_df['round'] == (thisRound-1)].reset_index(drop=True)
cols = playerData_df.columns
prevRoundsData = pd.DataFrame()

for i in range(1,thisRound-1):
    playerData_df1 = pd.read_csv('./prediction/Gameweeks/2020 Training Data/round'+str(i)+'Training.csv')
    playerData_df1=playerData_df1[cols]
    r1 = playerData_df[playerData_df['round'] == i].reset_index(drop=True)

    indexNames = playerData_df1[ ~playerData_df1['element'].isin(r1['element'].tolist()) ].index
    playerData_df1.drop(indexNames , inplace=True)
    indexNames = r1[ r1['element'].isin(playerData_df1['element'].tolist()) ].index
    r1.drop(indexNames , inplace=True)
    prevRoundsData = pd.concat([prevRoundsData,r1,playerData_df1], ignore_index=True)

playerData_df = pd.concat([prevRoundsData,r2], ignore_index=True)
players_raw_df = pd.concat([players_raw_df, playerData_df], ignore_index=True)
players_raw_df = players_raw_df.groupby(['element']).apply(muf.understatPlayerStats, yr=str(year))
players_raw_df = players_raw_df.reset_index(drop=True)
players_raw_df = players_raw_df.fillna(0)
players_raw_df = players_raw_df.sort_values(["player_name", "round"])

un_cols = ['shots','xG','xA','key_passes','npg','npxG','xGChain','xGBuildup']
players_raw_df = players_raw_df.groupby(['element']).apply(cf.shiftRows, colNames = un_cols)
players_raw_df = players_raw_df.reset_index(drop=True)

players_raw_df = players_raw_df.groupby(['element']).apply(muf.understatPlayerHistoricStats, yr=year, keepCols=un_cols)
players_raw_df = players_raw_df.reset_index(drop=True)

players_raw_df = players_raw_df.groupby(['element']).apply(muf.understatMultipleFixturePlayerStats, cols=un_cols, round=thisRound)
players_raw_df = players_raw_df.reset_index(drop=True)

players_raw_df.to_csv(savePredectionDataTo + 'predictionData1.csv', encoding='utf-8', index = False)

#######################################################################################################################################################

#Uncomment below block for 2-4 for each gw
un_cols = ['shots','xG','xA','key_passes','npg','npxG','xGChain','xGBuildup']
playerData_df = pd.read_csv(savePredectionDataTo + 'predictionData1.csv')
playerData_df = playerData_df[playerData_df['round'] == thisRound].reset_index(drop=True)
playerData_df = pd.pivot_table(playerData_df, values=un_cols, index=['element'], aggfunc=np.mean).reset_index()
for i in range(2,5):
    players_raw_df = pd.read_csv(savePredectionDataTo + 'predictionData'+str(i)+'.csv')
    players_raw_df[un_cols] = np.nan
    for col in un_cols:
        players_raw_df[col] = players_raw_df.element.map(playerData_df.set_index('element')[col])

    players_raw_df.to_csv(savePredectionDataTo + 'predictionData'+str(i)+'.csv', encoding='utf-8', index = False)


########################################################################################################################################################

#round sent in understat function is just the current one
for i in range(1,5):
    players_raw_df = pd.read_csv(savePredectionDataTo + 'predictionData' + str(i) + '.csv')
    players_raw_df = players_raw_df.groupby(['player_team']).apply(muf.understatTeamStats, yr=year, group = 'player_team', round = i+thisRound-1)
    players_raw_df = players_raw_df.reset_index(drop=True)
    players_raw_df = players_raw_df.groupby(['opponent_team']).apply(muf.understatTeamStats, yr=year, group = 'opponent_team', round = i+thisRound-1)
    players_raw_df = players_raw_df.reset_index(drop=True)
    players_raw_df = players_raw_df.sort_values(["player_name", "round"])
    players_raw_df.to_csv(savePredectionDataTo + 'predictionData' + str(i) + '.csv', encoding='utf-8', index = False)

###########################################################################################################################################################

#round sent in understat function is just the current one
for i in range(1,5):
    players_raw_df = pd.read_csv(savePredectionDataTo + 'predictionData' + str(i) + '.csv')
    players_raw_df = players_raw_df.groupby(['player_team']).apply(muf.understatMultipleFixtureTeamStats, group = 'player_team', round = i+thisRound-1)
    players_raw_df = players_raw_df.reset_index(drop=True)
    players_raw_df = players_raw_df.groupby(['opponent_team']).apply(muf.understatMultipleFixtureTeamStats, group = 'opponent_team', round = i+thisRound-1)
    players_raw_df = players_raw_df.reset_index(drop=True)
    players_raw_df = players_raw_df.sort_values(["player_name", "round"])
    players_raw_df.to_csv(savePredectionDataTo + 'predictionData' + str(i) + '.csv', encoding='utf-8', index = False)

###########################################################################################################################################################

p1 = pd.read_csv(savePredectionDataTo + 'predictionData1.csv')
p2 = pd.read_csv(savePredectionDataTo + 'predictionData2.csv')
p3 = pd.read_csv(savePredectionDataTo + 'predictionData3.csv')
p4 = pd.read_csv(savePredectionDataTo + 'predictionData4.csv')
players_raw_df = pd.concat([p1,p2,p3,p4])


players_raw_df = players_raw_df.groupby(['player_team']).apply(muf.understatTeamHistoricStats, yr=year, group = 'player_team')
players_raw_df = players_raw_df.reset_index(drop=True)
players_raw_df = players_raw_df.groupby(['opponent_team']).apply(muf.understatTeamHistoricStats, yr=year, group = 'opponent_team')
players_raw_df = players_raw_df.reset_index(drop=True)
players_raw_df = players_raw_df.sort_values(["player_name", "round"])

players_raw_df = players_raw_df.groupby(['element_type']).apply(cf.GetNanValues)

players_raw_df['value'] = players_raw_df.value.astype(float)
players_raw_df['value'] = players_raw_df['value']/10

players_raw_df.to_csv(savePredectionDataTo + 'predictionData.csv', encoding='utf-8', index = False)

####################################################################################################################################################################

players_raw_df = pd.read_csv(savePredectionDataTo + 'predictionData.csv')

avg90 = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded', 'goals_scored', 'ict_index',\
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

for r in range(1,thisRound):
    df = players_raw_df[players_raw_df['round'] == r].reset_index(drop=True)
    df.to_csv(savePredectionDataTo + 'round'+str(r)+'Training.csv', encoding='utf-8', index = False)

players_raw_df = players_raw_df[players_raw_df['round'] >=thisRound].reset_index(drop=True)
players_raw_df.to_csv(savePredectionDataTo + 'predictionData.csv', encoding='utf-8', index = False)
