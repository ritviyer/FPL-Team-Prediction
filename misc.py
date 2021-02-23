import json
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 100)


##update prediction data with latest stats
#playerData_df = pd.read_csv('./prediction/Gameweeks/10/next_games/predictionData.csv')
#playerRaw_df = pd.read_csv('./current year/2020-21/players_raw.csv')
#playerData_df['value'] = playerData_df.element.map(playerRaw_df.set_index('id').now_cost)
#playerData_df['value'] = playerData_df.value.astype(float)
#playerData_df['value'] = playerData_df['value']/10
#playerData_df['selected_by_percent'] = playerData_df.element.map(playerRaw_df.set_index('id').selected_by_percent)
#playerRaw_df['transfers_balance'] = playerRaw_df['transfers_in'] - playerRaw_df['transfers_out'] 
#playerData_df['transfers_balance'] = playerData_df.element.map(playerRaw_df.set_index('id').transfers_balance)

#playerData_df.to_csv('./prediction/Gameweeks/10/next_games/predictionData1.csv', encoding='utf-8', index = False)



##fix training values
#path = './prediction/gameweeks/2020 training data/'
#rd = 10
#start = 1
#for i in range(start, rd + 1):
#    filename = 'round' + str(i)+ 'training.csv'
#    multi = rd - i
#    multi = pow(10,multi);
#    team_df = pd.read_csv(path+filename)
#    #team_df['value'] = team_df['value'] * multi
#    team_df['value'] = team_df['value'] * 10
#    team_df.to_csv(path+filename, encoding='utf-8', index = False)



#combine all models
round = 25
lr = pd.read_csv('./prediction/gameweeks/' + str(round) + '/prediction/LinearRegression/PredictLR.csv')
gbm = pd.read_csv('./prediction/gameweeks/' + str(round) + '/prediction/GBM/PredictGBM.csv')
rf = pd.read_csv('./prediction/gameweeks/' + str(round) + '/prediction/RandomForest/PredictRF.csv')

df = pd.concat([lr,gbm,rf])
df = pd.pivot_table(df, values=['value','points'], index=['player_name', 'player_team','element_type','element','round'], aggfunc=np.mean).reset_index()

df = df.sort_values(["round","points"], ascending=[True,False])
df.to_csv('./prediction/gameweeks/' + str(round) + '/prediction/PredictCombine.csv', encoding='utf-8', index = False)
