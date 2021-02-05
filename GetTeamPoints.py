import requests
import pandas as pd
import numpy as np
import json

pd.options.mode.chained_assignment = None
url = "https://fantasy.premierleague.com/api/element-summary/{}/"

rd = 22
start = 22

teams = pd.read_csv("./current year/2020-21/teams.csv")
path = './prediction/Gameweeks/2020 Training Data/'
for i in range(start, rd + 1):
    fileName = 'round' + str(i)+ 'Training.csv'
    round = i
    training_df = pd.read_csv(path+fileName)
    training_df['opp_id'] = training_df.opponent_team.map(teams.set_index('name').id)
    training_df['label'] = np.nan
    for ind in training_df.index: 
        playerID = str(training_df['element'][ind])
        playerURL = url.format(playerID)
        r = requests.get(playerURL)
        json = r.json()
        gw_df = pd.DataFrame(json['history'])

        if not gw_df.empty:
            if gw_df['round'].isin([round]).any().any():
                training_df['label'][ind] = gw_df.loc[(gw_df['round']==round) & (gw_df['opponent_team']==training_df['opp_id'][ind]), 'total_points']

    training_df = training_df.drop(columns=['opp_id'])
    training_df.to_csv(path+fileName, encoding='utf-8', index = False)


predictionPath = './prediction/Gameweeks/'+str(rd)+'/prediction/'
lr = predictionPath+'LinearRegression/'
gbm = predictionPath+'GBM/'
rf = predictionPath+'RandomForest/'
pt = 'PredictedTeam.csv'
ptr = 'PredictedTeamOnlyThisRound.csv'

fileNames = [predictionPath+pt, lr+pt,lr+ptr,gbm+pt,gbm+ptr,rf+pt,rf+ptr]
training_df = pd.read_csv(path+'round' + str(rd)+ 'Training.csv')
training_df = pd.pivot_table(training_df, values=['label'], index=['element'], aggfunc=np.sum).reset_index()

for file in fileNames:
    df = pd.read_csv(file)
    df['label'] = np.nan
    df['label'] = df.element.map(training_df.set_index('element').label)
    df.to_csv(file, encoding='utf-8', index = False)



#Accuracy of predicted points
st = rd
savePath = './prediction/Gameweeks/Accuracy/'
df = pd.DataFrame()
for i in range(st, rd+1):
    predictDataLR = pd.read_csv('./prediction/Gameweeks/'+str(i)+'/prediction/LinearRegression/PredictLR.csv')
    predictDataRF = pd.read_csv('./prediction/Gameweeks/'+str(i)+'/prediction/RandomForest/PredictRF.csv')
    predictDataGBM = pd.read_csv('./prediction/Gameweeks/'+str(i)+'/prediction/GBM/PredictGBM.csv')
    predictDataCombine = pd.read_csv('./prediction/Gameweeks/'+str(i)+'/prediction/PredictCombine.csv')
    
    trainingData = pd.read_csv('./prediction/Gameweeks/2020 Training Data/round' + str(i)+ 'Training.csv')
    predictDataLR = predictDataLR[predictDataLR['round'] == i].reset_index(drop=True)
    predictDataRF = predictDataRF[predictDataRF['round'] == i].reset_index(drop=True)
    predictDataGBM = predictDataGBM[predictDataGBM['round'] == i].reset_index(drop=True)
    predictDataCombine = predictDataCombine[predictDataCombine['round'] == i].reset_index(drop=True)

    trainingData = pd.pivot_table(trainingData, values=['label'], index=['element','round'], aggfunc=np.sum).reset_index()

    predictDataLR['actual_points'] = predictDataLR.element.map(trainingData.set_index('element').label)
    predictDataRF['actual_points'] = predictDataRF.element.map(trainingData.set_index('element').label)
    predictDataGBM['actual_points'] = predictDataGBM.element.map(trainingData.set_index('element').label)
    predictDataCombine['actual_points'] = predictDataCombine.element.map(trainingData.set_index('element').label)

    predictDataLR['diff_points'] = abs(predictDataLR['actual_points'] - predictDataLR['points'])
    predictDataRF['diff_points'] = abs(predictDataRF['actual_points'] - predictDataRF['points'])
    predictDataGBM['diff_points'] = abs(predictDataGBM['actual_points'] - predictDataGBM['points'])
    predictDataCombine['diff_points'] = abs(predictDataCombine['actual_points'] - predictDataCombine['points'])

    predictDataLR = predictDataLR.sort_values(["points"], ascending=[False])
    predictDataRF = predictDataRF.sort_values(["points"], ascending=[False])
    predictDataGBM = predictDataGBM.sort_values(["points"], ascending=[False])
    predictDataCombine = predictDataCombine.sort_values(["points"], ascending=[False])

    predictDataLR = predictDataLR.head(250)
    predictDataRF = predictDataRF.head(250)
    predictDataGBM = predictDataGBM.head(250)
    predictDataCombine = predictDataCombine.head(250)

    predictDataLR25 = predictDataLR.head(25)
    predictDataRF25 = predictDataRF.head(25)
    predictDataGBM25 = predictDataGBM.head(25)
    predictDataCombine25 = predictDataCombine.head(25)

    predictDataLR25['p_rank'] = predictDataLR25['points'].rank(ascending=False)
    predictDataLR25['a_rank'] = predictDataLR25['actual_points'].rank(ascending=False)
    predictDataLR25['diff_rank'] = abs(predictDataLR25['a_rank'] - predictDataLR25['p_rank'])

    predictDataRF25['p_rank'] = predictDataRF25['points'].rank(ascending=False)
    predictDataRF25['a_rank'] = predictDataRF25['actual_points'].rank(ascending=False)
    predictDataRF25['diff_rank'] = abs(predictDataRF25['a_rank'] - predictDataRF25['p_rank'])

    predictDataGBM25['p_rank'] = predictDataGBM25['points'].rank(ascending=False)
    predictDataGBM25['a_rank'] = predictDataGBM25['actual_points'].rank(ascending=False)
    predictDataGBM25['diff_rank'] = abs(predictDataGBM25['a_rank'] - predictDataGBM25['p_rank'])    
    
    predictDataCombine25['p_rank'] = predictDataCombine25['points'].rank(ascending=False)
    predictDataCombine25['a_rank'] = predictDataCombine25['actual_points'].rank(ascending=False)
    predictDataCombine25['diff_rank'] = abs(predictDataCombine25['a_rank'] - predictDataCombine25['p_rank'])

    new_row = {'LRRank25':predictDataLR25['diff_rank'].mean(),'RFRank25':predictDataRF25['diff_rank'].mean(), 'GBMRank25':predictDataGBM25['diff_rank'].mean(),'CombineRank25':predictDataCombine25['diff_rank'].mean(),\
        'round': i, 'LRPoints250':predictDataLR['diff_points'].mean(), 'LRPoints25':predictDataLR25['diff_points'].mean(),\
       'RFPoints250':predictDataRF['diff_points'].mean(), 'RFPoints25':predictDataRF25['diff_points'].mean(),\
      'GBMPoints250':predictDataGBM['diff_points'].mean(), 'GBMPoints25':predictDataGBM25['diff_points'].mean(),\
      'CombinePoints250':predictDataCombine['diff_points'].mean(), 'CombinePoints25':predictDataCombine25['diff_points'].mean()}
    df = df.append(new_row, ignore_index=True)

df.to_csv(savePath + 'accuracy'+str(rd)+'.csv', encoding='utf-8', index = False)



