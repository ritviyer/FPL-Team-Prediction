import json
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import re
import MapUnderstatToFPL as muf

sns.set_style('whitegrid')
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 100)

#pdata = pd.read_csv("./prediction/Gameweeks/1/trainingData.csv")

#pdata = pdata.loc[pdata.element_type==4]
#df = pd.DataFrame(pdata.corr()['label'])
#df.to_csv("./prediction/Gameweeks/1/corr.csv", encoding='utf-8', index = True)
#print(pdata.pivot_table(index='element_type', values='label', aggfunc=np.mean))
#print(pdata.pivot_table(index='element_type', values='label', aggfunc=np.median))

#f = plt.figure(figsize=(16,9))
#ax1 = f.add_subplot(2,2,1)
#ax2 = f.add_subplot(2,2,2,sharex=ax1, sharey=ax1)
#ax3 = f.add_subplot(2,2,3,sharex=ax1, sharey=ax1)
#ax4 = f.add_subplot(2,2,4,sharex=ax1, sharey=ax1)
#ax1.set_title('FWD')
#sns.distplot(pdata[pdata.element_type==4].label, label='FWD',ax=ax1)
#ax1.axvline(np.mean(pdata[pdata.element_type==4].label),color='red', label='mean')
#ax2.set_title('MID')
#sns.distplot(pdata[pdata.element_type==3].label, label='MID',ax=ax2)
#ax2.axvline(np.mean(pdata[pdata.element_type==3].label),color='red', label='mean')
#ax3.set_title('DEF')
#sns.distplot(pdata[pdata.element_type==2].label, label='DEF',ax=ax3)
#ax3.axvline(np.mean(pdata[pdata.element_type==2].label),color='red', label='mean')
#ax4.set_title('GKP')
#sns.distplot(pdata[pdata.element_type==1].label, label='GKP',ax=ax4)
#ax4.axvline(np.mean(pdata[pdata.element_type==1].label),color='red', label='mean')
#plt.show()


def correlation_heatmap(train):
    correlations = train.corr()['label']
    print(correlations)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
    
#pdata = pdata.drop(columns=['element','opponent_team','player_name','web_name','player_team',\
#        'understat_id','h_team','a_team','round'])
#corr_df = pd.DataFrame(pdata.corr()['label'])
#colNames = corr_df[corr_df['label'].between(-0.05,0.05)].index
#pdata.drop(columns=colNames , inplace=True)
#correlation_heatmap(pdata)



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
round = 23
lr = pd.read_csv('./prediction/gameweeks/' + str(round) + '/prediction/LinearRegression/PredictLR.csv')
gbm = pd.read_csv('./prediction/gameweeks/' + str(round) + '/prediction/GBM/PredictGBM.csv')
rf = pd.read_csv('./prediction/gameweeks/' + str(round) + '/prediction/RandomForest/PredictRF.csv')

df = pd.concat([lr,gbm,rf])
df = pd.pivot_table(df, values=['value','points'], index=['player_name', 'player_team','element_type','element','round'], aggfunc=np.mean).reset_index()

df = df.sort_values(["round","points"], ascending=[True,False])
df.to_csv('./prediction/gameweeks/' + str(round) + '/prediction/PredictCombine.csv', encoding='utf-8', index = False)
