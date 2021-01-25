import pandas as pd
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

thisRound = 20
dataPath = "./prediction/Gameweeks/"
savePath = "./prediction/Gameweeks/"+str(thisRound)+"/prediction/LinearRegression/"
trainingData_df = pd.read_csv(dataPath + "trainingData.csv")
trainingData_year_df = pd.DataFrame()
for r in range(1,thisRound):
    df = pd.read_csv(dataPath + "2020 Training Data/round"+str(r)+"Training.csv")
    trainingData_year_df = pd.concat([trainingData_year_df,df], ignore_index=True)
predictData_df = pd.read_csv(dataPath + str(thisRound) + "/next_games/predictionData.csv")

save_df = pd.DataFrame()
save_df['player_name'] = predictData_df['player_name']
save_df['player_team'] = predictData_df['player_team']
save_df['element_type'] = predictData_df['element_type']
save_df['value'] = predictData_df['value']
save_df['element'] = predictData_df['element']
save_df['round'] = predictData_df['round']


trainingData_df = trainingData_df.drop(columns=['element','opponent_team','player_name','web_name','player_team',\
    'understat_id','h_team','a_team','round'])
trainingData_year_df = trainingData_year_df.drop(columns=['element','opponent_team','player_name','web_name','player_team',\
    'understat_id','h_team','a_team','round'])
predictData_df = predictData_df.drop(columns=['element','opponent_team','player_name','web_name','player_team',\
    'understat_id','h_team','a_team','round'])

trainingData_df = trainingData_df.apply(pd.to_numeric)
trainingData_year_df = trainingData_year_df.apply(pd.to_numeric)
predictData_df = predictData_df.apply(pd.to_numeric)


label_df = pd.concat([trainingData_df['label'],trainingData_year_df['label']])
    
#Remove features which have low correlation
#corr_df = pd.DataFrame(trainingData_df.corr()['label'])
#colNames = corr_df[corr_df['label'].between(-0.1,0.1)].index
#trainingData_df.drop(columns=colNames , inplace=True)
#predictData_df.drop(columns=colNames , inplace=True)

trainingData_df = trainingData_df.drop(columns=['label'])
trainingData_year_df = trainingData_year_df.drop(columns=['label'])
trainingData_df = pd.concat([trainingData_df,trainingData_year_df])

X = np.array(trainingData_df)
y = np.array(label_df)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=20)

clf = LinearRegression(n_jobs=-1).fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

X = np.array(predictData_df)
forecast = clf.predict(X)
save_df['points'] = forecast
save_df = pd.pivot_table(save_df, values=['points'], index=['player_name', 'player_team','element_type','element','round','value'], aggfunc=np.sum).reset_index()

df = save_df.sort_values(["round","points"], ascending=[True,False])
df.to_csv(savePath + "PredictLR.csv", encoding='utf-8', index = False)