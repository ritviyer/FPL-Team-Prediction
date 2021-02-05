# xgboost for regression
from numpy import asarray
from numpy import mean
from numpy import std
from xgboost import XGBRegressor
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print("GBM")
thisRound = 23
dataPath = "./prediction/Gameweeks/"
savePath = "./prediction/Gameweeks/"+str(thisRound)+"/prediction/GBM/"
os.makedirs(savePath,exist_ok=True)

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

trainingData_df = trainingData_df.drop(columns=['label'])
trainingData_year_df = trainingData_year_df.drop(columns=['label'])
trainingData_df = pd.concat([trainingData_df,trainingData_year_df])

X = np.array(trainingData_df)
y = np.array(label_df)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=23)
clf = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 

clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)
accuracy = round(r2_score(y_test,y_test_pred)*100,2)
print(accuracy)
f = open(savePath+str(accuracy) + ".txt","w+")
f.write("RMSE : "+str(mean_squared_error(y_test, y_test_pred, squared=False))+"\n")
f.write("MAE : "+str(mean_absolute_error(y_test, y_test_pred)))
f.close()

X = np.array(predictData_df)
forecast = clf.predict(X)
save_df['points'] = forecast
save_df = pd.pivot_table(save_df, values=['points'], index=['player_name', 'player_team','element_type','element','round','value'], aggfunc=np.sum).reset_index()

df = save_df.sort_values(["round","points"], ascending=[True,False])
df.to_csv(savePath + "PredictGBM.csv", encoding='utf-8', index = False)