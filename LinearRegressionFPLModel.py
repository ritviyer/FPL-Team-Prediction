import pandas as pd
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

style.use('ggplot')

thisRound = 29
dataPath = "./prediction/Gameweeks/"
savePath = "./prediction/Gameweeks/"+str(thisRound)+"/prediction/LinearRegression/"
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
    
#Remove features which have low correlation
corr_df = pd.DataFrame(trainingData_df.corr()['label'])
colNames = corr_df[corr_df['label'].between(-0.0025,0.0025)].index
trainingData_df.drop(columns=colNames , inplace=True)
trainingData_year_df.drop(columns=colNames , inplace=True)
predictData_df.drop(columns=colNames , inplace=True)

trainingData_df = trainingData_df.drop(columns=['label'])
trainingData_year_df = trainingData_year_df.drop(columns=['label'])
trainingData_df = pd.concat([trainingData_df,trainingData_year_df])

X = np.array(trainingData_df)
y = np.array(label_df)

##Polynomial features
#polynomial_features = PolynomialFeatures(degree=2)
#X = polynomial_features.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=29)

splits = 2
folds = KFold(n_splits = splits)
hyper_params = [{'n_features_to_select': list(range(len(predictData_df.columns)-24, len(predictData_df.columns)+1)), 'step':[5]}]
clf = LinearRegression(n_jobs=-1).fit(X_train, y_train)
rfe = RFE(clf)
clf = GridSearchCV(estimator = rfe, param_grid = hyper_params, scoring= 'neg_median_absolute_error', cv = folds, verbose = 3, return_train_score=True, n_jobs=-1)   
clf.fit(X_train, y_train)
cv_results = pd.DataFrame(clf.cv_results_)
cv_results.to_csv(savePath + "bestParam.csv", encoding='utf-8', index = False)

with open(savePath+'LR.pickle','wb') as f:
    pickle.dump(clf,f)

'''
pickle_in = open('LR.pickle','rb')
clf = pickle.load(pickle_in)
'''

y_test_pred = clf.predict(X_test)
accuracy = round(r2_score(y_test,y_test_pred)*100,2)
print(accuracy)
f = open(savePath+str(accuracy) + ".txt","w+")
f.write("RMSE : "+str(mean_squared_error(y_test, y_test_pred, squared=False))+"\n")
f.write("MAE : "+str(mean_absolute_error(y_test, y_test_pred)))
f.close()

X = np.array(predictData_df)

##Polynomial features
#X = polynomial_features.fit_transform(X)

forecast = clf.predict(X)
save_df['points'] = forecast
save_df = pd.pivot_table(save_df, values=['points'], index=['player_name', 'player_team','element_type','element','round','value'], aggfunc=np.sum).reset_index()

df = save_df.sort_values(["round","points"], ascending=[True,False])
df.to_csv(savePath + "PredictLR.csv", encoding='utf-8', index = False)