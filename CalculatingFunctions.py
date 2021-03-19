import pandas as pd
import numpy as np


def shiftRows(epl_df, colNames):
    for col in colNames:
        epl_df[col] = epl_df[col].shift(1)
    return epl_df

def GetNanValues(playerData_df):
    emptyCols = playerData_df.columns[playerData_df.isnull().any()].tolist()
    playerData_df[emptyCols] = playerData_df[emptyCols].apply(pd.to_numeric)
    playerData_df[emptyCols] = playerData_df[emptyCols].fillna(playerData_df[emptyCols].mean())
    return playerData_df

def PercentSelected(playerData_df, year):
    totalPlayers = 4000000
    if year == 2017:
        totalPlayers = 5500000
    elif year == 2018:
        totalPlayers = 5800000
    elif year == 2019:
        totalPlayers = 6250000
    elif year == 2020:
        totalPlayers = 8000000

    playerData_df['selected'] = (playerData_df['selected']/totalPlayers)*100
    playerData_df.rename(columns={"selected": "selected_by_percent"}, inplace = True)
    return playerData_df

def RunningAverage90(df,prev,params):
    df['avg_matches'] = df['minutes'].rolling(min_periods=1, window=prev).sum()/90
    df['avg_matches'] = np.where(df['avg_matches'] == 0, 1, df['avg_matches'])
    newParams = []
    for para in params:
        newParams.append(para + '_last_' + str(prev))
    df[newParams] = df[params].rolling(min_periods=1, window=prev).sum()
    for npara in newParams:
        df[npara] = df[npara]/df['avg_matches']
    df = df.drop(columns=['avg_matches'])
    return df

def RunningAverage(df,prev,params):
    newParams = []
    for para in params:
        newParams.append(para + '_last_' + str(prev))
    df[newParams] = df[params].rolling(min_periods=1, window=prev).mean()
    return df