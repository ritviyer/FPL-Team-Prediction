import requests
import pandas as pd
import numpy as np
import os

def GetPlayerData(path):
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()

    elements_df = pd.DataFrame(json['elements'])
    elements_types_df = pd.DataFrame(json['element_types'])

    elements_df.to_csv(path + 'players_raw.csv', encoding='utf-8', index = False)
    elements_types_df.to_csv(path + 'players_type.csv', encoding='utf-8', index = False)
    

def GetTeams(path):
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    r = requests.get(url)
    json = r.json()

    teams_df = pd.DataFrame(json['teams'])
    teams_df.to_csv(path + 'teams.csv', encoding='utf-8', index = False)

def GetFixtures(path):
    url = 'https://fantasy.premierleague.com/api/fixtures/'
    r = requests.get(url)
    json = r.json()

    fixtures_df = pd.DataFrame(json)
    fixtures_df.to_csv(path + 'fixtures.csv', encoding='utf-8', index = False)

def GetPlayerHistoricData(path, playerData):
    url = "https://fantasy.premierleague.com/api/element-summary/{}/"
    players = os.listdir(path)
    for ind in playerData.index: 
        playerPath = playerData['first_name'][ind] + '_' + playerData['second_name'][ind] + '_' + str(playerData['id'][ind])
        if playerPath not in players:
            playerURL = url.format(str(playerData['id'][ind]))
            r = requests.get(playerURL)
            json = r.json()
            history_df = pd.DataFrame(json['history_past'])
            if not history_df.empty:
                os.mkdir(path + playerPath)
                history_df.to_csv(path + playerPath + '/history.csv', encoding='utf-8', index = False)
        else:
            if not os.path.isfile(path + playerPath + "/history.csv"):
                playerURL = url.format(str(playerData['id'][ind]))
                r = requests.get(playerURL)
                json = r.json()
                history_df = pd.DataFrame(json['history_past'])
                if not history_df.empty:
                    history_df.to_csv(path + playerPath + '/history.csv', encoding='utf-8', index = False)


def GetPlayerGameweekData(path, playerData):
    url = "https://fantasy.premierleague.com/api/element-summary/{}/"
    players = os.listdir(path)
    for ind in playerData.index: 
        playerPath = playerData['first_name'][ind] + '_' + playerData['second_name'][ind] + '_' + str(playerData['id'][ind])
        playerURL = url.format(str(playerData['id'][ind]))
        r = requests.get(playerURL)
        json = r.json()
        history_df = pd.DataFrame(json['history'])
        if not history_df.empty:
            if playerPath not in players:
                os.mkdir(path + playerPath)
            history_df.to_csv(path + playerPath + '/gw.csv', encoding='utf-8', index = False)

