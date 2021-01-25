import FetchUnderstatData as fud
import pandas as pd
import numpy as np
import asyncio
from difflib import SequenceMatcher



def GetUnderstatID(playerData_df,yr):
    team = playerData_df['player_team'].values[0]
    #use if only for 1st gameweek
    #if not (team == "Aston Villa" or team=="Manchester United" or team=="Manchester City" or team=="Burnley"):
    loop = asyncio.get_event_loop()
    understatPlayerData_df = loop.run_until_complete(fud.Get_League_Players(yr,team))
    #Map FPL and Understat data using player name that match exactly
    playerData_df['understat_id'] = playerData_df.player_name.map(understatPlayerData_df.set_index('player_name').id)

    #For those name that don't exactly match
    #Add FPL ID to understat Data
    eplPivotTable_df = playerData_df.dropna()
    eplPivotTable_df['understat_id'] = eplPivotTable_df.understat_id.astype(int)
    eplPivotTable_df = pd.pivot_table(eplPivotTable_df,index=["player_name"], values = ["understat_id","element"])
    understatPlayerData_df['id'] = understatPlayerData_df.id.astype(int)
    understatPlayerData_df['fpl_id'] = understatPlayerData_df.id.map(eplPivotTable_df.set_index('understat_id').element)
    
    #Get unmapped players
    fplData = playerData_df.loc[pd.isna(playerData_df['understat_id'])]
    understatData = understatPlayerData_df.loc[pd.isna(understatPlayerData_df['fpl_id'])]

    pivotTable_df = pd.pivot_table(fplData,index=["player_name","player_team","web_name"], values = ["element"]).reset_index()
    pivotTable_df["understat_id"] = np.nan


    for i in understatData.index:
        playerName = understatData['player_name'][i]
        playerName = playerName.split()
        playerTeam = understatData['team_title'][i]
        for j in pivotTable_df.index:
            if pivotTable_df['player_team'][j] == playerTeam:
                match = 0
                for name in playerName:
                    if name in pivotTable_df['player_name'][j]:
                        match = match + 1
                    else:
                        s = SequenceMatcher(None, name, pivotTable_df['player_name'][j])
                        if(s.ratio()>0.95):
                            match = match + 1
                        else:
                            match = match - 1
                if match >= 1:
                    pivotTable_df['understat_id'][j] = understatData['id'][i]
                    understatData['fpl_id'][i] = pivotTable_df['element'][j]
                    break

    #Map not very close matching names
    df_epl = pivotTable_df.loc[pd.isna(pivotTable_df['understat_id'])]
    df_understat = understatData.loc[pd.isna(understatData['fpl_id'])]
    for i in df_understat.index:
        playerName = df_understat['player_name'][i]
        playerTeam = df_understat['team_title'][i]
        highest = 0.5
        for j in df_epl.index:
            if df_epl['player_team'][j] == playerTeam:
                if df_epl['web_name'][j] == playerName:
                    df_epl['understat_id'][j] = df_understat['id'][i]
                    df_understat['fpl_id'][i] = df_epl['element'][j]
                    break
                else:
                    s = SequenceMatcher(None, playerName, df_epl['player_name'][j])
                    if(s.ratio()>=highest):
                        highest = s.ratio()
                        df_epl['understat_id'][j] = df_understat['id'][i]
                        df_understat['fpl_id'][i] = df_epl['element'][j]
    

    #Players which aren't mapped, if needed later
    pdf1 = df_epl.loc[df_epl['understat_id'].isna()]
    pdf2 = df_understat.loc[df_understat['fpl_id'].isna()]

    #Consolidate not perfect matches
    if not pivotTable_df.empty:
        pivotTable_df['element'] = pivotTable_df.element.astype(int)
        df_epl['element'] = df_epl.element.astype(int)
        pivotTable_df['understat_id_1'] = pivotTable_df.element.map(df_epl.set_index('element').understat_id)
        pivotTable_df = pivotTable_df.fillna(0)
        pivotTable_df["understat_id"] = pivotTable_df["understat_id"].astype(int) + pivotTable_df["understat_id_1"].astype(int)

        #Add consolidated names above to perfect amtched names
        playerData_df['element'] = playerData_df.element.astype(int)
        playerData_df['understat_id_1'] = playerData_df.element.map(pivotTable_df.set_index('element').understat_id)
        playerData_df = playerData_df.fillna(0)
        playerData_df["understat_id"] = playerData_df["understat_id"].astype(int) + playerData_df["understat_id_1"].astype(int)
        playerData_df = playerData_df.drop(columns=['understat_id_1'])
    
    indexNames = playerData_df[ playerData_df['understat_id'] == 0 ].index
    playerData_df.drop(indexNames , inplace=True)
    return playerData_df

def understatPlayerStats(epl_df, yr):
    id = epl_df['understat_id'].values[0]
    
    #Get player match statistics
    loop = asyncio.get_event_loop()
    un_match_df = loop.run_until_complete(fud.Get_Player_Matches(id, yr))
    if not un_match_df.empty:
        un_match_df = un_match_df[['h_team','a_team','shots', 'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain','xGBuildup']]
        df_ret = pd.merge(epl_df, un_match_df, on=['h_team', 'a_team'], how='left')
    else:
        df_ret = epl_df

    ##Get player shot statistics
    #loop = asyncio.get_event_loop()
    #un_shots_df = loop.run_until_complete(fud.Get_Player_Shots(id, yr))
    #if not un_shots_df.empty:
    #    un_shots_df['xG'] = un_shots_df.xG.astype(float)
    #    table1_df = pd.pivot_table(un_shots_df, values='xG', index=['h_team', 'a_team'],columns=['situation'], aggfunc=np.sum).reset_index()
    #    table2_df = pd.pivot_table(un_shots_df, values='xG', index=['h_team', 'a_team','result'],columns=['situation'], aggfunc="count").reset_index()
    #    table2_df = table2_df.loc[table2_df.result == 'Goal']
    #    table2_df = table2_df.drop(columns=['result'])
        
    #    un_shots_df = pd.merge(table1_df, table2_df, on=['h_team', 'a_team'], suffixes = ("_player_xG","_player_goals"), how='left')

    #    un_match_df = pd.merge(un_match_df, un_shots_df, on=['h_team', 'a_team'], how='left')

    #    df_ret = pd.merge(epl_df, un_match_df, on=['h_team', 'a_team'], how='left')
    #else:
    #    df_ret = pd.merge(epl_df, un_match_df, on=['h_team', 'a_team'], how='left')
    return df_ret


def understatTeamStats(epl_df, yr, group, round = 39):
    team = epl_df[group].values[0]
    loop = asyncio.get_event_loop()
    un_team_df = loop.run_until_complete(fud.Get_Teams(yr, team))
    hTeam = np.nan
    aTeam = np.nan

    if round < 39:
        df = epl_df[epl_df['round'] == round].reset_index(drop=True)
        if not df.empty:
            index = df.index[0]
            hTeam = df.loc[index, 'h_team']
            aTeam = df.loc[index, 'a_team']


    df_ret = pd.DataFrame()
    if not un_team_df.empty:
        data = pd.DataFrame(un_team_df.ppda.tolist())
        un_team_df['ppda_att'] = data['att']
        un_team_df['ppda_def'] = data['def']
        data = pd.DataFrame(un_team_df.ppda_allowed.tolist())
        un_team_df['ppda_allowed_att'] = data['att']
        un_team_df['ppda_allowed_def'] = data['def']
        un_team_df = un_team_df.drop(columns=['ppda', 'ppda_allowed','h_a','result','xpts','wins','draws','loses','pts'])

        loop = asyncio.get_event_loop()
        un_team_result_df = loop.run_until_complete(fud.Get_Team_Results(team,yr))
        un_team_result_df = un_team_result_df[['h','a','datetime']]
        data = pd.DataFrame(un_team_result_df.h.tolist())
        un_team_result_df['h'] = data['title']
        data = pd.DataFrame(un_team_result_df.a.tolist())
        un_team_result_df['a'] = data['title']
        un_team_result_df.rename(columns={"datetime": "date", "h": "h_team", "a": "a_team"}, inplace = True)
        un_team_result_df['h_team'] = un_team_result_df.h_team.shift(-1)
        un_team_result_df['a_team'] = un_team_result_df.a_team.shift(-1)
        un_team_result_df['h_team'] = un_team_result_df['h_team'].fillna(hTeam)
        un_team_result_df['a_team'] = un_team_result_df['a_team'].fillna(aTeam)

        un_team_df = pd.merge(un_team_df, un_team_result_df, on=['date'], how='left')
        un_team_df = un_team_df.drop(columns=['date'])
        un_team_df = un_team_df.add_suffix('_' + group)
        un_team_df.rename(columns={"h_team_" + group: "h_team", "a_team_" + group: "a_team"}, inplace = True)
        df_ret = pd.merge(epl_df, un_team_df, on=['h_team', 'a_team'], how='left')
    else:
        df_ret = epl_df
    return df_ret


def understatPlayerHistoricStats(epl_df, yr, keepCols):
    id = epl_df['understat_id'].values[0]
    yr = str(yr - 1)
    index = epl_df.index[0]

    #Get player match statistics
    loop = asyncio.get_event_loop()
    un_match_df = loop.run_until_complete(fud.Get_Player_Matches(id, yr))
    if not un_match_df.empty:
        un_match_df = un_match_df[keepCols]
        un_match_df = un_match_df.apply(pd.to_numeric)
        epl_df.loc[index, keepCols] = un_match_df[keepCols].mean()

    return epl_df


def understatTeamHistoricStats(epl_df, yr, group):
    yr = yr-1
    team = epl_df[group].values[0]
    loop = asyncio.get_event_loop()
    un_team_df = loop.run_until_complete(fud.Get_Teams(yr, team))
    
    if not un_team_df.empty:
        data = pd.DataFrame(un_team_df.ppda.tolist())
        un_team_df['ppda_att'] = data['att']
        un_team_df['ppda_def'] = data['def']
        data = pd.DataFrame(un_team_df.ppda_allowed.tolist())
        un_team_df['ppda_allowed_att'] = data['att']
        un_team_df['ppda_allowed_def'] = data['def']
        un_team_df = un_team_df.drop(columns=['ppda', 'ppda_allowed','h_a','result','xpts','wins','draws','loses','pts','date'])
        un_team_df = un_team_df.add_suffix('_' + group)
        un_team_df = un_team_df.apply(pd.to_numeric)
        cols = un_team_df.columns.tolist()
        #uncomment bellow line only for rprediction data of gw1
        #epl_df[cols] = np.nan
        if(set(cols).issubset(set(epl_df.columns.tolist()))): 
            epl_df[cols] = epl_df[cols].fillna(un_team_df[cols].mean())
        else:
            epl_df[cols] = np.nan
            epl_df[cols] = epl_df[cols].fillna(un_team_df[cols].mean())

    return epl_df


def understatMultipleFixtureTeamStats(epl_df, group, round = 39):
    cols = ['xG', 'xGA', 'npxG', 'npxGA', 'deep', 'deep_allowed', 'scored', 'missed', 'npxGD', 'ppda_att', 'ppda_def', 'ppda_allowed_att', 'ppda_allowed_def']
    suf = '_' + group
    cols = [sub + suf for sub in cols]
    data = epl_df.loc[~epl_df['xG'+suf].isna()] 
    data = data.loc[data['round'] == round]
    if not data.empty:
        index = data.index[0]
        data = data.loc[index, cols]

    to_fill = epl_df.loc[epl_df['xG'+suf].isna()]
    if not to_fill.empty:
        to_fill = to_fill.loc[to_fill['round'] == round]
        to_fill[cols] = to_fill[cols].fillna(data)

        ind = list(to_fill.index)
        s = len(ind)
        ind = [i for i in range(0,s)]

        epl_df.loc[to_fill.index[ind]] = to_fill.iloc[ind]

    return epl_df


def understatMultipleFixturePlayerStats(epl_df, cols, round):
    data = epl_df.loc[epl_df['round'] == round]
    rows = data.shape[0]
    if rows>1:
        index = data.index[0]
        data = data.loc[index, cols]
        epl_df.loc[epl_df['round'] == round,cols] = data.to_list()
    return epl_df