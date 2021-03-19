import pandas as pd
import numpy as np
import pulp


def optimize_team(expected_scores, prices, positions, clubs, penalty, total_budget=100, sub_factor=0):
    num_players = len(expected_scores)
    model = pulp.LpProblem("FPL Points Optimization", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    captain_decisions = [
        pulp.LpVariable("y{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    sub_decisions = [
        pulp.LpVariable("z{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]


    # objective function:
    model += sum(((captain_decisions[i] + decisions[i] + sub_decisions[i]*sub_factor) * expected_scores[i]) - ((decisions[i] + sub_decisions[i]) * penalty[i])
                 for i in range(num_players)), "Objective"

    # cost constraint
    model += sum((decisions[i] + sub_decisions[i]) * prices[i] for i in range(num_players)) <= total_budget  # total cost

    # position constraints
    # 1 starting goalkeeper
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 1) == 1
    # 2 total goalkeepers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 1) == 2

    # 3-5 starting defenders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 2) <= 5
    # 5 total defenders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 2) == 5

    # 3-5 starting midfielders
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) >= 3
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 3) <= 5
    # 5 total midfielders
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 3) == 5

    # 1-3 starting attackers
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) >= 1
    model += sum(decisions[i] for i in range(num_players) if positions[i] == 4) <= 3
    # 3 total attackers
    model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if positions[i] == 4) == 3

    # club constraint
    for club_id in np.unique(clubs):
        model += sum(decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 2  # max 2 players from one team in starting 11
        model += sum(decisions[i] + sub_decisions[i] for i in range(num_players) if clubs[i] == club_id) <= 3  # max 3 players overall

    model += sum(decisions) == 11  # total team size
    model += sum(captain_decisions) == 1  # 1 captain
    
    for i in range(num_players):  
        model += (decisions[i] - captain_decisions[i]) >= 0  # captain must also be on team
        model += (decisions[i] + sub_decisions[i]) <= 1  # subs must not be on team

    model.solve()
    print("Total expected score = {}".format(model.objective.value()))

    return decisions, captain_decisions, sub_decisions


def SelectTeam(current_team, playerData_df, penal, balance,subFactor=0):
    expectedScores = playerData_df['points']
    prices = playerData_df['value']
    positions = playerData_df['element_type']
    clubs = playerData_df['player_team']
    names = playerData_df['player_name']
    element = playerData_df['element']

    print(playerData_df[playerData_df.element.isin(current_team)])
    budget = playerData_df[playerData_df.element.isin(current_team)].value.sum() + balance
    playerData_df['penalty'] = penal
    for sel in current_team:
        indexs = playerData_df[playerData_df['element'] == sel].index
        playerData_df.loc[indexs, 'penalty'] = 0
    penalty = playerData_df['penalty']


    decisions, captain, subs = optimize_team(expectedScores.values, prices.values, positions.values, clubs.values, penalty.values, budget, subFactor)
    new_team=[]
    print("Players in Starting X1\n")
    for i in range(playerData_df.shape[0]):
        if decisions[i].value()!=0:
            new_team.append(element[i])
            print("**{}** {} Points = {}, Price = {}".format(names[i],element[i],expectedScores[i], prices[i]))

    print("Captain\n")
    for i in range(playerData_df.shape[0]):
        if captain[i].value()==1:
            print("**Captain {}** {} Points = {}, Price = {}".format(names[i],element[i],expectedScores[i], prices[i]))


    print("Substitute Players\n")
    for i in range(playerData_df.shape[0]):
        if subs[i].value()!=0:
            new_team.append(element[i])
            print("**Subs {}** {} Points = {}, Price = {}".format(names[i],element[i],expectedScores[i], prices[i]))

    print("\n\nTotal Budget: ",budget)
    balance = budget - playerData_df[playerData_df.element.isin(new_team)].value.sum()
    print("New Team Value: ",playerData_df[playerData_df.element.isin(new_team)].value.sum())
    print("New Balance: ", balance)
    playerData_df = playerData_df.drop(columns=['penalty','points'])
    removed = [x for x in current_team if x not in new_team]
    added = [x for x in new_team if x not in current_team]
    removed_df = playerData_df[playerData_df['element'].isin(removed)]
    added_df = playerData_df[playerData_df['element'].isin(added)]
    newTeam_df = playerData_df[playerData_df['element'].isin(new_team)]
    newTeam_df = pd.concat([newTeam_df,removed_df])
    newTeam_df.loc[playerData_df['element'].isin(added),"transfers"] = "IN"
    newTeam_df.loc[playerData_df['element'].isin(removed),"transfers"] = "OUT"
    newTeam_df.loc[playerData_df['element'].isin(new_team[-4:]),"Substitute"] = "YES"
    return new_team, newTeam_df

method = 'RandomForest/'
round = 29
balance = 0.8
playerData_df = pd.read_csv('./prediction/Gameweeks/'+str(round)+'/prediction/'+ method +'PredictRF.csv')
my_team = pd.read_csv('./prediction/Gameweeks/'+str(round-1)+'/prediction/'+method+'PredictedTeam.csv')
indexNames = my_team[ my_team['transfers'] == 'OUT' ].index
my_team.drop(indexNames , inplace=True)
playerData_df['value'] = playerData_df.value.astype(float)
my_team['value'] = my_team.value.astype(float)
my_team['value'] = np.round(my_team['value'],1)
playerData_df['value'] = np.round(playerData_df['value'],1)
playerData_df['selling_value'] = playerData_df['value']
playerData_df['selling_value'] = ((playerData_df['element'].map(my_team.set_index('element')['value']) + playerData_df['selling_value'])/2).fillna(playerData_df['selling_value'])
playerData_df['selling_value'] = (playerData_df['selling_value']*10).astype(int)/10

#comment for only this round
playerData_df['value'] = np.where(playerData_df['value']<playerData_df['selling_value'],playerData_df['value'],playerData_df['selling_value'])

my_team = my_team['element'].tolist()
playerRaw_df = pd.read_csv('./current year/2020-21/players_raw.csv')
playerData_df['next_match'] = playerData_df.element.map(playerRaw_df.set_index('id').chance_of_playing_next_round)
playerData_df['ep_next'] = playerData_df.element.map(playerRaw_df.set_index('id').ep_next)
playerData_df['penalty'] = playerData_df.element.map(playerRaw_df.set_index('id').penalties_order)
playerData_df['freekick'] = playerData_df.element.map(playerRaw_df.set_index('id').direct_freekicks_order)
playerData_df['corner'] = playerData_df.element.map(playerRaw_df.set_index('id').corners_and_indirect_freekicks_order)

playerData_df.loc[(playerData_df['round']==round) & (playerData_df['next_match']<=50),"points"] = 0
#For Free Hit, to get all players
#playerData_df.loc[(playerData_df['round']==round+1),"points"] = 0
playerData_df.loc[(playerData_df['round']==round) & (playerData_df['ep_next']<1),"points"] = \
    playerData_df.loc[(playerData_df['round']==round) & (playerData_df['ep_next']<1),"points"]\
   * playerData_df.loc[(playerData_df['round']==round) & (playerData_df['ep_next']<1),"ep_next"]\
   * (2 - playerData_df.loc[(playerData_df['round']==round) & (playerData_df['ep_next']<1),"ep_next"])
playerData_df.loc[(playerData_df['penalty']==1),"points"] = playerData_df.loc[(playerData_df['penalty']==1),"points"] * 1.02
playerData_df.loc[(playerData_df['freekick']==1),"points"] = playerData_df.loc[(playerData_df['freekick']==1),"points"] * 1.02
playerData_df.loc[(playerData_df['corner']==1),"points"] = playerData_df.loc[(playerData_df['corner']==1),"points"] * 1.02

playerData_df = playerData_df.drop(columns=['next_match','penalty','freekick','corner','ep_next','selling_value'])

data1 = playerData_df[playerData_df['round'] == round].reset_index(drop=True)
data2 = playerData_df[playerData_df['round'] == (round+1)].reset_index(drop=True)
data3 = playerData_df[playerData_df['round'] == (round+2)].reset_index(drop=True)
data4 = playerData_df[playerData_df['round'] == (round+3)].reset_index(drop=True)

playerData_df = pd.pivot_table(playerData_df, values=['points'], index=['player_name', 'player_team','element_type','element','value'], aggfunc=np.sum).reset_index()


rone = pd.pivot_table(pd.concat([data1,data2]), values=['points'], index=['player_name', 'player_team','element_type','element','value'], aggfunc=np.sum).reset_index()
rtwo = pd.pivot_table(pd.concat([data2,data3]), values=['points'], index=['player_name', 'player_team','element_type','element','value'], aggfunc=np.sum).reset_index()
rthree = pd.pivot_table(pd.concat([data3,data4]), values=['points'], index=['player_name', 'player_team','element_type','element','value'], aggfunc=np.sum).reset_index()

my_team, saveTeam_df = SelectTeam(my_team,rone,0,balance,0.2)

#my_team, saveTeam_df = SelectTeam(my_team,data1,100,0)
#my_team = SelectTeam(my_team,data3,1)
#my_team = SelectTeam(my_team,data4,1)

my_team = pd.read_csv('./prediction/Gameweeks/'+str(round-1)+'/prediction/'+method+'PredictedTeam.csv')
indexNames = my_team[ my_team['transfers'] == 'OUT' ].index
my_team.drop(indexNames , inplace=True)

#comment for only this round
saveTeam_df['value'] = saveTeam_df['element'].map(my_team.set_index('element')['value']).fillna(saveTeam_df['value'])
saveTeam_df.to_csv('./prediction/Gameweeks/'+str(round)+'/prediction/'+method+'PredictedTeam.csv', index=False)

#saveTeam_df.to_csv('./prediction/Gameweeks/'+str(round)+'/prediction/'+method+'PredictedTeamOnlyThisRound.csv', index=False)


my_team = pd.read_csv('./prediction/Gameweeks/'+str(round)+'/prediction/'+method+'PredictedTeam.csv')
indexNames = my_team[ my_team['transfers'] == 'OUT' ].index
my_team_list = my_team.drop(indexNames)
my_team_list = my_team_list['element'].tolist()
my_team_list, saveTeam_df = SelectTeam(my_team_list,data1,100,0)
my_team['Substitute'] = my_team.element.map(saveTeam_df.set_index('element').Substitute)
my_team.to_csv('./prediction/Gameweeks/'+str(round)+'/prediction/'+method+'PredictedTeam.csv', index=False)
