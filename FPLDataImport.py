import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
path = './current year/2020-21'
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
r = requests.get(url)
json = r.json()
#get keys that we are concerned with - elements(Players), elements_type(position details), teams.
elements_df = pd.DataFrame(json['elements'])
elements_types_df = pd.DataFrame(json['element_types'])
teams_df = pd.DataFrame(json['teams'])

elements_df.to_csv(path + 'players_raw.csv', encoding='utf-8', index = False)
elements_types_df.to_csv(path + 'elements_types.csv', encoding='utf-8', index = False)
teams_df.to_csv(path + 'teams.csv', encoding='utf-8', index = False)

##smaller data frame with certain required columns
#slim_elements_df = elements_df[['second_name','team','element_type','selected_by_percent','now_cost','minutes','transfers_in','value_season','total_points']]

##add a new column position of the player. Map function works like vlookup in excel
#slim_elements_df['position'] = slim_elements_df.element_type.map(elements_types_df.set_index('id').singular_name)

##Add team name insead of ID
#slim_elements_df['team'] = slim_elements_df.team.map(teams_df.set_index('id').name)

##Change all values to same data type
#slim_elements_df['value'] = slim_elements_df.value_season.astype(float)

##Update the data fram after sorting the data in decreasing order of value
##slim_elements_df = slim_elements_df.sort_values('value',ascending=False)

##remove players who have played 0 minutes
#slim_elements_df = slim_elements_df.loc[slim_elements_df.value > 0]

##pivot elements on position
#pivot_position = slim_elements_df.pivot_table(index='position', values='value', aggfunc = np.mean).reset_index()
#pivot_team = slim_elements_df.pivot_table(index='team', values='value', aggfunc = np.mean).reset_index()

##filtered dataframe for each position
#fwd_df = slim_elements_df.loc[slim_elements_df.position == 'Forward']
#mid_df = slim_elements_df.loc[slim_elements_df.position == 'Midfielder']
#def_df = slim_elements_df.loc[slim_elements_df.position == 'Defender']
#goal_df = slim_elements_df.loc[slim_elements_df.position == 'Goalkeeper']

##plot histogram
#def_df.value.hist()
##plt.show()

##export to CSV
##slim_elements_df.to_csv('fplData.csv', encoding='utf-8', index = False)
