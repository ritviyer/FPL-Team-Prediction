import aiohttp
from understat import Understat
import pandas as pd

async def test():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_team_fixtures(
            "Arsenal",
            2020
        )
        df = pd.DataFrame(player)
        print(df)
        #df = df['2019'].tolist()
        #df = [x for x in df if x == x]
        #print(df)
        #df= pd.DataFrame(df)
        #print(df)
        #df = pd.pivot_table(df, values='xG', index=['season'],columns=['situation'], aggfunc=np.sum).reset_index()
        #print(df)
        ##df.to_csv('groupedPlayer.csv', encoding='utf-8', index = False)

async def Get_League_Players(year,team):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_league_players("epl", year, team_title = team)
        df = pd.DataFrame(player)
        return df


async def Get_Player_Matches(id,year):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_player_matches(id, season = year)
        df = pd.DataFrame(player)
        return df


async def Get_Player_Shots(id,year):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_player_shots(id,season=year)
        df = pd.DataFrame(player)
        return df


async def Get_Teams(year, team):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_teams( "epl",year, title=team)
        df = pd.DataFrame(player)
        if not df.empty:
            df = df.history
            df = df[0]
            df = pd.DataFrame(df)
        return df


async def Get_Team_Results(team,year):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        player = await understat.get_team_results(team,year)
        df = pd.DataFrame(player)
        return df

#import asyncio
#import numpy as np
#loop = asyncio.get_event_loop()
#loop.run_until_complete(test())