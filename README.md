# FPL-Team-Prediction
This project uses machine learning to predict the fantasy premier league performance of each player.

** current year/ - ** Current season's FPL data of each player
** data/ - ** Historical FPL data of each player
** prediction/ - ** Each gameweek's prediction data as well as model predictions
** training-data_2016-19/ - ** Training data from previos seasons
** CalculatingFunctions.py - ** Functions to shift rows, fill Nan values, calculate average of features in last few select gws
** FetchFPLData.py - ** Functions to fetch team and player data using FPL APIs - (Fixture, gw data, player history, etc)
** FetchUnderstatData.py - ** Functions to fetch team and player data from understat - (player xG, xA, team results, offensive/defensive form, etc)
** GBMFPLModel.py - ** Build an XGBoost model for points prediction
** LinearRegressionFPLModel.py - ** Build a Linear Regression model for points prediction
** RandomForrestFPLModel.py - ** Build a Random Forest model for points prediction
** GetTeamPoints.py - ** Fetch gameweek points of each player using FPL APIs
** MapUnderstatToFPL.py - ** Functions to match and Consolidate the player's FPL and understat data
** PickTeam.py - ** Use Linear Programming to pick optimal team
** PreparePredictionData.py - ** Gather data for each player to make a prediction for the following gameweek
** PrepareTrainingData.py - ** Gather historical data for each player for training the models
** ReadFPLData.py - ** Read and process the collected FPL data of each player

For each gameweek, the models are trained using all historical data prior to that week starting from the 2016/17 season.
Some of the features used are:
* Player performance in the season so far, ex - goals, assists, clean sheets, bonus points, minutes played, etc
* Understat player data such as xG, xA and more
* Offensive and defensive form of the player's team
* Offensive and defensive form of the opponent team

After points forecast, Linear Programming is used to find an optimal team.

A comparison of the performance of all models will be added at the end of the season.

## Acknowledgments
** Sincere thanks to: ** 
* https://github.com/vaastav/Fantasy-Premier-League for providing historical FPL data of each player
* https://github.com/amosbastian/understat for providing a Python package for Understat