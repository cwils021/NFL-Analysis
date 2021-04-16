import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 100)

stadium_data = pd.read_csv("https://raw.githubusercontent.com/cwils021/NFL-Analysis/main/Datasets/KaggleData/nfl_stadiums.csv", encoding = "latin-1")
game_data = pd.read_csv('https://raw.githubusercontent.com/cwils021/NFL-Analysis/main/Datasets/KaggleData/spreadspoke_scores.csv')
teams = pd.read_csv('https://raw.githubusercontent.com/cwils021/NFL-Analysis/main/Datasets/KaggleData/nfl_teams.csv')
elo = pd.read_csv("https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv")
nfl_scores_and_predictors = pd.read_excel("https://github.com/cwils021/NFL-Analysis/blob/main/NFL_final_dataset.xlsx?raw=true")


# stadium_profile = ProfileReport(stadium_data, title="Pandas Profiling Report for Stadium Data")
# stadium_profile.to_file("stadium_data.html")

# teams_profile = ProfileReport(teams, title="Pandas Profiling Report for Teams Data")
# teams_profile.to_file("teams_data.html")

# game_data_profile = ProfileReport(game_data, title="Pandas Profiling Report for Game Data")
# game_data_profile.to_file("game_data.html")

# elo_profile = ProfileReport(elo, title="Pandas Profiling Report for ELO Data")
# elo_profile.to_file("elo_data.html")


# drop rows where betting data is not available
over_under_mask = nfl_scores_and_predictors['over_under_line'].isna()

print(len(nfl_scores_and_predictors.loc[over_under_mask]))
print(len(nfl_scores_and_predictors.drop(nfl_scores_and_predictors[over_under_mask].index)))

nfl_scores_and_predictors = nfl_scores_and_predictors.drop(nfl_scores_and_predictors[over_under_mask].index)

# nfl_scores_and_predictors_profile = ProfileReport(nfl_scores_and_predictors, title="final dataset")
# nfl_scores_and_predictors_profile.to_file("final_ds_data.html")

print(nfl_scores_and_predictors.columns)
'''

'game_id', 'date_string', 'schedule_date', 'schedule_season',
       'schedule_week', 'schedule_playoff', 'home_team_id', 'home_city',
       'home_teamname', 'away_city', 'away_teamname', 'away_team_id', 'result',
       'team_home', 'score_home', 'score_away', 'team_away',
       'team_favorite_id', 'spread_favorite', 'over_under_line', 'stadium',
       'address', 'stadium_neutral', 'dt_for_home', 'dt_for_away',
       'bearing_away', 'bearing_home', 'compass_away', 'compass_home', 'team1',
       'team2', 'elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'qbelo1_pre',
       'qbelo2_pre', 'qb1', 'qb2', 'qb1_value_pre', 'qb2_value_pre',
       'qbelo_prob1', 'qbelo_prob2', 'home_fav'

'''

model_data = nfl_scores_and_predictors[[ 'result', 'spread_favorite', 'over_under_line','stadium_neutral', 
'dt_for_home', 'dt_for_away', 'compass_away', 'compass_home', 'elo1_pre', 'elo2_pre', 'elo_prob1', 
'elo_prob2', 'qbelo1_pre','qbelo2_pre',  'qb1_value_pre', 'qb2_value_pre','qbelo_prob1',
 'qbelo_prob2', 'home_fav' ]]

print(model_data.head())

# model_profile = ProfileReport(model_data, title = "model dataset profile")
# model_profile.to_file("model_data_profile.html")


model_data_final = model_data[['result', 'stadium_neutral', 'dt_for_away', 'dt_for_home', 'compass_away', 'qbelo1_pre',
'qbelo2_pre', 'qb1_value_pre', 'qb2_value_pre', 'home_fav', 'spread_favorite' ]]
model_data_final = model_data_final.reset_index()
print(model_data_final.head())

home_dist_mask = model_data_final['dt_for_home'] == 0

model_data_final.loc[home_dist_mask, 'dt_for_home'] += 1

# model_profile_final = ProfileReport(model_data_final, title = "model dataset final profile")
# model_profile_final.to_file("model_data__final_profile.html")

# y = model_data_final['result']
# X = model_data_final.drop(['result'], axis = 1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
train, test = train_test_split(model_data_final, test_size=0.25, random_state=0)

print(train.head(1))
print(test.head(1))
print(f" train shape: {train.shape}")
print(f" test shape: {test.shape}")
print(f"length of model_data_final: {len(model_data_final)}, length of train + test: {len(train) + len(test)}")

train.to_excel("train.xlsx")
test.to_excel("test.xlsx")
