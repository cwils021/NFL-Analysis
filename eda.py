import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport


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

# nfl_scores_and_predictors_profile = ProfileReport(nfl_scores_and_predictors, title="Pandas Profiling Report for Final Dataset")
# nfl_scores_and_predictors_profile.to_file("final_ds_data.html")
