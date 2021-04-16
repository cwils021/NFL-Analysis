from datetime import datetime
print("data_cleaning.py starting")
start_time = datetime.now()
import numpy as np
import pandas as pd
import re
import collections
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy import distance
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import functions


pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 100)

print(f"imports successful({datetime.now() - start_time}s), time elapsed: {datetime.now() - start_time}s")
load_df_time = datetime.now()
# load the data into data frames 

stadium_data = pd.read_csv("https://raw.githubusercontent.com/cwils021/NFL-Analysis/main/Datasets/KaggleData/nfl_stadiums.csv", encoding = "latin-1")
game_data = pd.read_csv('https://raw.githubusercontent.com/cwils021/NFL-Analysis/main/Datasets/KaggleData/spreadspoke_scores.csv')
teams = pd.read_csv('https://raw.githubusercontent.com/cwils021/NFL-Analysis/main/Datasets/KaggleData/nfl_teams.csv')
elo = pd.read_csv("https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv")

print(f"dfs loaded successfully ({datetime.now() - load_df_time}s), time elapsed: {datetime.now() - start_time}s")
# clean up elo df
clean_time = datetime.now()
## Convert to date and drop pre 1966 season
elo['date'] = pd.to_datetime(elo.loc[:,'date'], infer_datetime_format=True)
date_cutoff = pd.to_datetime("1966-09-01")
_pre66_mask = (elo['date'] > date_cutoff) 
elo66 = elo.loc[_pre66_mask]    # elo df to use moving forward

# replace OAK with LVR (Oakland Raiders became LV raiders in 2020 season)
elo66 = elo66.replace({'team1':'OAK', 'team2':'OAK'},  'LVR')

# split city and team name into seperate columns
new = game_data['team_home'].str.rsplit(" ", 1, expand = True)
game_data["home_city"] = new[0]
game_data["home_teamname"] = new[1]
new = game_data['team_away'].str.rsplit(" ", 1, expand = True)
game_data["away_city"] = new[0]
game_data["away_teamname"] = new[1]

# add team id col
home_team_id = game_data['team_home'].replace(
    teams.set_index('team_name')['team_id'])
away_team_id = game_data['team_away'].replace(
    teams.set_index('team_name')['team_id'])
game_data.insert(4, "home_team_id", home_team_id)
game_data.insert(5, "away_team_id", away_team_id)

# rearrange column order
_temp = game_data[['home_city','home_teamname', 'away_city', 'away_teamname']]
game_data.drop(labels=['home_city','home_teamname', 'away_city', 'away_teamname'], axis = 1, inplace= True)
for i in range(4):
  col_name = _temp.columns[i]
  game_data.insert((i + 5),col_name , _temp.iloc[:,i])

# convert date to datetime and add string date column
game_data['schedule_date'] = pd.to_datetime(game_data['schedule_date'], infer_datetime_format= True)
_temp = game_data['schedule_date'].dt.strftime("%a %b %d, %Y")
game_data.insert(0, "date_string", _temp)

# add missing score from superbowl LV
_sb_lv_idx = game_data.loc[game_data['schedule_date'] == pd.to_datetime("2/7/2021")].index[0]
game_data.iat[_sb_lv_idx, game_data.columns.get_loc("score_away")] = 9 
game_data.iat[_sb_lv_idx, game_data.columns.get_loc("score_home")] = 31 

# convert scores to integer type
game_data[['score_home', 'score_away']] = game_data[['score_home', 'score_away']].astype(np.int64)

# create and add result column next to scores 1 = home team wins, 0 = away team wins
game_data['result'] = np.where(game_data['score_home'] > game_data['score_away'], 1, 0)
_temp = game_data['result']
game_data.drop(columns=['result'], inplace=True)
game_data.insert(11, "result", _temp)

# copy stadium names and addresses to new df
stadiums = stadium_data[['stadium_name', 'stadium_address' ]]

# merge stadium addresses with game_data df, dropping duplicate stadium name col
game_data = game_data.merge(stadiums, "left", left_on= "stadium", right_on="stadium_name",
                validate = "m:1").drop(
                    columns = ["stadium_name"])
# add stadium address column next to stadium name in game data
_temp = game_data['stadium_address']
game_data.drop(columns=['stadium_address'], inplace= True)
game_data.insert(20, "address", _temp)

# create distance travelled for
game_data["dt_for_home"] = 0
game_data["dt_for_away"] = -1

# fill missing addresses
allegiant_stadium = "3333 Al Davis Way, Las Vegas, NV 89118, United States"
alltel_stadium = "1 TIAA Bank Field Dr, Jacksonville, FL 32202, United States"
dolphin_stadium = "347 Don Shula Dr, Miami Gardens, FL 33056, United States"
jack_murphy = "9449 Friars Rd, San Diego, CA 92108, United States"
joe_robbie = "347 Don Shula Dr, Miami Gardens, FL 33056, United States"
merc_benz = "409 Nelson St SW, Atlanta, GA 30313, United States"
pro_player = "347 Don Shula Dr, Miami Gardens, FL 33056, United States"
rose_bowl = "1001 Rose Bowl Dr, Pasadena, CA 91103, United States"
sofi = "1000 S Prairie Ave, Inglewood, CA 90301, United States"
stanford = "625 Nelson Rd, Stanford, CA 94305, United States"
tampa = "4201 N Dale Mabry Hwy, Tampa, FL 33607, United States"
yankee = "1 E 161 St, The Bronx, NY 10451, United States"

# create list of stadiums with missing/incorrect addresses in game_data
missing_stadiums = "Allegiant Stadium, Alltel Stadium, Dolphin Stadium, Jack Murphy Stadium, Joe Robbie Stadium, Mercedes-Benz Stadium, Pro Player Stadium, Rose Bowl, SoFi Stadium, Stanford Stadium, Tampa Stadium, Yankee Stadium"
missing_stadiums = missing_stadiums.split(", ")

found_addresses = [allegiant_stadium, alltel_stadium, dolphin_stadium,
                       jack_murphy, joe_robbie, merc_benz, pro_player,
                       rose_bowl, sofi, stanford, tampa, yankee]

# dictionary of stadiums and corressponding confirmed address
address_mapper = dict(zip(missing_stadiums, found_addresses))

# add additional missing/incorrect addresses to dictionary
address_mapper['FedEx Field'] = "1600 Fedex Way, Landover, MD 20785, United States"
address_mapper['Fenway Park'] = "4 Jersey St, Boston, MA 02215, United States"
address_mapper['Legion Field'] = "400 Graymont Ave W, Birmingham, AL 35204, United States"
address_mapper['TIAA Bank Field'] = "1 TIAA Bank Field Dr, Jacksonville, FL 32202, United States"
address_mapper['Tottenham Hotspur Stadium'] = "782 High Rd, Tottenham, London N17 0BX, United Kingdom"
address_mapper['Tottenham Stadium'] = "782 High Rd, Tottenham, London N17 0BX, United Kingdom"

# fill in missing addresses and remove zip codes
game_data.address = game_data.address.fillna(game_data.stadium.map(address_mapper))
game_data['address'] = game_data['address'].apply(functions.remove_zipcode)

# fix vikings city 
vikings_home = (game_data['home_teamname'] == 'Vikings')
game_data.loc[vikings_home, 'home_city'] = 'Minneapolis'
vikings_away = (game_data['away_teamname'] == 'Vikings')
game_data.loc[vikings_away, 'away_city'] = 'Minneapolis'

#change to current name
giants_stadium = (game_data['stadium'] == 'Giants Stadium')
game_data.loc[giants_stadium, 'stadium'] = 'Metlife Stadium'

# change to current stadium name
dolphin_stadium = (game_data['stadium'] == 'Dolphin Stadium')
game_data.loc[dolphin_stadium, 'stadium'] = 'Hard Rock Stadium'

# fix remaining home/away_city
state_to_city = {'Arizona':'Phoenix', 'Carolina':'Charlotte', 
                 'New England': 'Foxborough', 'Washington': 'Ashburn',
                 'Washington Football': 'Ashburn' }

game_data.loc[game_data['home_city'].isin(state_to_city.keys()), 'home_city'] = game_data['home_city'].map(state_to_city)
game_data.loc[game_data['away_city'].isin(state_to_city.keys()), 'away_city'] = game_data['away_city'].map(state_to_city)

kaggle_game_ids = game_data.apply(lambda x: functions.create_game_id(x['schedule_date'],
                                             str(x['home_team_id']), str(x['away_team_id'])), axis = 1)
elo66_game_ids = elo66.apply(lambda x: functions.create_game_id(x['date'],
                                             str(x['team1']), str(x['team2'])), axis = 1)
game_data.insert(0, "game_id", kaggle_game_ids)
elo66.insert(0, "game_id", elo66_game_ids)

# change stadium names to current if only renamed, to avoid geocoding errors
game_data['stadium'] = game_data.apply(lambda x: "TIAA Bank Field" if x['stadium'] == "Alltel Stadium" else x['stadium'], axis = 1)
game_data['stadium'] = game_data.apply(lambda x: "TIAA Bank Field" if x['stadium'] == "EverBank Field" else x['stadium'], axis = 1)
game_data['address'] = game_data.apply(lambda x: "South Capitol Avenue, Indianapolis" if x['stadium'] == "RCA Dome" else x['address'], axis = 1)

print(f"data cleaning successful({datetime.now() - clean_time}s), time elapsed: {datetime.now() - start_time}s")
away_dist_time = datetime.now()
# setup geopy
geolocator = Nominatim(user_agent="NFL-eda")
geocoder = RateLimiter(geolocator.geocode, min_delay_seconds = 1)
point_cache = {}    # to minimize calls to Nominatim

game_data['dt_for_away'] = game_data.apply(lambda x: functions.get_distance_travelled(x['away_city']
                                                                            , x['address'],
                                                                            point_cache),
                                            axis = 1)

# recalculate distances flagged as errors
point_cache.clear()
game_data['dt_for_away'] = game_data.apply(lambda x: functions.get_distance_travelled(x['away_city'], x['stadium'], point_cache)
 if x['dt_for_away'] > 5000 else x['dt_for_away'], axis = 1)

# set dual team city games (i.e New York Giants vs New York Jets (both play out of Metlife Stadium)) to 1
_dual_city_mask = game_data['dt_for_away'] < 25
game_data.loc[_dual_city_mask, 'dt_for_away'] = 1

print(f"away team distance calculations successful({datetime.now() - away_dist_time}s), time elapsed: {datetime.now() - start_time}s")

#calculate home team distance
home_dist_time = datetime.now()
_mask = (game_data['stadium_neutral'] == True)
neutral_games = game_data[_mask]
# point_cache.clear()
game_data.loc[_mask,'dt_for_home'] =  neutral_games.apply(lambda x: 
                                                    functions.get_distance_travelled(
                                                        x['home_city'],
                                                        x['address'],
                                                        point_cache),
                                                    axis=1)

#check for obvious errors
print(game_data.dt_for_home.groupby(pd.cut(game_data['dt_for_home'],[-1, 50, 100, 1000, 3000,10000])).count())

#filter for obvious errors
incorrect_address_mask = (game_data['dt_for_home'] > 3000)
# point_cache.clear()
# run again using stadium name in place of address
game_data.loc[incorrect_address_mask, 'dt_for_home'] = game_data[incorrect_address_mask].apply(lambda x: 
                                                    functions.get_distance_travelled(
                                                        x['home_city'],
                                                        x['stadium'],
                                                        point_cache),
                                                    axis=1) 
#check for obvious errors
print(game_data.dt_for_home.groupby(pd.cut(game_data['dt_for_home'],[-1, 25, 100, 1000, 3000,10000])).count())

print(f"home team distance calculations successful ({datetime.now() - home_dist_time}s), time elapsed: {datetime.now() - start_time}s")

# calculate bearing of away team travel
compass_time = datetime.now()
_away_mask = (game_data['dt_for_away'] > 0)
game_data['bearing_away'] = 0
game_data.loc[_away_mask, 'bearing_away'] = game_data[_away_mask].apply(lambda x: functions.get_bearing(x['away_city'], x['address'], point_cache), axis = 1)

_address_mask = (game_data['bearing_away'] == 1000)
game_data.loc[_address_mask,'bearing_away'] = game_data[_address_mask].apply(lambda x: functions.get_bearing(x['away_city'], x['stadium'], point_cache), axis = 1)

# calculate bearing of home team travel
_home_mask = (game_data['dt_for_home'] > 0)
game_data['bearing_home'] = 0
game_data.loc[_home_mask, 'bearing_home'] = game_data[_home_mask].apply(lambda x: functions.get_bearing(x['home_city'], x['address'], point_cache), axis = 1)

_address_mask = (game_data['bearing_home'] == 1000)
game_data.loc[_address_mask,'bearing_home'] = game_data[_address_mask].apply(lambda x: functions.get_bearing(x['home_city'], x['stadium'], point_cache), axis = 1)

# get compass directions
game_data['compass_away'] = game_data['bearing_away'].apply(functions.get_compass_direction)
game_data['compass_home'] = game_data['bearing_home'].apply(functions.get_compass_direction)

print(f"compass direction calculations successful({datetime.now() - compass_time}s), time elapsed: {datetime.now() - start_time}s")

# prepare to merge dfs 
merge_time = datetime.now()
elo66.set_index('game_id', inplace=True)
game_data.set_index('game_id', inplace= True)

final_dataset = game_data.merge(elo66, how='inner', left_index=True, right_index=True)
final_dataset = final_dataset.drop(labels=['weather_temperature', 'weather_wind_mph','weather_humidity', 'weather_detail','date', 'season',
 'neutral', 'playoff','elo1_post', 'elo2_post','qb1_adj','qb2_adj','qb1_game_value', 'qb2_game_value', 'qb1_value_post','qb2_value_post', 
 'qbelo1_post', 'qbelo2_post','score1', 'score2'], axis = 1)

final_dataset['home_fav'] = [1 if x == y else 0 for (x,y) in zip(final_dataset['home_team_id'], final_dataset['team_favorite_id'])]

print(f'final merge successful({datetime.now() - merge_time}s), time elapsed: {datetime.now() - start_time}s')

print([item for item, count in collections.Counter(kaggle_game_ids).items() if count > 1])
print([item for item, count in collections.Counter(elo66_game_ids).items() if count > 1])
print(set(elo66['team1']).union(elo66['team2']))
print(len(game_data))
print(len(elo66))
print(game_data.head(1))
print(functions.dt_for_away_sanity_checker(game_data))
print(len(final_dataset))

final_dataset.to_excel("NFL_final_dataset.xlsx")
final_dataset.to_json("final_dataset.json")


print(f"total time elapsed: {datetime.now() - start_time}s")