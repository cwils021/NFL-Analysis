import re
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy import distance
from geographiclib.geodesic import Geodesic

# setup geopy
geolocator = Nominatim(user_agent="NFL-eda")
geocoder = RateLimiter(geolocator.geocode, min_delay_seconds = 1)
point_cache = {}


def remove_zipcode(address):
  zip_re = "[0-9]{5}"

  return re.sub(zip_re, "", address)

# create game_id to be able to join with elo data
def create_game_id(date, home_team_id, away_team_id):
  date_str = date.strftime('%Y%m%d')
  game_id = str(date_str + home_team_id + away_team_id)
  return game_id

def get_point(address, point_cache):
  if address in point_cache.keys():
    return point_cache.get(address)
  else:
    geo = geocoder(address)
    if geo:
      point = tuple([geo.point[0], geo.point[1]])
    else:
      point = tuple()
    
    point_cache[address] = point
    return point

def get_distance_travelled(start, dest, point_cache):  
  # get points using get_point
  start_point = get_point(start, point_cache)
  dest_point = get_point(dest, point_cache)
  
  # if both points valid calculate distance else return -1
  if all(start_point) & all(dest_point):
    dist = distance.distance(start_point, dest_point).miles
    return dist
  else:
    return -1

def dt_for_away_sanity_checker(game_data):
  i = np.random.randint(0, len(game_data))
  print(game_data.iloc[i,:])
  return 1

def get_bearing(start, dest, point_cache):  
  # get points using get_point
  start_point = get_point(start, point_cache)
  dest_point = get_point(dest, point_cache)
  
  # if both points valid calculate distance else return -1
  try:
    azimuth = Geodesic.WGS84.Inverse(start_point[0], start_point[1],
                                     dest_point[0], dest_point[1], outmask = 512)['azi1']
    bearing = azimuth
    if bearing < 0: bearing += 360
    
    return bearing
  except IndexError:
    return 1000  

def get_compass_direction(bearing):
  if bearing > 0.0 and bearing < 90.0:
    return "NE"
  elif bearing >= 90.0 and bearing < 180.0:
    return "SE"
  elif bearing >= 180.0 and bearing < 270.0:
    return "SW"
  elif bearing >= 270.0 and bearing < 360.0:
    return "NW"
  else:
    return "N/A"