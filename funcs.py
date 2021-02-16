import requests, json
import pandas as pd
import numpy as np
import pickle, time, os
from sklearn.model_selection import GridSearchCV

#load needed values for prediction
df_means = pd.read_csv('means_of_cities.csv', index_col=0)
cities_dict = pd.read_csv('cities_dict.csv', index_col=0)
inverse_cities_dict = cities_dict.copy()
inverse_cities_dict.index = cities_dict['city_num']
inverse_cities_dict['city_num'] = cities_dict.index
df_beaches = pd.read_csv('beachs_coordinates.csv', index_col = 0)
df_hospitals = pd.read_csv('Hospitals_coordinates.csv', index_col = 0)
df_downts = pd.read_csv('cities_with_coordinates.csv', index_col=0)
df_downts = df_downts.loc[cities_dict.index]
df_beaches = pd.DataFrame([df_beaches['latitude'].values,df_beaches['longtitude'].values],index=None)
df_hospitals = pd.DataFrame([df_hospitals['latitude'].values,df_hospitals['longtitude'].values],index=None)
#load trained model
regr = pickle.load(open('finalized_model.sav', 'rb'))

def check_value(num):
### checking if input is valid ###
    try:
        num = float(num)
        return True
    except ValueError as ve:
        if num == '':
            return True
        return False


def handle_missing(num, city, col):
### handling missing inputs ###
    global df_means
    if num == '':
        return df_means.loc[city, col]
    else:
        return float(num)

def get_distance(coor1,coor2):
### given two coordinates, returns the distance ###
    dis = ((coor1[0]-coor2[0])**2)+((coor1[1]-coor2[1])**2)
    dis = np.sqrt(dis)
    return dis

def predict(house):
    ###
    # 1. find closest beach, hospit, and downtown
    # 2. assign date to todays TARES index value
    # 3. predict with tarined model
    ###

    global cities_dict, df_beaches, df_hospitals, df_downts, inverse_cities_dict
    full_address = house['city'] + ' ' + house['address']
    coors = get_coordinates(full_address)
	
	# 1. find closest beach, hospit, and downtown
    closest_beach = np.min(get_distance(coors,df_beaches.values))
    closest_hospi = np.min(get_distance(coors, df_hospitals.values))
    dist_list = list()
    for coors2 in df_downts.values:
        dist_list.append(get_distance(coors,coors2))
    # find colsest city
    closest_downtown = np.min(dist_list)
    city_num = np.argmin(dist_list)
	
	# 2. assign date to todays TARES index value
    index_val = 850 #approximatly accordinf to todays date
    house['city'] = inverse_cities_dict.iloc[city_num].values[0]
    city = house['city']
    house2 = {'beds': handle_missing(house['beds'],city,'beds'), 'area': handle_missing(house['area'],city,'size'),
              'yearBuilt': handle_missing(house['yearBuilt'],city,'yearBuilt'),
              'floor': handle_missing(house['floor'],city,'floor'), 'city': city_num, 'index_value': index_val,
              'beach_dist': closest_beach, 'hospi_dist': closest_hospi, 'down_town_dist': closest_downtown}
    house2 = pd.DataFrame.from_dict(house2,orient='index')
	
	# 3. predict with tarined model
    return regr.predict(house2[0].values.reshape(1,-1))

def get_coordinates(full_address):
	### given address, returns coordinates of it ###
    lat = 0
    lon = 0
    url = ['https://nominatim.openstreetmap.org/search?q=','&format=json']
    url = url[0] + full_address + url[1]
    req = requests.get(url)
    if len(req.text)>=10:
        req = json.loads(req.text)
        req = req[0]
        lon = float(req['lon'])
        lat = float(req['lat'])
    return [lat,lon]


def check_input(house_dict):
	### check if all inputs are valid ###
    if house_dict['city'] == '':
        return False
    else:
        valid_inputs = 0
        for key in house_dict.keys():
            if house_dict[key] != '':
                valid_inputs+=1
                if valid_inputs >= 3:
                    return True
        return False


def calculate_price(house_dict):
	### given house properties, returns its price prediction ###
    status = True
    price = 0
    keys = list(house_dict.keys())
    for key in keys[2:]:
        status = check_value(house_dict[key])
        if not status:
            return status, str(price)
    price = predict(house_dict)
    price = np.round(price/100000)/10
    return status, str(price[0] + 1)

