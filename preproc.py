import pandas as pd
import numpy as np
import scipy.stats as stats
import http.client, urllib.parse
import pickle, traceback, json, time
from tqdm import tqdm
tqdm.pandas()

# dataset variables
user_id_col = 'user_id'
city_id_col = 'city_id'
city_name_col = 'city_name'
lat_col = 'latitude'
long_col = 'longitude'
time_col = 'utc_time'
poi_id_col = 'venue_id'
country_code_col = 'country_code'

def getColumnStats(df, col_name):
    valToInd = {val : i for i,val in enumerate(df[col_name].unique())} #giving an index for each unique value
    df2 = [0 for _ in range(len(valToInd))]
    for ind in df.index:
        df2[valToInd[df[col_name][ind]]] += 1 # updating frequency at the assigned index

    print('max',np.max(df2))
    print('min',np.min(df2))
    print('mean',np.mean(df2))
    print('median',np.median(df2))
    print('mode',stats.mode(df2))

def getStats(df_name):
    df = pd.read_csv(df_name, sep = '\t')
    df.drop('city_id', axis = 1)
    df.dropna(axis = 0, inplace = True)
    print(df.tail())
    print('No of users = ', len(df[user_id_col].unique()))
    print('No of cities = ', len(df[city_id_col].unique()))
    print('No of pois = ', len(df[poi_id_col].unique()))

    # take no of checkins by each user into an array and print stats
    print('Stats on no. of checkins by users')
    getColumnStats(df, user_id_col)

    # take no of checkins in each city into an array and print stats
    print('Stats on no. of checkins to citites')
    getColumnStats(df, city_id_col)

    # take no of checkins in each poi into an array and print stats
    print('Stats on no. of checkins to pois')
    getColumnStats(df, poi_id_col)



#handles call to api
def getCityApi(lat, long, country_name):

    # print(lat, long)
    conn = http.client.HTTPConnection('geocode.xyz')
    params = urllib.parse.urlencode({
        'locate': f'{lat},{long}',
        'region': f'{country_name}',
        'json': 1,
        })
    conn.request('GET', '/?{}'.format(params))
    res = conn.getresponse()
    data = res.read()
    return data


# pickles of dicts containing previously searched values
primary = None
others = None # to store info if it has errors in parsing to utf-8
try:
    f = open("./spickles/prim.pkl", "rb")
    primary = pickle.load(f)
    f.close()
except Exception:
    print('not found1')
    primary = dict()
print(len(primary))

try:
    f = open("./spickles/citylist_other.pkl", "rb")
    others = pickle.load(f)
    f.close()
except Exception:
    print('not found2')
    others = dict()
print(len(others))

err = 0

# function to apply on each row
def getCity(l):
    global err

    # if info is available already (or) if the it was fetched earlier no need to fetch again
    if not pd.isna(l[city_name_col]):
        return l[city_name_col]
    if (l[poi_id_col] in primary.keys()):
        return primary[l[poi_id_col]]
    if (l[poi_id_col] in others.keys()):
        return np.NaN

    try:
        data = None
        data = getCityApi(l[lat_col], l[long_col], l[country_code_col])
        time.sleep(.4) # max 3 calls/second
        data = (json.loads(data.decode('utf-8')))
            # if 'success' not in data.keys() or data['success'] == True:
            #     break
        if 'city' in data.keys():
            primary[l[poi_id_col]] = data['city']
            f = open("./spickles/prim.pkl", "wb")
            pickle.dump(primary, f)
            f.close()
            return data['city']
        else:
            others[l[poi_id_col]] = data
            f = open("./spickles/citylist_other.pkl", "wb")
            pickle.dump(others, f)
            f.close()
            return np.NaN

    except Exception as e:
        err += 1
        traceback.print_exc()
        print(e)
        if data is not None:
            others[l[poi_id_col]] = data
            f = open("./spickles/citylist_other.pkl", "wb")
            pickle.dump(others, f)
            f.close()
        return np.NaN


if __name__ == "__main__":
    df = pd.read_csv("./Datasets/dataset_TIST2015/smalldata.csv", sep = "\t")
    print(df.tail())
    print(df['city_name'].isna().sum())
    # getCityApi(d2[lat_col][3320], d2[long_col][3320], d2[country_code_col][3320])
    df[city_name_col] = df.progress_apply(getCity, axis=1)
    print(df['city_name'].isna().sum())
    df.to_csv("./Datasets/dataset_TIST2015/smalldata_part.csv")
    getStats("./Datasets/dataset_TIST2015/smalldata_part.csv")
