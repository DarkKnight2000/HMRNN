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
    df = pd.read_csv(df_name)
    # df.drop('city_id', axis = 1)
    print('No of unknown city entries = ', df['city_name'].isna().sum())
    df.dropna(axis = 0, inplace = True)
    print(df.tail())
    print('No of users = ', len(df[user_id_col].unique()))
    print('No of cities = ', len(df[city_name_col].unique()))
    print('No of pois = ', len(df[poi_id_col].unique()))

    # take no of checkins by each user into an array and print stats
    print('Stats on no. of checkins by users')
    getColumnStats(df, user_id_col)

    # take no of checkins in each city into an array and print stats
    print('Stats on no. of checkins to citites')
    getColumnStats(df, city_name_col)

    # take no of checkins in each poi into an array and print stats
    print('Stats on no. of checkins to pois')
    getColumnStats(df, poi_id_col)

    #finding no of switches
    valToInd = {val : i for i,val in enumerate(df[user_id_col].unique())} #giving an index for each unique value
    lastVisits = [0 for _ in range(len(valToInd))]
    switches = [0 for _ in range(len(valToInd))]
    for ind in df.index:
        if(lastVisits[valToInd[df[user_id_col][ind]]] != df[city_name_col][ind]):
            switches[valToInd[df[user_id_col][ind]]] += 1 # updating frequency at the assigned index
        lastVisits[valToInd[df[user_id_col][ind]]] = df[city_name_col][ind] # changing lastvisited city entry
    print('Stats on no of switches : ')
    print('max',np.max(switches))
    print('min',np.min(switches))
    print('mean',np.mean(switches))
    print('median',np.median(switches))
    print('mode',stats.mode(switches))


def addIdColumn(df, col_name, result_col_name):
    dmap = {k:i for i,k in enumerate(df[col_name].unique())}
    df[result_col_name] = df[col_name].map(dmap)
    return df



api_key = ''

api_keys = ['715579682156860741540x5503', '564332717213180365095x5504', '564332717213180365095x5504', '163136140253566128886x5505', '163136140253566128886x5505', '7279258122286242974x5506', '7279258122286242974x5506', '365592742643973351323x5507', '365592742643973351323x5507', '303063150889330221344x5508', '303063150889330221344x5508', '118318145272064304466x5509', '118318145272064304466x5509', '896077810645614897041x5510', '896077810645614897041x5510', '394445711086250263653x5511', '394445711086250263653x5511']

#handles call to api
def getCityApi(lat, long, country_name):

    # print(lat, long)
    conn = http.client.HTTPConnection('geocode.xyz')
    params = urllib.parse.urlencode({
        'locate': f'{lat},{long}',
        'region': f'{country_name}',
        'json': 1,
        'auth' : api_key
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
isLimit = False

# function to apply on each row
def getCity(l):
    global err, isLimit

    # if info is available already (or) if the it was fetched earlier no need to fetch again
    if not pd.isna(l[city_name_col]):
        return l[city_name_col]
    if (l[poi_id_col] in primary.keys()):
        return primary[l[poi_id_col]]
    if (l[poi_id_col] in others.keys()):
        try:
            if 'town' in others[l[poi_id_col]].keys():
                primary[l[poi_id_col]] = others[l[poi_id_col]]['town']
                del others[l[poi_id_col]]
                return primary[l[poi_id_col]]
        except:
            return np.NaN
        return np.NaN

    err += 1
    if(isLimit): return np.NaN
    try:
        data = None
        data = getCityApi(l[lat_col], l[long_col], l[country_code_col])
        # data = Geocoder.reverse_geocode(df['latitude'][0], df['longitude'][0])
        time.sleep(0) # max 3 calls/second
        data = (json.loads(data.decode('utf-8')))
            # if 'success' not in data.keys() or data['success'] == True:
            #     break
        if 'city' in data.keys():
            primary[l[poi_id_col]] = data['city']
            f = open("./spickles/prim.pkl", "wb")
            pickle.dump(primary, f)
            f.close()
            #print('got knew result')
            return data['city']
        elif 'town' in data.keys():
            primary[l[poi_id_col]] = data['town']
            f = open("./spickles/prim.pkl", "wb")
            pickle.dump(primary, f)
            f.close()
            #print('got knew result')
            return data['town']
        elif (('success' in data.keys()) and (not data['success'])) or ('error' in data.keys()):
            isLimit = True
            print(data)
            return np.NaN
        else:
            others[l[poi_id_col]] = data
            f = open("./spickles/citylist_other.pkl", "wb")
            pickle.dump(others, f)
            f.close()
            print('keyerror')
            print(data)
            return np.NaN
        # else:
        #     time.sleep(3)
        #     return np.NaN

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
    df = pd.read_csv("./Datasets/dataset_TIST2015/smalldata_final.csv")
    print(df.tail())
    print(df['city_name'].isna().sum())

    # applying map from last
    df = df.reindex(index=df.index[::-1])
    for i in range(len(api_keys)):
        api_key = api_keys[i]
        isLimit = False
        err = 0
        df[city_name_col] = df.progress_apply(getCity, axis=1)
        if err == 0: break
        isLimit = False
        err = 0
    print(df['city_name'].isna().sum()) # checking no of nan left
    addIdColumn(df, city_name_col, city_id_col) # adding city id column
    df = df.reindex(index=df.index[::-1]) # reversing data frame

    # changing column type
    # print(df.dtypes)
    # df = df.astype({user_id_col:'int32', 'category_id':'int64', 'country_code_id':'int64'})


    # saving to file
    df.to_csv("./Datasets/dataset_TIST2015/smalldata_final.csv", index=False)
    print(df.head())
    print(df.dtypes)
    print('err', err)





'''

smalldata_unproc.csv - before joining with poi
smalldata_final.csv - with city name and city id from two apis
smalldata.csv - city names only from geopy and with category column
smalldata2.csv - city names only from geopy and without category column

'''