import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random
import pandas as pd
import numpy as np
import pickle

from models import SingleLSTMModel, MultiLSTMModel, MultiLstmTrain

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)

# dataset variables
user_id_col = 'user_id'
city_id_col = 'city_id'
poi_id_col = 'venue_id'
lat_col = 'latitude'
long_col = 'longitude'
time_col = 'utc_time'

# training vars
batch_size = 1
seq_len = 1
num_layers = 2
epochs = 100


def getData(df):
    inputs, labels, lastCheckins = [], [], []
    for i,uid in enumerate(df[user_id_col].unique()):
        sub_df = df[df[user_id_col] == uid]
        if not i % 50: print('usr id', uid, ' dflen ', len(sub_df.index))
        # inp = lat, long, time, cid
        # out = lat, long, time, binary, cid
        inps = np.array((sub_df[lat_col].to_list(), sub_df[long_col].to_list(), sub_df[time_col].to_list(), sub_df[city_id_col].to_list())).T/np.array([90,180,1,1])# normalising
        outs = np.array((sub_df[lat_col].to_list(), sub_df[long_col].to_list(), sub_df[time_col].to_list(), np.zeros(len(sub_df.index)), sub_df[city_id_col].to_list())).T[1:]/np.array([90,180,1,1,1])
        # print(inps[0])
        # print(np.array((sub_df[lat_col].to_list(), sub_df[long_col].to_list(), sub_df[time_col].to_list(), sub_df[city_id_col].to_list())).T[0])
        for i in range(len(inps)-1):
            # 2 for timestamp, 3 for binary output, 4 for city id
            outs[i][2] -= inps[i][2]
            if (i != len(inps) - 1):
                inps[i+1][2] -= inps[i][2] # giving time difference as input
            outs[i][3] = 1 if outs[i][4] != inps[i][3] else 0

        inputs.append(inps[:-1])
        labels.append(outs)
        lastCheckins.append(inps[-1])
    return inputs, labels, lastCheckins


def getSessionData(df):

    # assigning venue id with respect to each city
    city_count = df[city_id_col].max()
    df2 = [set() for _ in range(city_count+1)]
    max_venues = 0
    for ind in df.index:
        df2[df[city_id_col][ind]].add(df[poi_id_col][ind])
        max_venues = max_venues if max_venues > len(df2[df[city_id_col][ind]]) else len(df2[df[city_id_col][ind]])
    venue_id_map = dict()
    for i in range(len(df2)):
        df2[i] = list(df2[i])
        for j in range(len(df2[i])):
            venue_id_map[df2[i][j]] = j # map from venue_id to its index in list of all venue_ids in current city
    print('done venueids')
    data = []
    last_user_ind = -1
    for ind in df.index:
        if df[user_id_col][ind] != last_user_ind:
            data.append([[df[city_id_col][ind], []]])
            last_user_ind = df[user_id_col][ind]
        if data[-1][-1][0] == df[city_id_col][ind]:
            data[-1][-1][1].append([venue_id_map[df[poi_id_col][ind]], df[time_col][ind]])
        else:
            data[-1].append([df[city_id_col][ind], []])
            data[-1][-1][1].append([venue_id_map[df[poi_id_col][ind]], df[time_col][ind]])

    print('onecheckinentry', data[0][0])

    # data[user_id] = [[city_id, list of checkins info in that city]]
    return data, max_venues




# def getEncodedVec(input, size):
#     # Last value of input should be city id
#     # size is total no of cities
#     retVal = torch.cat((input[:-1], torch.zeros(size)), 0)
#     print('revalsize', retVal.size())
#     print('cityid', input[-1])
#     retVal[input.size()[0] - 1 + int(input[-1])] = 1
#     return retVal


# Writing curr time in log file
from datetime import datetime
f = open('logs.txt', 'a')
f.write('Time stamp : ' + str(datetime.now()) + '\n')

load_data = True
df = pd.read_csv('./Datasets/dataset_TIST2015/smalldata_final.csv')

if not load_data:
    # reading dataset
    # print(df.dtypes, file = f)
    print('Total checkins : ', len(df.index), file = f)
    # inputs, labels, lasts = getData(df)
    data = getSessionData(df)
    f = open("./DataPickles/session_data.pkl", "wb")
    pickle.dump(data, f)
    f.close()
else:
    f = open("./DataPickles/session_data.pkl", "rb")
    data = pickle.load(f)
    f.close()

'''
1 -> single lstm model
2 -> multi lstm model
'''
model_name = 2

if model_name == 1:

    print('cities ', len(df['city_name'].unique()))
    city_vec_len = len(df['city_name'].unique())
    input_dim = 4
    # dim without city id
    output_dim = 4
    inputs, labels, _ = getData(df)

    model = SingleLSTMModel(input_dim, output_dim + city_vec_len, city_vec_len, num_layers=num_layers, batch_size=batch_size)
    model.CustomTrain(df, inputs, labels, epochs)

elif model_name == 2:
    print('cities ', len(df['city_name'].unique()))
    city_vec_len = len(df['city_name'].unique())


    # creating model
    model = MultiLSTMModel(city_vec_len, data[1])
    MultiLstmTrain(model, data[0], 100)


f.close()
