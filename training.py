import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random
import pandas as pd
import numpy as np

from models import SingleLSTMModel, MultiLSTMModel

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

f = open('logs.txt', 'w')

def getData(df):
    inputs, labels, lastCheckins = [], [], []
    for i,uid in enumerate(df[user_id_col].unique()):
        sub_df = df[df[user_id_col] == uid]
        if not i % 50: print('usr id', uid, ' dflen ', len(sub_df.index), file = f)
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


def getEncodedVec(input, size):
    # Last value of input should be city id
    # size is total no of cities
    retVal = torch.cat((input[:-1], torch.zeros(size)), 0)
    print('revalsize', retVal.size())
    print('cityid', input[-1])
    retVal[input.size()[0] - 1 + int(input[-1])] = 1
    return retVal


# reading dataset
df = pd.read_csv('./Datasets/dataset_TIST2015/smalldata_final.csv')
print(df.dtypes, file = f)
print(len(df.index), file = f)
inputs, labels, lasts = getData(df)

'''
1 -> single lstm model
2 -> multi lstm model
'''
model_name = 3

if model_name == 1:

    print('cities ', len(df['city_name'].unique()), file = f)
    city_vec_len = len(df['city_name'].unique())
    input_dim = 4
    # dim without city id
    output_dim = 4

    model = SingleLSTMModel(input_dim, output_dim + city_vec_len, city_vec_len, num_layers=num_layers, batch_size=batch_size)
    model.CustomTrain(df, inputs, labels, epochs)

elif model_name == 2:
    print('cities ', len(df['city_name'].unique()), file = f)
    city_vec_len = len(df['city_name'].unique())
    input_dim = 4
    output_dim = 4


    # creating model
    model = MultiLSTMModel(input_dim, output_dim, city_vec_len, num_layers=[num_layers, num_layers], batch_size=batch_size)
    model.CustomTrain(df, inputs, labels, epochs)