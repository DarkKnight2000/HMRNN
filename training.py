import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random
import pandas as pd
import numpy as np

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
input_size = 3
hidden_size = 5
batch_size = 1
seq_len = 1
num_layers = 2
epochs = 100

def getData(df):
	inputs, labels, lastCheckins = [], [], []
	for i,uid in enumerate(df[user_id_col].unique()):
		sub_df = df[df[user_id_col] == uid]
		if not i % 50: print('usr id', uid, ' dflen ', len(sub_df.index))
		# inp = lat, long, cid	out = lat, long, time, cid, binary
		inps = np.array((sub_df[lat_col].to_list()/90, sub_df[long_col].to_list()/180, sub_df[time_col].to_list(), sub_df[city_id_col].to_list())).T
		outs = np.array((sub_df[lat_col].to_list()/90, sub_df[long_col].to_list()/180, sub_df[time_col].to_list(), sub_df[city_id_col].to_list(), np.zeros(len(sub_df.index)))).T[1:]
		# print(inps[-1])
		# print(outs[-1])
		for i in range(len(inps)-1):
			# 2 for timestamp, 3 for city id, 4 for binary output
			outs[i][2] -= inps[i][2]
			outs[i][4] = 1 if outs[i][3] != inps[i][3] else 0
		# print(inps.shape)
		# print(inps[0])
		inps = np.concatenate((inps.T[:2], inps.T[3:])).T
		# print(inps.shape)
		# print(outs.shape)
		# break
		inputs.append(inps[:-1])
		labels.append(outs)
		lastCheckins.append(inps[-1])
	return inputs, labels, lastCheckins


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, batch_first = True, num_layers=num_layers)
        self.endSeg = 0

    def forward(self, x, hidden):
        x = x.view(batch_size, -1, input_size)
        # print(hidden.size())
        # Input : (batch, seqlen, input_size)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, hidden_size)
        return hidden, out

    def init_hidden(self):
        return Variable(torch.zeros(num_layers, batch_size, hidden_size))


# reading dataset
df = pd.read_csv('./Datasets/dataset_TIST2015/smalldata2.csv', sep = '\t')
print(df.dtypes)
print(len(df.index))
inputs, labels, lasts = getData(df)

print('cities ', len(df['city_name'].unique()))

# training settings
model = Model()
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.05)

# training loop
for epoch in range(epochs):
	optimiser.zero_grad()
	loss = 0
	for i in range(len(inputs)):
		if random.random() > .25 : continue # FIXME: Runs only for 25% of users in each loop
		inps, outs = inputs[i], labels[i]
		hidden = model.init_hidden()
		cellState = model.init_hidden()
		for inp, label in zip(inps, outs):
			# print(hidden)
			# print(torch.Tensor(inp).view(batch_size, -1, input_size))
			(hidden, cellState), output = model(torch.Tensor(inp), (hidden, cellState))
			val, idx = output.max(1)
			# print(output)
			# print(output.size())
			# print(label)
			# print(label.size())
			# try:
			loss += criterion(output, torch.tensor([label], dtype=torch.float))
			# except AttributeError:
			#     print(label)
	if not epoch % 1 : print('\nepoch : ', epoch, ' loss: ', loss)

	loss.backward()
	optimiser.step()