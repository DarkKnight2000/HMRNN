import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from collections import defaultdict


use_cuda = False

class SingleLSTMModel(nn.Module):
    '''
    input -> lat, long, timediff, cityid
    output -> lat, long, timediff, binary switch(idicationg if visits in current city are over), one hot encoded vector of nextcityid
    '''

    def __init__(self, inp_size, hid_size, city_size, num_layers = 3, batch_size = 1):
        super(SingleLSTMModel, self).__init__()
        self.rnn = nn.LSTM(input_size = inp_size, hidden_size = hid_size, batch_first = True, num_layers=num_layers)
        self.endSeg = 0
        self.input_size = inp_size
        self.hidden_size = hid_size
        self.city_size = city_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.mseloss = nn.MSELoss()
        self.crossEntLoss = nn.CrossEntropyLoss()

        if use_cuda : self.cuda_dev = torch.device('cuda')     # Default CUDA device

    def forward(self, x, hidden):
        x = x.view(self.batch_size, -1, self.input_size)
        # print(hidden.size())
        # Input : (batch, seqlen, input_size)
        self.rnn.flatten_parameters()
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, self.hidden_size)
        # if use_cuda: del h2
        return hidden, out

    def init_hidden(self):
        if use_cuda:
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda(device=self.cuda_dev)), Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda(device=self.cuda_dev)))
        else:
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)), Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))


    def CustomLoss(self, outputs, labels):
        output_dim = self.hidden_size - self.city_size
        cur_loss = self.mseloss(outputs[0][:output_dim], torch.tensor(labels[:-1], dtype=torch.float).cuda(device=self.cuda_dev))
        return torch.add(cur_loss ,self.crossEntLoss(outputs[:, output_dim:], torch.tensor([labels[-1]], dtype=torch.long).cuda(device=self.cuda_dev)))

    def CustomTrain(self, df, inputs, labels, epochs):

        f = open('logs.txt', 'a')

        # if use_cuda: self.cuda(device=self.cuda_dev)

        optimiser = torch.optim.Adam(self.parameters(), lr = 0.05)

        # training loop
        for epoch in range(epochs):
            optimiser.zero_grad()
            loss = 0
            for i in range(len(inputs)):
                inps, outs = inputs[i], labels[i] # for each user
                hidden = self.init_hidden()
                for inp, label in zip(inps, outs):
                    # print(hidden)
                    # print(torch.Tensor(inp).view(batch_size, -1, input_size))
                    if use_cuda:
                        inp2 = torch.Tensor(inp).cuda(device=self.cuda_dev)
                        label2 = torch.Tensor(label).cuda(device=self.cuda_dev)
                    else:
                        inp2 = torch.Tensor(inp)
                    hidden, output = self(inp2, hidden)
                    # print(output)
                    # print(output.size())
                    # print('labe',label)
                    # print(label.size())
                    # try:
                    del inp2
                    loss += self.CustomLoss(output, label2)
                    # except AttributeError:
                    #     print(label)
            if not epoch % 1 : print('\nepoch : ', epoch, ' loss: ', loss, file=f)

            loss.backward()
            optimiser.step()

        f.close()



'''
class MultiLSTMModel(nn.Module):

    # For poi layer
    # input -> lat, long, timediff, cityid
    # output -> lat, long, timediff, binary switch(indicating if visits in current city are over)
    # For city layer
    # input -> curr city id
    # output vector -> next city id one hot encoded

    def __init__(self, inp_size, hid_size, city_size, num_layers = [3,3], batch_size = 1):
        super(MultiLSTMModel, self).__init__()
        self.poi = nn.LSTM(input_size = inp_size - 1, hidden_size = hid_size, batch_first = True, num_layers=num_layers[0])
        self.city = nn.LSTM(input_size = 1, hidden_size = city_size, batch_first = True, num_layers=num_layers[1])
        self.endSeg = 0
        self.input_size = inp_size
        self.hidden_size = hid_size
        self.city_size = city_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.mseloss = nn.MSELoss()
        self.crossEntLoss = nn.CrossEntropyLoss()

    def forward(self, x, hidden):
        x = x.view(self.batch_size, -1, self.input_size)
        x1 = torch.split(x, self.input_size-1, dim=2)

        # print('split tensor ', x1)
        # Input : (batch, seqlen, input_size)
        out = [None, None]
        out[0], hidden[0] = self.poi(x1[0], hidden[0])
        # print('split tensor ', x1[1].size())
        # print('hid tensor ', hidden[1][1].size())
        out[1], hidden[1] = self.city(x1[1], hidden[1])
        out[1] = out[1].view(-1, self.city_size)
        return hidden, out

    def init_hidden_poi(self):
        return (Variable(torch.zeros(self.num_layers[0], self.batch_size, self.hidden_size)), Variable(torch.zeros(self.num_layers[0], self.batch_size, self.hidden_size)))

    def init_hidden_city(self):
        return (Variable(torch.zeros(self.num_layers[1], self.batch_size, self.city_size)), Variable(torch.zeros(self.num_layers[1], self.batch_size, self.city_size)))


    def CustomLoss(self, outputs, labels):
        cur_loss = self.mseloss(outputs[0], torch.tensor(labels[:-1], dtype=torch.float).view((self.batch_size, -1, self.input_size)))
        if labels[3] == 1:
            return torch.add(cur_loss ,self.crossEntLoss(outputs[1], torch.tensor([labels[-1]], dtype=torch.long)))
        else:
            return cur_loss

    def CustomTrain(self, df, inputs, labels, epochs):

        optimiser = torch.optim.Adam(self.parameters(), lr = 0.05)

        # to store hidden state of each city
        city_state = defaultdict(lambda : self.init_hidden_poi())

        f = open('logs.txt', 'a')

        # training loop
        print("length of inputs " + str(len(inputs)))
        for epoch in range(epochs):
            print("epoch started " + str(epoch))
            optimiser.zero_grad()
            loss = 0
            for i in range(len(inputs)):
                print("i done " + str(i))
                inps, outs = inputs[i], labels[i] # for each user
                hidden = [self.init_hidden_poi(), self.init_hidden_city()]
                city_state[inputs[i][0][3]] = hidden[0]
                for inp, label in zip(inps, outs):
                    # print(hidden)
                    # print(torch.Tensor(inp).view(batch_size, -1, input_size))
                    hidden, output = self(torch.Tensor(inp), hidden)


                    # if checkins in current city have ended, update state into dict and get next city state
                    if label[3] == 1:
                        city_state[inp[3]] = hidden[0]
                        hidden[0] = city_state[label[4]]

                    # print(output)
                    # print(output.size())
                    # print('labe',label)
                    # print(label.size())
                    # try:
                    loss += self.CustomLoss(output, label)
                    # except AttributeError:
                    #     print(label)
            if not epoch % 1 : print('\nepoch : ', epoch, ' loss: ', loss, file = f)

            loss.backward()
            optimiser.step()
'''



class MultiLSTMModel(nn.Module):

    def __init__(self, num_cities, poi_vec_len, city_rep_len = 128):
        super(MultiLSTMModel, self).__init__()
        self.num_cities = num_cities
        self.poi_vec_len = poi_vec_len
        self.city_rep_len = city_rep_len
        self.inner_hidden_size = 1 + poi_vec_len
        self.innerLstm = None
        self.innerLstmDict = defaultdict(lambda : nn.LSTM(input_size = city_rep_len, hidden_size = self.inner_hidden_size, batch_first = True, num_layers = 1))
        self.innerHidDict = defaultdict(lambda : (Variable(torch.zeros(1, 1, self.inner_hidden_size)), Variable(torch.zeros(1, 1, self.inner_hidden_size))))
        self.outerLstm = nn.LSTM(input_size = num_cities, hidden_size = city_rep_len, batch_first = True, num_layers = 1)

        self.mseloss = nn.MSELoss()
        self.crossEntLoss = nn.CrossEntropyLoss()

    def changeCity(self, city_id):
        self.innerLstm = self.innerLstmDict[city_id]

    def init_hidden_outer(self):
        return (Variable(torch.zeros(1, 1, self.city_rep_len)), Variable(torch.zeros(1, 1, self.city_rep_len)))

    def checkInLoss(self, output, label):
        output = torch.split(output, self.poi_vec_len, dim=2)
        return torch.add(self.mseloss(output[1], torch.tensor([[[label[1]]]], dtype=torch.float)) ,self.crossEntLoss(output[0].view(-1, self.poi_vec_len), torch.tensor([label[0]], dtype=torch.long)))

def getEncodedVec(vec_len, on_at):
    ret = torch.zeros(vec_len, dtype = torch.float)
    print('ret', ret, 'onat ', on_at)
    ret[on_at] = 1
    if use_cuda : ret = ret.cuda(torch.device('cuda'))
    return ret

def MultiLstmTrain(model:MultiLSTMModel, data, epochs):

    if use_cuda : model = model.cuda(torch.device('cuda'))

    # optimiser_outer = torch.optim.Adam(model.outerLstm.parameters(), lr = 0.05)

    # to store no of visits by a user in each city
    user_city_visits = defaultdict(lambda : defaultdict(lambda : 0))

    f = open('logs.txt', 'a')

    # training loop
    for epoch in range(epochs):
        print("epoch started " + str(epoch))
        total_loss = 0
        # for each user
        print(data[0][0])
        for u_visit in data:
            # optimiser_outer.zero_grad()
            # outer_loss = 0
            outer_hidden = model.init_hidden_outer()
            # for each city a user visited
            for c_visit in u_visit:
                # print('cvisit1',c_visit[1])
                curr_city = c_visit[0]
                # print('curcity', curr_city)
                inp_inner, outer_hidden = model.outerLstm(getEncodedVec(model.num_cities, curr_city).view(1, 1, -1), outer_hidden)
                

                model.changeCity(curr_city)
                hidden_inner = model.innerHidDict[curr_city]
                innerLoss = 0
                optimiser_inner = torch.optim.Adam(model.innerLstm.parameters(), lr = 0.05)
                optimiser_inner.zero_grad()
                for cin in c_visit[1]:
                    pred_checkin, hidden_inner = model.innerLstm(inp_inner, hidden_inner)
                    innerLoss += model.checkInLoss(pred_checkin, cin)

                total_loss += innerLoss
                innerLoss.backward(retain_graph=True)
                optimiser_inner.step()
                del optimiser_inner
                model.innerHidDict[curr_city] = hidden_inner

            # outer_loss.backward()
            # optimiser_outer.step()

            # deleting hidden states of every city's lstm and starting newly for each user
            for k in model.innerHidDict.keys():
                del model.innerHidDict[k]

        if not epoch % 1 : print('\nepoch : ', epoch, ' loss: ', total_loss, file = f)
