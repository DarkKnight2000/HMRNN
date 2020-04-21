import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from collections import defaultdict


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

    def forward(self, x, hidden):
        x = x.view(self.batch_size, -1, self.input_size)
        # print(hidden.size())
        # Input : (batch, seqlen, input_size)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, self.hidden_size)
        return hidden, out

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)), Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))


    def CustomLoss(self, outputs, labels):
        output_dim = self.hidden_size - self.city_size
        cur_loss = self.mseloss(outputs[0][:output_dim], torch.tensor(labels[:-1], dtype=torch.float))
        return torch.add(cur_loss ,self.crossEntLoss(outputs[:, output_dim:], torch.tensor([labels[-1]], dtype=torch.long)))

    def CustomTrain(self, df, inputs, labels, epochs):

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
                    hidden, output = self(torch.Tensor(inp), hidden)
                    # print(output)
                    # print(output.size())
                    # print('labe',label)
                    # print(label.size())
                    # try:
                    loss += self.CustomLoss(output, label)
                    # except AttributeError:
                    #     print(label)
            if not epoch % 1 : print('\nepoch : ', epoch, ' loss: ', loss)

            loss.backward()
            optimiser.step()




class MultiLSTMModel(nn.Module):
    '''
    For poi layer
    input -> lat, long, timediff, cityid
    output -> lat, long, timediff, binary switch(indicating if visits in current city are over)
    For city layer
    input -> curr city id
    output vector -> next city id one hot encoded
    '''

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
        for epoch in range(epochs):
            optimiser.zero_grad()
            loss = 0
            for i in range(len(inputs)):
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