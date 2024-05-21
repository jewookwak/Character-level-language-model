
# -*- coding: utf-8 -*-
"""
First created on Mon May 17 22:40:38 2021

@author: Kiminjo 

Modified on Tue May 21 2024
feat.Kwakjewoo
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size=30, num_layer=1):
        super(CharRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch=100
        
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, bias=True, num_layers=self.num_layer, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

           
    def forward(self, input, hidden):

        x, hidden = self.rnn(input, hidden)
        x = x.reshape(x.size(0) * x.size(1), x.size(2))
        output = self.linear(x)
            
        return output, hidden
    
    
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layer, batch_size, self.hidden_size)
        
        return hidden_state


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer):
        super(CharLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch = 100
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bias=True, num_layers=self.num_layer, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.input_size)


    def forward(self, input, hidden):

        x, (hidden_state, cell_state) = self.lstm(input, hidden)
        x = x.reshape(x.size(0)*x.size(1), x.size(2))
        output = self.linear(x)
            
        return output, (hidden_state, cell_state)


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden_state = weight.new(self.num_layer, batch_size, self.hidden_size).zero_()
        cell_state = weight.new(self.num_layer, batch_size, self.hidden_size).zero_()
        
        return (hidden_state, cell_state)
