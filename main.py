# -*- coding: utf-8 -*-
"""
Created on Mon May 17 22:40:56 2021

@author: Kiminjo

Modified on Tue May 21 2024
Cuda is available now.
feat. KwakJewoo
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import Shakespeare, one_hot_encoding
from model import CharRNN, CharLSTM
from generate import generate
import warnings 
warnings.filterwarnings(action='ignore')

def train(model, trn_loader, device, criterion, optimizer, batch_size, network_type):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()
    
    total_batch = len(trn_loader)
    trn_loss = 0
    
    hidden = model.init_hidden(batch_size)
    if isinstance(hidden, tuple):
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = hidden.to(device)
    
    for batch_idx, batch in enumerate(trn_loader) :
        x, label = batch

        # input sequence x should be form of one hot vector 
        x = one_hot_encoding(x)
        x = x.to(device); label = label.to(device)
        if network_type == 'RNN' :
            hidden = tuple([each.data for each in hidden])[0].reshape(model.num_layer, batch_size, model.hidden_size).to(device)
        else :
            hidden = tuple([each.data for each in hidden])
            hidden = (hidden[0].to(device), hidden[1].to(device))

        optimizer.zero_grad()
        output, hidden = model.forward(x, hidden)
        cost = criterion(output, label.view(-1).long())
        cost.backward(retain_graph=True)
        optimizer.step()
        
        trn_loss += cost.item()
        
    trn_loss = round(trn_loss/total_batch, 3) 

    return trn_loss


@torch.no_grad()
def validate(model, val_loader, device, criterion, batch_size, network_type='RNN'):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()
    
    total_batch = len(val_loader)
    val_loss = 0
    
    hidden = model.init_hidden(batch_size)
    if isinstance(hidden, tuple):
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = hidden.to(device)
    
    for batch_idx, batch in enumerate(val_loader) :
        x, label = batch

        # input sequence x should be form of one hot vector 
        x = one_hot_encoding(x)
        x = x.to(device); label = label.to(device)
        if network_type == 'RNN' :
            hidden = tuple([each.data for each in hidden])[0].reshape(model.num_layer, batch_size, model.hidden_size).to(device)
        else :
            hidden = tuple([each.data for each in hidden])
            hidden = (hidden[0].to(device), hidden[1].to(device))

        output, hidden = model.forward(x, hidden)
        cost = criterion(output, label.view(-1).long())
        
        val_loss += cost.item()

    val_loss = round(val_loss/total_batch, 3) 

    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """
    
    input_file = open('data/shakespeare_train.txt', 'r').read()
    epochs = 10
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_dataset = Shakespeare(input_file, is_train=True)
    test_dataset = Shakespeare(input_file, is_train=False)
    
    train_data = DataLoader(train_dataset, batch_size=batch_size)
    test_data = DataLoader(test_dataset, batch_size=batch_size)
    
    ##################################################################
    #                   RNN model
    ##################################################################
    rnn_model = CharRNN(input_size=len(train_dataset.char2int), 
                    hidden_size=512, num_layer=1).to(device)
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.01)
    rnn_criterion = torch.nn.CrossEntropyLoss().to(device)
    
    rnn_trn_loss = []; rnn_val_loss = []
    
    print('RNN training start')
    for epoch in range(epochs) :
        train_loss = train(rnn_model, train_data, device, rnn_criterion, rnn_optimizer, batch_size, 'RNN')
        test_loss = validate(rnn_model, test_data, device, rnn_criterion, batch_size, 'RNN')
        print('epoch : {}, train loss : {}, validation loss : {}'.format(epoch+1, train_loss, test_loss))
        
        rnn_trn_loss.append(train_loss); rnn_val_loss.append(test_loss)
    for temperature in [1,2,3,4,5]:      
        rnn_generated_text = generate(rnn_model, 'The', temperature, 'RNN', train_dataset.char2int, train_dataset.int2char)
        rnn_text = open('rnn_T'+str(temperature)+'.txt', 'w')
        rnn_text.write(rnn_generated_text)
        rnn_text.close()
    
    
    
    ##################################################################
    #                   LSTM model
    ##################################################################
    lstm_model = CharLSTM(input_size=len(train_dataset.char2int), 
                    hidden_size=512, num_layer=1).to(device)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
    lstm_criterion = torch.nn.CrossEntropyLoss().to(device)
    
    lstm_trn_loss = []; lstm_val_loss = []
    print('\n LSTM training start')
    
    for epoch in range(epochs) :
        train_loss = train(lstm_model, train_data, device, lstm_criterion, lstm_optimizer, batch_size, 'LSTM')
        test_loss = validate(lstm_model, test_data, device, lstm_criterion, batch_size, 'LSTM')
        print('epoch : {}, train loss : {}, validation loss : {}'.format(epoch+1, train_loss, test_loss))
        
        lstm_trn_loss.append(train_loss); lstm_val_loss.append(test_loss)
        
    for temperature in [1,2,3,4,5]:      
        lstm_generated_text = generate(lstm_model, 'The', temperature, 'LSTM', train_dataset.char2int, train_dataset.int2char)
        lstm_text = open('lstm_T'+str(temperature)+'.txt', 'w')
        lstm_text.write(lstm_generated_text)
        lstm_text.close()
        
    draw_result_plot(rnn_trn_loss, rnn_val_loss, lstm_trn_loss, lstm_val_loss)
    


def draw_result_plot(rnn_trn, rnn_val, lstm_trn, lstm_val) :
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
    rnn_epoch = list(range(len(rnn_trn)))
    lstm_epoch = list(range(len(lstm_trn)))
    
    axes[0, 0].plot(rnn_epoch, rnn_trn)
    axes[0, 0].set_title('RNN model train loss')
    
    axes[0, 1].plot(rnn_epoch, rnn_val)
    axes[0, 1].set_title('RNN model validation loss')
    
    axes[1, 0].plot(lstm_epoch, lstm_trn)
    axes[1, 0].set_title('LSTM model train loss')
    
    axes[1, 1].plot(lstm_epoch, lstm_val)
    axes[1, 1].set_title('LSTM model validation loss')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('result.png')
    
    
if __name__ == '__main__':
     print('cuda is: ',torch.cuda.is_available())
     main()
