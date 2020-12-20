import csv
import numpy as np
import pandas as pd
import time 
import torch
import tqdm
from torch.utils.data import DataLoader

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class NeuralRegressorNet(torch.nn.Module):
    def __init__(self, n_input):
        super(NeuralRegressorNet, self).__init__()
        self.layer1 = torch.nn.Linear(n_input,10)
        self.layer2 = torch.nn.Linear(10, 1)
    
    def forward(self,x):
        out = self.layer1(x)
        out = torch.sigmoid(out)
        out = self.layer2(out)
        return torch.relu(out)

class NeuralRegressor:
    def __init__(self, n_input, batch_size, n_epochs):
        self.net = NeuralRegressorNet(n_input)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.batch_size = batch_size
        self.n_epochs = n_epochs



    def train_one_epoch(self, dataloader, lossfunc= torch.nn.L1Loss()):
        loss_epoch =0
        for x,y in dataloader:
            self.optimizer.zero_grad()
            scores = self.net(x)
            loss = lossfunc(scores,y)
            loss_epoch+=loss.item()
            loss.backward()
            self.optimizer.step()
        return loss_epoch/len(dataloader)
    
    def test(self, dataloader):
        # dataloader = DataLoader(testset, batch_size = 1)
        loss=0
        for x,y in dataloader:
            score = self.net(x).detach()
            loss+= abs(y-score)
        return loss/len(dataloader)
    
if __name__ == "__main__":
    X = pd.read_csv("X_train.csv")
    y =  pd.read_csv("y_train.csv")

    split_ratio = .85

    X_np = X.to_numpy(dtype = np.float32)
    y_np = y.to_numpy(dtype = np.float32)

    n_tot = X_np.shape[0]
    n_train = int(split_ratio*n_tot)

    X_train_np = X_np[:n_train,:]
    X_test_np  = X_np[n_train:, :]
    y_train_np = y_np[:n_train,:]
    y_test_np = y_np[n_train:, :]

    X_train_t = torch.tensor(X_train_np)
    y_train_t = torch.tensor(y_train_np)
    X_test_t =  torch.tensor(X_test_np)
    y_test_t = torch.tensor(y_test_np)

    n_input = X_train_t.size()[1]

    model = NeuralRegressor(n_input, 40, 50)
    trainset = list(zip(X_train_t, y_train_t))
    testset = list(zip(X_test_t,y_test_t ))
    name = 'neural_regression'

    trainloader = DataLoader(trainset, batch_size = model.batch_size)
    testloader = DataLoader(testset, batch_size = 1)
    for i in range(model.n_epochs):
        loss_train = model.train_one_epoch(trainloader)
        loss_test = model.test(testloader)
        if i%10==0:
            torch.save(model.net.state_dict(), 'model_reg_state_dict')
        print("epoch :  ", i+1)
        print("train_loss", str(round(loss_train, 5)))
        print("test_loss", str(round(loss_test.item(), 5)))
        with open("plots/text_files/train_loss_"+name+".txt","a") as f:
            f.write(str(round(loss_train, 5))+ '\n')
        with open("plots/text_files/test_loss_"+name+".txt","a") as f:
            f.write(str(round(loss_test.item(), 5))+ '\n')

