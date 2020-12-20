import csv
import numpy as np
import pandas as pd
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
# from xgboost import XGBClassifier, XGBRegressor
import time 
import torch
import tqdm
from torch.utils.data import DataLoader


from preprocess import *

def xgb_classifier_model(Xtrain, ytrain, max_d):
    t0 = time.time()
    model = XGBClassifier(max_depth = max_d, booster = 'gbtree', eval_metric = 'mae')
    model.fit(Xtrain, ytrain)
    print('time :', time.time()-t0)
    return model

L = [0, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]

def classifier(x):
    return sum([x>=elem for elem in L[1:]])

def inv_classifier(x):
    return L[x]
# debut=time.time()
# resultat = []

# X_train = pd.read_csv("X_train.csv")
# y_train =  pd.read_csv("y_train.csv")
# X_test =  pd.read_csv("X_test.csv")
# y_test =  pd.read_csv("y_test.csv")

# print('DATA LOADING prend : ', time.time() - debut)

# X_train_np = X_train.to_numpy()
# y_train_np = y_train.to_numpy()
# X_test_np =  X_test.to_numpy()
# y_test_np = y_test.to_numpy()

# y_train_class = np.array([classifier(y) for y in y_train_np])


# X_train_simple_np = X_train_np[:, :7]
# X_test_simple_np = X_test_np[:, :7]


# for i in range(3,7):
#     deb = time.time()
#     model_simple =xgb_classifier_model(X_train_simple_np, y_train_class, i)
#     print('\n')
#     print("MAX DEPTH:", i)
#     print("\n")
#     print("SANS TEXTE")
#     print("Prediction error test :", mean_absolute_error(y_true=y_test_np, y_pred=[inv_classifier(elem) for elem in model_simple.predict(X_test_simple_np)]))
#     print("Prediction error train :", mean_absolute_error(y_true=y_train_class, y_pred=[inv_classifier(elem) for elem in model_simple.predict(X_train_simple_np)]))
#     print("TEMPS PRIS : ", time.time()-debut)

#     # print(X_train_np.shape)

#     # print('DATA LOADING prend : ', time.time() - debut)
#     deb = time.time()
#     model_texte = xgb_classifier_model(X_train_np, y_train_class, i)
#     print("AVEC TEXTE")
#     print("Prediction error test :", mean_absolute_error(y_true=y_test_np, y_pred=[inv_classifier(elem) for elem in model_texte.predict(X_test_np)]))
#     print("Prediction error train :", mean_absolute_error(y_true=y_train_class, y_pred=[inv_classifier(elem) for elem in model_texte.predict(X_train_np)]))
#     print("TEMPS PRIS : ", time.time()-debut)




# for i in range(3,7):
#     print('max_depth :', i)
#     model = xgb_classifier_model(X_train_class, y_train_class_real, i)
#     resultat.append(model)
#     print("Prediction error test :", mean_absolute_error(y_true=y_test_real, y_pred=[inv_classifier(elem) for elem in model.predict(X_test_class)]))
#     print("Prediction error train :", mean_absolute_error(y_true=y_train_class_real, y_pred=[inv_classifier(elem) for elem in model.predict(X_train_class)]))
#     print('\n')


class NeuralClassifierNet(torch.nn.Module):
    def __init__(self, n_input, n_classes):
        super(NeuralClassifierNet, self).__init__()
        self.n_classes = n_classes
        self.layer1 = torch.nn.Linear(n_input,10)
        self.layer2 = torch.nn.Linear(10, n_classes)
    
    def forward(self,x):
        out = self.layer1(x)
        out = torch.sigmoid(out)
        out = self.layer2(out)
        return out

class NeuralClassifier:
    def __init__(self, n_input, n_classes, batch_size, n_epochs):
        self.net = NeuralClassifierNet(n_input, n_classes)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.batch_size = batch_size
        self.n_epochs = n_epochs



    def train_one_epoch(self,trainloader, lossfunc = torch.nn.CrossEntropyLoss()):
        loss_epoch =0
        for x,y in trainloader:
            self.optimizer.zero_grad()
            scores = self.net(x)
            loss = lossfunc(scores,y.squeeze())
            loss_epoch+=loss.item()
            loss.backward()
            self.optimizer.step()
        return loss_epoch/len(trainloader)
    
    def test_clf(self, testset):
        dataloader = DataLoader(testset, batch_size = 1)
        loss=0
        for x,y in dataloader:
            score = self.net(x)
            classe_hat = torch.argmax(torch.sigmoid(score))
            yhat = inv_classifier(classe_hat)
            nb_tw = inv_classifier(y)
            loss+= abs(nb_tw-yhat)
        return loss/len(dataloader)
    
    def test_reg(self, dataloader):
        # dataloader = DataLoader(testset, batch_size = 1)
        loss=0
        for x,y in dataloader:
            score = self.net(x)
            classe_hat = torch.argmax(torch.sigmoid(score))
            yhat = inv_classifier(classe_hat)
            loss+= abs(y-yhat)
        return loss/len(dataloader)

if __name__ == "__main__":
    L = [0, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
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


    y_train_class_t = torch.tensor([classifier(y) for y in y_train_np])
    y_test_class_t = torch.tensor([classifier(y) for y in y_test_np])

    ## pour tester sans texte
    X_train_t = X_train_t[:, :7] 
    X_test_t = X_test_t[:, :7] 
    n_input = X_train_t.size()[1]
    # print(X_test_t.shape)
    name = 'neural_clf_no_text'

    model = NeuralClassifier(n_input,len(L), 40,50)
    trainset = list(zip(X_train_t, y_train_class_t))
    testset_reg = list(zip(X_test_t,y_test_t ))
    

    trainloader = DataLoader(trainset, batch_size = model.batch_size)
    testloader = DataLoader(testset_reg, batch_size = 1)
    for i in range(model.n_epochs):
        loss_train = model.train_one_epoch(trainloader)
        loss_test = model.test_reg(testloader)
        if i%10==0:
            torch.save(model.net.state_dict(),name+'_state_dict')
        print("epoch :  ", i+1)
        print("train_loss", str(round(loss_train, 5)))
        print("test_loss", str(round(loss_test.item(), 5)))
        with open("plots/text_files/train_loss_"+name+".txt","a") as f:
            f.write(str(round(loss_train, 5))+ '\n')
        with open("plots/text_files/test_loss_"+name+".txt","a") as f:
            f.write(str(round(loss_test.item(), 5))+ '\n')
