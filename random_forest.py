# sklearn.ensemble.RandomForestClassifier

import csv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import time 
from preprocess import *

def classifier(x):
    return sum([x>=elem for elem in L[1:]])

def inv_classifier(x):
    return L[x]

class RandomForestModel:
    def __init__(self, *args, **kwargs):
        self.model =  RandomForestClassifier(*args, **kwargs)

    def fit_model(self, X,y):
        self.model.fit(X, y)

    def eval_model(self, X,y):
        y_hat = self.model.predict(X)
        y_pred = [inv_classifier(e) for e in y_hat]
        mae = mean_absolute_error(y_true = y, y_pred = y_pred)
        return mae

def get_results(max_depth, X_train_np,y_train_class, y_train_np, X_test_np, y_test_np):
    rand_forest_model = RandomForestModel(max_depth=max_depth)
    rand_forest_model.fit_model(X_train_np, y_train_class)
    mae_train = rand_forest_model.eval_model(X_train_np, y_train_np)
    mae_test = rand_forest_model.eval_model(X_test_np, y_test_np)
    print("MAX DEPTH : ", max_depth)
    print("MAE Train : ", mae_train)
    print("MAE Test : ", mae_test)
    return mae_train,mae_test

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
    y_train_class = np.array([classifier(y) for y in y_train_np])
    ## pour tester sans texte
    X_train_np = X_train_np[:, :7] 
    X_test_np = X_test_np[:, :7] 
    name = 'notext'

    depths = [5+i for i in range(45)]
    for d in depths:
        deb = time.time()
        tr,tst= get_results(d, X_train_np, y_train_class, y_train_np, X_test_np, y_test_np)
        with open("plots/text_files/train_loss_rf_"+name+".txt","a") as f:
            f.write(str(round(tr, 5))+ '\n')
        with open("plots/text_files/test_loss_rf_"+name+".txt","a") as f:
            f.write(str(round(tst, 5))+ '\n')
        print('TEMPS PRIS :', time.time()-deb, '\n')