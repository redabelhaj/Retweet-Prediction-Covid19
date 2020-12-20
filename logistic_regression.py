import csv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import time 
from preprocess import *


def classifier(x):
    return sum([x>=elem for elem in L[1:]])

def inv_classifier(x):
    return L[x]


class LogRegModel:
    def __init__(self):
        self.model =  LogisticRegression()

    def fit_model(self, X,y):
        self.model.fit(X, y)

    def eval_model(self, X,y):
        y_hat = self.model.predict(X)
        y_pred = [inv_classifier(e) for e in y_hat]
        mae = mean_absolute_error(y_true = y, y_pred = y_pred)
        return mae

def get_results(X_train_np,y_train_class, y_train_np, X_test_np, y_test_np):
    logreg = LogRegModel()
    logreg.fit_model(X_train_np, y_train_class)
    mae_train = logreg.eval_model(X_train_np, y_train_np)
    mae_test = logreg.eval_model(X_test_np, y_test_np)
    print("MAE Train : ", mae_train)
    print("MAE Test : ", mae_test)


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
    # X_train_np = X_train_np[:, :7] 
    # X_test_np = X_test_np[:, :7] 


    get_results( X_train_np, y_train_class, y_train_np, X_test_np, y_test_np)
