import csv
import numpy as np
import pandas as pd
from collections import Counter
from Functions import *
from verstack.stratified_continuous_split import scsplit
from datetime import datetime
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import spacy
from spacy.cli import download
import sys
import tqdm


L = [0, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]

def classifier(x):
    return sum([x>=elem for elem in L[1:]])

def inv_classifier(x):
    return L[x]



def transf(X, y): # Time, Hashtags, 

    X = add_col(X, lambda x : decimal_hashtags(extract_website_url(x), url_most_used), 'urls', 'urls_enc')
    X = add_col(X, lambda x : decimal_hashtags(hashtags_list_to_list(x), hashtags_most_used), 'hashtags', 'hashtags_enc')
    X = add_col(X, lambda x : int(list(set(hashtags_list_to_list(x)) & set(covid_hashtags))!=[]), 'hashtags', 'covid_hashtags')
    X = add_col(X, lambda x : np.log(1+x), 'user_followers_count', 'log_follower')
    X = add_col(X, lambda x : np.log(1+x), 'user_statuses_count', 'log_statuses')
    X = add_col(X, lambda x : np.log(1+x), 'user_friends_count', 'log_friends')
    Xt = X[['user_verified', 'log_statuses', 'log_friends', 'log_follower', 
           'urls_enc', 'hashtags_enc', 'covid_hashtags']]
    yt = y
    return Xt, yt

def spacy_sum_vectors(phrase, nlp):
    dec = nlp(phrase)
    return sum(w.vector for w in dec)

def spacy_word2vec_features(X, nlp):
    l = []
    for p in tqdm.tqdm(X):
        l.append(spacy_sum_vectors(p, nlp))
    return np.vstack(l)


def get_nlp_cols_train(X_train, nlp_model,n_pca=5):
    word_features = spacy_word2vec_features(X_train["text"], nlp_model) ## c'est un pd dataframe 
    pca = PCA()
    pca.fit(word_features)
    features_pca = pca.transform(word_features)[:, :n_pca]
    name_feats = ["word2vec_"+str(i) for i in range(n_pca)]
    text_df = pd.DataFrame(features_pca, columns = name_feats)
    return text_df, pca

def get_nlp_cols_test(X_test, nlp_model, pca, n_pca=5):
    word_features = spacy_word2vec_features(X_test["text"], nlp_model) 
    features_pca = pca.transform(word_features)[:, :n_pca]
    name_feats = ["word2vec_"+str(i) for i in range(n_pca)]
    text_df = pd.DataFrame(features_pca, columns = name_feats)
    return text_df



def get_tranformed_dataset(X_train,y_train, X_test, y_test, nlp_model):
    pca = PCA()
    text_df_train, pca = get_nlp_cols_train(X_train, nlp_model, n_pca=5)
    X_tn, y_tn = transf(X_train, y_train)
    X_tn_final = pd.concat([X_tn,text_df_train], axis = 1)
    text_df_test = get_nlp_cols_test(X_test, nlp_model, pca, n_pca=5)
    X_ts, y_ts = transf(X_test, y_test)
    X_ts_final = pd.concat([X_ts,text_df_test], axis = 1)
    return X_tn_final,y_tn,X_ts_final,y_ts

def triage(elem):
    return elem[1]


if __name__ == "__main__":
    # name = 'debugging'
    # n_debug = 1500

    # params
    lim_number_urls = 40
    lim_number_hashtags = 50
    splitt_coeff = 0.999

    train_data = pd.read_csv("data/train.csv")
    X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweet_count'], stratify=train_data['retweet_count'], train_size=splitt_coeff, test_size=1-splitt_coeff)

    # X_train = X_train.drop(['retweet_count'], axis=1)[:n_debug] ##### a enlever une foid debug
    # X_test = X_test.drop(['retweet_count'], axis=1)[:n_debug] ##### a enlever une foid debug

    # y_train = y_train[:n_debug]##### a enlever une foid debug
    # y_test = y_test[:n_debug]##### a enlever une foid debug

    X_train = X_train.drop(['retweet_count'], axis=1)
    X_test = X_test.drop(['retweet_count'], axis=1)


    liste_url_ = []
    for elem in X_train['urls']:
        if not type(elem) == float:
            liste_url_.append(extract_website_url(elem))

    uniques_url = Counter(liste_url_).keys()
    nombre_url = Counter(liste_url_).values()

    plus_utilises = [(list(uniques_url)[i], list(nombre_url)[i]) for i in range(len(nombre_url))]
    plus_utilises.sort(key = triage, reverse = True)

    url_most_used = [elem[0] for elem in plus_utilises[:lim_number_urls]]


    liste_hashtags = []
    for elem in X_train['hashtags']:
        if not type(elem) == float:
            liste_hashtags.extend(hashtags_list_to_list(elem))

    uniques_hashtags = Counter(liste_hashtags).keys()
    nombre_hashtags = Counter(liste_hashtags).values()

    plus_utilises = [(list(uniques_hashtags)[i], list(nombre_hashtags)[i]) for i in range(len(nombre_hashtags))]
    plus_utilises.sort(key = triage, reverse = True)

    hashtags_most_used = [elem[0] for elem in plus_utilises[:lim_number_hashtags]]

    covid_hashtags = ['COVID19', 'coronavirus', 'Coronavirus', 'Covid19', 'covid19', 'COVID-19', 
                'CoronaVirus', 'Covid_19', 'COVID', 'Corona', 'COVID__19', 'Corona', 'SARSCoV2', 
                'CoronavirusPandemic']

    download("en_core_web_sm")
    nlp_model = spacy.load("en_core_web_sm")

    X_train,y_train,X_test,y_test = get_tranformed_dataset(X_train,y_train, X_test, y_test, nlp_model)
    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)
    
    X_train.to_csv("X_train.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)


    