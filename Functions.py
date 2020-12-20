import pandas as pd
import numpy as np

def add_col(dfm, calc, name_col, name): # pandas.DataFrame, (val -> val), str, str
    return pd.concat([dfm.reset_index(drop=True), pd.DataFrame({name:[calc(elem) for elem in dfm[name_col].tolist()]})], axis=1)

def mention(elem):
    if type(elem) == float:
        return 0
    else:
        count = 1
        for x in elem:
            if x==',':
                count+=1
        return count
    
def is_url(elem):
    return type(elem) != float

def extract_website_url(url):
    if type(url) == float:
        return ''
    x = url.find('/')
    if x == -1:
        return url
    else:
        return url[:x]

def weekday(date): # str "2019-03-01 10:11:12" -> int 0:6
    return pd.Timestamp(date[0:10]).dayofweek

def hour(date): # str "2019-03-01 10:11:12" -> int 10
    return int(date[11:13])

import random

def equilibrer(dataset, y):
    n = max(y)+1
    print(n)
    liste = [[] for i in range(n)]
    for i, elem in enumerate(dataset):
        liste[y[i]].append(elem)
        
    long = [len(elem) for elem in liste]
    print(long)
    obj = max(long)
    
    for i in range(n):
        while(len(liste[i]))<obj:
            liste[i].append(random.choice(liste[i]))
    
    dataset_final = []
    y_final = []
    for i, elem in enumerate(liste):
        dataset_final = dataset_final+elem
        y_final += [i for k in range(len(elem))]
    return np.array(dataset_final), y_final

def instance_classe(y):
    n = max(y)+1
    liste = [0 for i in range(n)]
    for elem in y:
        liste[elem]+=1
    return liste

def difference(X, Y):
    return([X[i]-Y[i] for i in range(len(X))])

def equilibrage(X, y, nombre_instance):
    n = max(y)+1
    
    liste_instance = [[] for i in range(n)]
    for i, elem in enumerate(X):
        liste_instance[y[i]].append(elem)
    
    nombre_instance_r = [len(elem) for elem in liste_instance]
    liste_final = [[] for i in range(n)]
    
    obj = difference(nombre_instance, nombre_instance_r)
    
    for i, elem in enumerate(obj):
        if elem<=0:
            liste_final[i] = liste_instance[i][0:len(liste_instance[i])+elem]
        else:
            liste_final[i] = liste_instance[i]
            for j in range(elem):
                liste_final[i].append(random.choice(liste_instance[i]))
    
    dataset_final = []
    y_final = []
    for i, elem in enumerate(liste_final):
        dataset_final = dataset_final+elem
        y_final += [i for k in range(len(elem))]
    return np.array(dataset_final), y_final

def hashtags_list_to_list(h_list):
    if type(h_list) == float:
        return []
    list_of_hashtags = []
    rank = 0
    flag = False
    for i, elem in enumerate(h_list):
        if elem == ',':
            flag = True
            if rank == 0:
                list_of_hashtags.append(h_list[rank:i])
            else:
                list_of_hashtags.append(h_list[rank+2:i])
            rank = i
    if flag:
        list_of_hashtags.append(h_list[rank+2:i+1])
    else:
        list_of_hashtags.append(h_list)
    return list_of_hashtags

def vec_encode(elem, hashtags_most_used):
    return [hashtags_most_used[i] in hashtags_list_to_list(elem) for i in range(len(hashtags_most_used))]

def binary(vect):
    binar = 0
    for i, elem in enumerate(vect):
        binar += elem*2**i
    return int(binar)

def binary_hashtags(elem, hashtags_most_used):
    if type(elem) == float:
        return 0
    else:
        return binary(vec_encode(elem, hashtags_most_used))
    
def decimal_hashtags(elem, hashtags_most_used):
    if type(elem) == float:
        return 0
    else:
        vec = vec_encode(elem, hashtags_most_used)
        try:
            return np.max(np.nonzero(vec))
        except:
            return 0

