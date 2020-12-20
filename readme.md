# Covid Retweet Prediction challenge

## Requirements

Standard ML libraries (Scikit, numpy, pandas) plus :
torch
verstack
spacy

## Data

Please put the data files train.csv and test.csv in the data folder.

## Preprocessing

The file preprocess.py preprocesses the data. It will create csv files X_train.csv X_test.csv y_train.csv y_test.csv

## Models

These files are used by the models (logistic regression, neural classification, neural regression, random_forest, xgboost classifier). The structure is always the same : we define a model class with methods to train it and to evaluate it. The main method contains a training + evaluation script

When running these files, results of the corresponding experiments will be printed on the screen, and sometimes some data (for plots) will be saved. 
Please create a folder plots with two subfolders text_files and images : the files for plots will be stored in text_files and the images (plots) will be stored in images

## Plots

Running plots.py creates the plots and saves the images in the corresponding folder

## Other files
Functions.py contains various functions used in the rest of the code