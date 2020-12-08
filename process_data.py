""" 
MIT License
Copyright (c) 2020 Yoga Suhas Kuruba Manjunath
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import sys
import pickle
#import config
import numpy as np
#import config
import socket, struct
import pandas as pd
import datetime as dt
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random
from regressors import stats
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LSTM, Reshape
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import time
import os
import config
from matplotlib import pyplot 
import sys
#from process_data import get_dataset, train_test_validation_set_split
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from urllib.request import urlopen
import zipfile, io

random.seed(10)

def load_data():
    url = "https://iotanalytics.unsw.edu.au/iottestbed/csv/16-09-23.csv.zip"
        
    archive = zipfile.ZipFile(io.BytesIO(urlopen(url).read())) # Takes some time
    
    csv_path = archive.namelist()
    
    df = pd.read_csv(io.BytesIO(archive.read(csv_path[0])))
    
    
    labelencoder = LabelEncoder()
    df['eth.src'] = labelencoder.fit_transform(df['eth.src'])
    df['eth.dst'] = labelencoder.fit_transform(df['eth.dst'])
    df['IP.src'] = labelencoder.fit_transform(df['IP.src'])
    df['IP.dst'] = labelencoder.fit_transform(df['IP.dst'])
    

    df = df.sample(frac = 1)

    X = np.array(df.iloc[:,df.columns !="eth.src"])
    y = np.array(df["eth.src"])

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    df = df.drop(columns=["eth.src"])

    return X,y

def get_dataset():

    path, dirs, file = next(os.walk("."))

    if ( "X.pickle" in file) and ("y.pickle" in file) and ("num_classes.pickle" in file):
        pickle_in = open("X.pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("y.pickle","rb")
        y = pickle.load(pickle_in)
        
        pickle_in = open("num_classes.pickle","rb")
        config.num_classes = pickle.load(pickle_in)        
        
        return X, y
    else :
        X, y = load_data()        

        config.num_classes = len(set(y))
        
        pickle_out = open("X.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()        
           
        pickle_out = open("y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        
        pickle_out = open("num_classes.pickle","wb")
        pickle.dump(config.num_classes, pickle_out)
        pickle_out.close()    
                
        return X, y
        

def train_test_validation_set_split(x, y, train_ratio, test_ratio, validation_ratio):
    x_train, x_interim = np.split(x, [int(train_ratio *len(x))])
    y_train, y_interim = np.split(y, [int(train_ratio *len(y))])

    x_test, x_val = np.split(x_interim, [int(test_ratio *len(x))])
    y_test, y_val = np.split(y_interim, [int(test_ratio *len(y))])

    return x_train, x_test, x_val, y_train, y_test, y_val




if __name__ == "__main__":
    #X,y = load_data()
    
    #print(X)
    #print(y)
    #download_data()
    
