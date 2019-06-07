import numpy as np
import os
import sys
import time
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Input
from keras.models import Model
import logging

##from tqdm._tqdm_notebook import tqdm_notebook
##from keras.wrappers.scikit_learn import KerasClassifier

from matplotlib import pyplot as plt

import FP_config as FPc
import FP_data as FPd
import FP_getdata as FPg



Lstm_1 = 10
samples = 100

##fetch data
cols = ['bidopen', 'bidclose', 'bidhigh', 'bidlow']
data_width = len(cols)
TARGET_COL = 1
train_cols = cols


in1 = Input(shape=(1,data_width))
out1 = LSTM(Lstm_1,dropout=0.4, recurrent_dropout=0.0, stateful=True, return_sequences=True)(in1)
out1 = Model(inputs=in1, outputs=out1)

in2 = Input(shape=(1,data_width))
out2 = LSTM(Lstm_1,dropout=0.4, recurrent_dropout=0.0, stateful=True, return_sequences=True)(in2)
out2 = Model(inputs=in2, outputs=out2)

in3 = Input(shape=(data_width,))
out3 = LSTM(Lstm_1,dropout=0.4, recurrent_dropout=0.0, stateful=True, return_sequences=True)(in3)
out3 = Model(inputs=in3, outputs=out3)


in4 = concatenate([out1.output, out2.output])
out4 = LSTM(Lstm_1,dropout=0.4, recurrent_dropout=0.0, stateful=True, return_sequences=True)(in4)
out4 = Model(inputs=in4, outputs=out4)

in5 = concatenate([out3.output, out4.output])
out5 = LSTM(Lstm_1,dropout=0.4, recurrent_dropout=0.0)(in5)
out5 = Model(inputs=in5, outputs=out5)

in6 = out5
out6 = Dense(2, activation="relu")(in6)
out6 = Dense(1, activation="linear")(out6)

model = Model(inputs=[in1.input,in2.input,in3.input], outputs=out6)


data = [[i+j for j in range(data_width)] for i in range(samples)]
data = np.array(data, dtype=np.float32)
target = [[i+j+1 for j in range(data_width)] for i in range(1,samples+1)]
target = np.array(data, dtype=np.float32)


