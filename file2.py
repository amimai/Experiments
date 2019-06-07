import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os
import sys
import time
import pandas as pd
from tqdm._tqdm_notebook import tqdm_notebook
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

import FP_config as FPc
import FP_data as FPd
import FP_timeseries as fpt

OUTPUT_PATH = FPc.OUTPUT_PATH

train_cols = FPc.train_cols
BATCH_SIZE = FPc.BATCH_SIZE
TIME_STEPS = FPc.TIME_STEPS
lr = 0.00100000 #LearRate

print ('getting data')
df_ge = FPd.fetchData()

df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()

x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

print('transforming')
x_t, y_t = fpt.build_timeseries(x_train, 1)
x_t = fpt.trim_dataset(x_t, BATCH_SIZE)
y_t = fpt.trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = fpt.build_timeseries(x_test, 1)
x_val, x_test_t = np.split(fpt.trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(fpt.trim_dataset(y_temp, BATCH_SIZE),2)

print('startinf NN')
lstm_model = Sequential()
lstm_model.add(LSTM(100,
                    batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                    dropout=0.0, recurrent_dropout=0.0,
                    stateful=True,
                    kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))
optimizer = optimizers.RMSprop(lr=lr)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)

print('Opening logs')
csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'NN_Log1' + '.log'), append=True)

model = None
try:
    model = pickle.load(open("lstm_model", 'rb'))
    print("Loaded saved model...")
except FileNotFoundError:
    print("Model not found")

print('training')
history = model.fit(x_t, y_t, epochs=10, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
trim_dataset(y_val, BATCH_SIZE)), callbacks=[csv_logger])