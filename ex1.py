import numpy as np
import os
import sys
import time
import pandas as pd
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
##from tqdm._tqdm_notebook import tqdm_notebook
##from keras.wrappers.scikit_learn import KerasClassifier

from matplotlib import pyplot as plt

#network structure
lstm_pre_layers = 1
lstm_pre_units = 9

lstm_mid_layers = 2
lstm_mid_units = 9

lstm_re_layers = 0
lstm_re_units = 3

dense_layers = 0
dense_units = 20

out_layers = 0

params = {
    'learning_rate' : 0.001
}

#training sets
epochs = 500
batchsize = 50

leng = 9
transamples = 1000

normalization = 1000


data = [[i+j for j in range(leng)] for i in range(transamples)]
data = np.array(data, dtype=np.float32)
target = [[i+j+1 for j in range(leng)] for i in range(1,transamples+1)]
target = np.array(data, dtype=np.float32)

print (data[1],target[1])
data = data.reshape(transamples,1,leng)/normalization
target = target.reshape(transamples,1,leng)/normalization
print (data[1])

model = Sequential()
for i in range(lstm_pre_layers):
    model.add(LSTM(lstm_pre_units,
                   input_shape=(1,leng),
                   return_sequences=True,
                   activation='sigmoid',
                   dropout = 0.0,
                   kernel_initializer='random_uniform'
                   ))

for i in range(lstm_mid_layers):
    model.add(LSTM(lstm_mid_units,
                   return_sequences=True,
                   activation='sigmoid',
                   dropout = 0.0,
                   kernel_initializer='random_uniform'
                   ))

for i in range(lstm_re_layers):
    model.add(LSTM(lstm_re_units,
                   input_shape=(1,leng),
                   return_sequences=True,
                   activation='sigmoid',
                   dropout = 0.0,
                   kernel_initializer='random_uniform'
                   ))

for i in range(dense_layers):
    model.add(Dense(dense_units,
                    activation='relu'))
for i in range(out_layers):
    model.add(Dense(leng, activation='sigmoid'))

optimizer = optimizers.RMSprop(lr=params["learning_rate"])
model.compile(loss='mean_squared_error'
              ,optimizer=optimizer #'adam'
              ,metrics=['accuracy'])

model.fit(data,target,epochs=epochs
          ,batch_size=batchsize
          ,validation_data=(data,target))

predict = model.predict(data)

print (target[1]*normalization)
print (predict[1]*normalization)

print (target[30]*normalization)
print (predict[30]*normalization)


pred = model.predict(data)
pred *= normalization
v_yn =target*normalization
v_xn =data*normalization
for i in range (10):
    print(v_xn[i])
    print ("testset = ",pred[i], ":::", v_yn[i]," = real")

