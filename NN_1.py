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

import FP_config as FPc
import FP_data as FPd


def mdm_norm(x_dat, y_dat):
    x_max, x_min = (max(x_dat), min(x_dat))
    x_temp, y_temp = (x_dat - x_min, y_dat - x_min)
    width = max(abs(x_max), abs(x_min))
    x_ret, y_ret = (x_temp / width, y_temp / width)
    n_val = [x_max, width]
    return x_ret, y_ret, n_val, width


def mdm_reform(y_predict, n_val):
    r = (y_predict * n_val[1]) + n_val[0]
    return r


def print_time(text, stime):
    seconds = (time.time() - stime)
    print(text, seconds // 60, "minutes : ", np.round(seconds % 60), "seconds")


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0] % batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat



PATH_TO_DRIVE_ML_DATA = FPc.OUTPUT_PATH

def nn_gen(nn_batchsize,nn_epochs,nn_learn,nn_timesteps,nn_width,nn_name):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ['TZ'] = 'Asia/Kolkata'  # to set timezone; needed when running on cloud

    params = {
        "batch_size": nn_batchsize,  # 20<16<10, 25 was a bust
        "epochs": nn_epochs,
        "lr": nn_learn,
        "time_steps": nn_timesteps,
        "first_layer_width": nn_width,
        "name": nn_name
    }

    iter_changes = nn_name

    INPUT_PATH = PATH_TO_DRIVE_ML_DATA + "/inputs"
    OUTPUT_PATH = PATH_TO_DRIVE_ML_DATA + "/outputs/lstm_best_14-5-19_9AM/" + iter_changes
    TIME_STEPS = params["time_steps"]
    BATCH_SIZE = params["batch_size"]

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print("Directory created", OUTPUT_PATH)
    else:
        raise Exception("Directory already exists. Don't override.")

    def build_timeseries(mat, y_col_index):
        """
        Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
        number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
        :param mat: ndarray which holds the dataset
        :param y_col_index: index of column which acts as output
        :return: returns two ndarrays-- input and output in format suitable to feed
        to LSTM.
        """
        # total number of time-series samples would be len(mat) - TIME_STEPS
        dim_0 = mat.shape[0] - TIME_STEPS
        dim_1 = mat.shape[1]
        x = np.zeros((dim_0, TIME_STEPS, dim_1))
        y = np.zeros((dim_0,))
        print("dim_0", dim_0)
        for i in range(dim_0):
            x[i] = mat[i:TIME_STEPS + i]
            y[i] = mat[TIME_STEPS + i, y_col_index]
        print("length of time-series i/o", x.shape, y.shape)
        return x, y

    stime = time.time()
    print(os.listdir(INPUT_PATH))
    df_ge = FPd.fetchData()
    print(df_ge.shape)
    print(df_ge.columns)
    print(df_ge.dtypes)
    train_cols = ['bidopen', 'bidhigh', 'bidlow', 'bidclose', 'tickqty', "bid_hl", "bid_cl", "bid_ho", "bid_co"]
    df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
    print("Train--Test size", len(df_train), len(df_test))

    # scale the feature MinMax, build array
    x = df_train.loc[:, train_cols].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:, train_cols])

    print("Deleting unused dataframes of total size(KB)",
          (sys.getsizeof(df_ge) + sys.getsizeof(df_train) + sys.getsizeof(df_test)) // 1024)

    del df_ge
    del df_test
    del df_train
    del x

    print("Are any NaNs present in train/test matrices?", np.isnan(x_train).any(), np.isnan(x_train).any())
    x_t, y_t = build_timeseries(x_train, 3)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    print("Batch trimmed size", x_t.shape, y_t.shape)



#network structure
lstm_pre_layers = 1
lstm_pre_units = 3

lstm_mid_layers = 2
lstm_mid_units = 3

lstm_re_layers = 0
lstm_re_units = 3

dense_layers = 0
dense_units = 20

out_layers = 0

params = {
    'learning_rate' : 0.001
}

#training sets
epochs = 2000
batchsize = 50

leng = 3
transamples = 100

normalization = 200

data = [[i+j for j in range(leng)] for i in range(transamples)]
data = np.array(data, dtype=np.float32)
target = [[i+j+1 for j in range(leng)] for i in range(1,transamples+1)]
target = np.array(data, dtype=np.float32)

print (data[1])
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

print (data[1]*normalization)
print (target[1]*normalization)
print (predict[1]*normalization)
