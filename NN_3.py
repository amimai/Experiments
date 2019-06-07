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
import logging

##from tqdm._tqdm_notebook import tqdm_notebook
##from keras.wrappers.scikit_learn import KerasClassifier

from matplotlib import pyplot as plt

import FP_config as FPc
import FP_data as FPd
import FP_getdata as FPg


##init block
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TZ'] = 'Asia/Kolkata'  # to set timezone; needed when running on cloud
#time.tzset()

stime = time.time()
iter_changes = "dropout_layers_0.4_0.4"
PATH_TO_DRIVE_ML_DATA = FPc.OUTPUT_PATH
INPUT_PATH = PATH_TO_DRIVE_ML_DATA + "/inputs"
OUTPUT_PATH = PATH_TO_DRIVE_ML_DATA + u"/outputs/lstm_best_23-5-19_{0}/".format(str(stime)) + iter_changes


# check if directory already exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print("Directory created", OUTPUT_PATH)
else:
    raise Exception("Directory already exists. Don't override.")


def build_timeseries(mat, y_col_index, TIME_STEPS, MULT, HIGH):
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
        x[i] = (mat[i:TIME_STEPS + i] -HIGH) * MULT
        y[i] = (mat[TIME_STEPS + i, y_col_index] - HIGH) * MULT
    print("length of time-series i/o", x.shape, y.shape)
    return x, y


def build_n_timeseries(mat, y_col_index, TIME_STEPS):
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
    h = np.zeros((dim_0,))
    print("dim_0", dim_0)
    for i in range(dim_0):
        height = mat[TIME_STEPS + i - 1, y_col_index]
        x[i] = mat[i:TIME_STEPS + i] - height
        y[i] = mat[TIME_STEPS + i, y_col_index] - height
        h[i] = height
    print("length of time-series i/o", x.shape, y.shape)
    return x, y, h


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0] % batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


def print_time(text, stime):
    seconds = (time.time() - stime)
    print(text, seconds // 60, "minutes : ", np.round(seconds % 60), "seconds")


DATASET = 8000
MULT = 1.
HIGH = 0.
TIME_STEPS = 30
TARGET_COL = 1

BATCH_SIZE = 20
LEARNING_RATE = 1e-3
EPOCHS = 20

Lstm_1 = 60
Lstm_2 = 0


##fetch data
cols = ['bidopen', 'bidclose', 'bidhigh', 'bidlow']
TARGET_COL = TARGET_COL
train_cols = cols
data = FPd.ticks(columns=cols, number=DATASET)
data.astype('float32')
data /= 64
df_train, df_test = train_test_split(data, train_size=0.9, test_size=0.1, shuffle=False)

##build out timeseries and normalise
x = df_train.loc[:,train_cols].values
proc_nx, proc_ny = build_timeseries(x, 1, TIME_STEPS, MULT, HIGH)
#proc_nx, proc_ny, proc_nh = build_n_timeseries(x, 1, TIME_STEPS)

##trim data
#train_nx, train_ny, train_nh = trim_dataset(proc_nx, BATCH_SIZE),trim_dataset(proc_ny, BATCH_SIZE),trim_dataset(proc_nh, BATCH_SIZE)
train_nx, train_ny = trim_dataset(proc_nx, BATCH_SIZE),trim_dataset(proc_ny, BATCH_SIZE)

##repeat for test set
x = df_test.loc[:,train_cols].values
proc_nx, proc_ny = build_timeseries(x, 1, TIME_STEPS, MULT, HIGH)
#test_nx, test_ny, test_nh = trim_dataset(proc_nx, BATCH_SIZE),trim_dataset(proc_ny, BATCH_SIZE),trim_dataset(proc_nh, BATCH_SIZE)
test_nx, test_ny = trim_dataset(proc_nx, BATCH_SIZE),trim_dataset(proc_ny, BATCH_SIZE)


##build out neuralnet
def create_model():
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    if Lstm_2 == 0:
        lstm_model.add(LSTM(Lstm_1, batch_input_shape=(BATCH_SIZE, TIME_STEPS, test_nx.shape[2]),
                            dropout=0.0, recurrent_dropout=0.0,  # stateful=True, return_sequences=True,
                            kernel_initializer='random_uniform'))
        lstm_model.add(Dropout(0.4))
    if Lstm_2 > 0:
        lstm_model.add(LSTM(Lstm_1, batch_input_shape=(BATCH_SIZE, TIME_STEPS, test_nx.shape[2]),
                            dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                            kernel_initializer='random_uniform'))
        lstm_model.add(Dropout(0.4))
        lstm_model.add(LSTM(Lstm_2, dropout=0.0)) #, stateful=True, return_sequences=True
        lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(20, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=LEARNING_RATE)
    # optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model

##load model
model = None
try:
    model = pickle.load(open("lstm_model", 'rb'))
    print("Loaded saved model...")
except FileNotFoundError:
    print("Model not found")

##train model
is_update_model = True
if model is None or is_update_model:
    from keras import backend as K

    print("Building model...")
    print("checking if GPU available", K.tensorflow_backend._get_available_gpus())
    model = create_model()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=40, min_delta=0.0001)

    mcp = ModelCheckpoint(os.path.join(OUTPUT_PATH,
                                       "best_model.h5"), monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)

    # Not used here. But leaving it here as a reminder for future
    r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30,
                                  verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    csv_logger = CSVLogger((OUTPUT_PATH + '/training_log_' + str(time.time()).replace(" ", "_") + '.log'),
                           append=True)

    history = model.fit(train_nx, train_ny, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE,
                        shuffle=False, validation_data=(test_nx, test_ny),
                        callbacks=[es, mcp, csv_logger])

    print("saving model...")
pickle.dump(model, open("lstm_model", "wb"))

y_pred = model.predict(test_nx, batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = test_ny
error = mean_squared_error(y_test_t, y_pred)
print("Error is", error, y_pred.shape, y_test_t.shape)
print(y_pred[0:15])
print(y_test_t[0:15])

# convert the predicted value to range of real data
y_pred_org = y_pred
# min_max_scaler.inverse_transform(y_pred)
y_test_t_org = y_test_t
# min_max_scaler.inverse_transform(y_test_t)
print(y_pred_org[0:15])
print(y_test_t_org[0:15])

# Visualize the training data
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
plt.savefig((OUTPUT_PATH + '/train_vis_BS_'+str(BATCH_SIZE)+"_"+str(time.time())+'.png'))

saved_model = load_model(os.path.join(OUTPUT_PATH, 'best_model.h5')) # , "lstm_best_7-3-19_12AM",
print(saved_model)

y_pred = saved_model.predict(test_nx, batch_size=BATCH_SIZE)
y_pred = y_pred.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
error = mean_squared_error(y_test_t, y_pred)
print("Error is", error, y_pred.shape, y_test_t.shape)
print(y_pred[0:15])
print(y_test_t[0:15])
y_pred_org = y_pred
y_test_t_org = y_test_t
print(y_pred_org[0:15])
print(y_test_t_org[0:15])

# Visualize the prediction
plt.figure()
plt.plot(y_pred_org)
plt.plot(y_test_t_org)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
#plt.show()
plt.savefig((OUTPUT_PATH + '/pred_vs_real_BS'+str(BATCH_SIZE)+"_"+str(time.time())+'.png'))
print_time("program completed ", stime)