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
import FP_getdata as FPg

#network structure
lstm_pre_layers = 1
lstm_pre_units = 240

lstm_mid_layers = 0
lstm_mid_units = 360

lstm_re_layers = 0
lstm_re_units = 0

dense_layers = 1
dense_units = 20

out_layers = 1
out_units = 1





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

def nn_gen(nn_batchsize,nn_epochs,nn_learn,nn_timesteps,nn_name):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ['TZ'] = 'Asia/Kolkata'  # to set timezone; needed when running on cloud

    params = {
        "batch_size": nn_batchsize,  # 20<16<10, 25 was a bust
        "epochs": nn_epochs,
        "learning_rate": nn_learn,
        "time_steps": nn_timesteps,
        "name": nn_name
    }

    iter_changes = nn_name

    INPUT_PATH = PATH_TO_DRIVE_ML_DATA + "/inputs"
    OUTPUT_PATH = PATH_TO_DRIVE_ML_DATA + "/outputs/lstm_best_16-5-19_3AM/" + iter_changes
    TIME_STEPS = params["time_steps"]
    BATCH_SIZE = params["batch_size"]

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print("Directory created", OUTPUT_PATH)
    else:
        raise Exception("Directory already exists. Don't override.")

    stime = time.time()

    def create_model():
        model = Sequential()
        for i in range(lstm_pre_layers):
            model.add(LSTM(lstm_pre_units
                           ,batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2])
                           ,dropout=0.0, recurrent_dropout=0.
                           #,stateful=True ,return_sequences=True
                           #,activation='sigmoid'
                           ,kernel_initializer='random_uniform'
                           ))
            model.add(Dropout(0.4))
        for i in range(lstm_mid_layers):
            model.add(LSTM(lstm_mid_units
                           ,stateful=True
                           #,return_sequences=True ,activation='sigmoid'
                           ,dropout=0.0, recurrent_dropout=0.
                           ,kernel_initializer='random_uniform'
                           ))
            model.add(Dropout(0.4))
        for i in range(lstm_re_layers):
            model.add(LSTM(lstm_re_units
                           ,batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2])
                           #,stateful=True ,return_sequences=True
                           #,activation='sigmoid'
                           , dropout=0.0, recurrent_dropout=0.
                           ,kernel_initializer='random_uniform'
                           ))
            model.add(Dropout(0.4))
        for i in range(dense_layers):
            model.add(Dense(dense_units,
                            activation='relu'))
        for i in range(out_layers):
            model.add(Dense(out_units, activation='sigmoid'))

        optimizer = optimizers.RMSprop(lr=params["learning_rate"])
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    model = None
    try:
        model = pickle.load(open("lstm_model", 'rb'))
        print("Loaded saved model...")
    except FileNotFoundError:
        print("Model not found")

    data = FPd.ticks(number=10000)
    x_t, y_t = FPg.get_dataset(data,TIME_STEPS,BATCH_SIZE,80.0,50.0)
    x_t,x_temp = train_test_split(x_t, train_size=0.8, test_size=0.2, shuffle=False)
    y_t, y_temp = train_test_split(y_t, train_size=0.8, test_size=0.2, shuffle=False)

    x_val, x_test_t = np.split(x_temp, 2)
    y_val, y_test_t = np.split(y_temp, 2)

    print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)

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

        history = model.fit(x_t, y_t, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                            shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                            trim_dataset(y_val, BATCH_SIZE)),
                            callbacks=[es, mcp, csv_logger])

        print("saving model...")
    pickle.dump(model, open("lstm_model", "wb"))

    # model.evaluate(x_test_t, y_test_t, batch_size=BATCH_SIZE
    y_pred = model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
    y_pred = y_pred.flatten()
    y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
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
    # plt.show()
    plt.savefig((OUTPUT_PATH + '/train_vis_BS_' + str(BATCH_SIZE) + "_" + str(time.time()) + '.png'))

    # load the saved best model from above
    saved_model = load_model(os.path.join(OUTPUT_PATH, 'best_model.h5'))  # , "lstm_best_7-3-19_12AM",
    print(saved_model)

    y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
    y_pred = y_pred.flatten()
    y_test_t = trim_dataset(y_test_t, BATCH_SIZE)
    error = mean_squared_error(y_test_t, y_pred)
    print("Error is", error, y_pred.shape, y_test_t.shape)
    print(y_pred[0:15])
    print(y_test_t[0:15])
    y_pred_org = (y_pred )  # min_max_scaler.inverse_transform(y_pred)
    y_test_t_org = (y_test_t)  # min_max_scaler.inverse_transform(y_test_t)
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
    # plt.show()
    plt.savefig((OUTPUT_PATH + '/pred_vs_real_BS' + str(BATCH_SIZE) + "_" + str(time.time()) + '.png'))
    print_time("program completed ", stime)

    r = 0.0
    return r