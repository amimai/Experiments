
import FP_data as FPd
import FP_config as FPc
import numpy as np
import time


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

def build_timeseries(mat, y_col_index, TIME_STEPS,height,width):
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
    #n = []
    print("dim_0",dim_0)
    for i in range(dim_0):
        #width = max(mat[i:TIME_STEPS+i, y_col_index]) - min(mat[i:TIME_STEPS+i, y_col_index])
        #height = mat[TIME_STEPS+i-1, y_col_index]
        x[i] = (mat[i:TIME_STEPS+i] - height) / width
        y[i] = (mat[TIME_STEPS+i, y_col_index] - height) / width
#         if i < 10:
#           print(i,"-->", x[i,-1,:], y[i])
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

data = FPd.ticks(number=500)
time_steps = 120
batch_size = 40

def get_dataset(data=data, TIME_STEPS=time_steps, BATCH_SIZE=batch_size,height=100.0,width=10.0):
    cols = FPc.colums
    x = data.loc[:, cols].values

    x_t, y_t = build_timeseries(x, 1, TIME_STEPS,height,width)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)

    return x_t, y_t


