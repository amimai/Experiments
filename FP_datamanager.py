import numpy as np
import time
from matplotlib import pyplot  as plt
from math import sqrt
from math import cos

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.preprocessing import MinMaxScaler

# fx columns

cols = ['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty', "bid_hl", "bid_cl", "bid_ho", "bid_co"]
def gen_data(amm):
    x = np.zeros((amm, 1))
    for i in range(amm):
        n = 0
        m = 1
        p = 1
        '''
        if i > 0:
            n = x[i-1][4]+i
            p = x[i-1][2]*2
        if i>1:
            m=  x[i-1][3]+x[i-2][3]
            # '''
        x[i] = [i]  # norm,norm,geometric,fibinacci,exponential
    return x


def fitData(data,cols): # onvert forex data to feed
    high = max(data[cols[2]])
    low = min(data[cols[3]])
    for i in range(0,4):
        data[cols[i]] = (data[cols[i]]-low)/(high-low)
    for i in range(4,8):
        data[cols[i]] = data[cols[i]]/max(data[cols[i]])
    i = 8
    mx = max(max(data[cols[i]]), abs(min(data[cols[i]])))
    data[cols[i]] = data[cols[i]] / mx
    return data

def difDataFX(data):
    dlen = len(data)
    data = np.array(data)
    mxtick = max(data[:,4])
    for i in range(dlen):
        cur = dlen-i-1
        data[cur, 0:4] = data[cur, 0:4]-data[cur-1, 0:4]
        data[cur, 4] = data[cur, 4]/mxtick
    return DataFrame(data[1:],columns=cols)

# combined dataloader
def getData(file,back,fore,targ): # forex datahandler
    data = read_csv(file, header=0, index_col=0)
    print(data.head())
    values = data
    print(values.head())
    values = values.values
    len_v_x = int(.05 * len(values))

    x = []
    y = []
    for i in range(len(values) - back - 1 - fore):
        t = []
        for j in range(0, back):
            t.append(values[[(i + j)], :])
        x.append(t)
        tmpy = ()
        tmpy += (values[i + back-1, targ],)  # origin value for mapping
        for n in range(fore):  # waypoints
            tmpy += (values[i + back + n, targ],)
        y.append(tmpy)

    x, y = np.array(x), np.array(y)
    t_x, t_y = x[len_v_x + back:], y[len_v_x + back:]
    v_x, v_y = x[:len_v_x + back], y[:len_v_x + back]
    t_x, v_x = t_x.reshape(t_x.shape[0], back, t_x.shape[3]), v_x.reshape(v_x.shape[0], back, v_x.shape[3])
    # t_y,v_y = t_y.reshape(t_y.shape[0],1,1),v_y.reshape(v_y.shape[0],1,1)
    print(t_x.shape)
    print(t_y.shape)
    print(v_x.shape)
    print(v_y.shape)
    return x, y, t_x, t_y, v_x, v_y


def getDataFX(file,back,fore,targ): # forex datahandler
    data = read_csv(file, header=0, index_col=0)
    print(data.head())
    values = fitData(data,cols)
    print(values.head())
    values = values.values
    len_v_x = int(.05 * len(values))

    x = []
    y = []
    for i in range(len(values) - back - 1 - fore):
        t = []
        for j in range(0, back):
            t.append(values[[(i + j)], :])
        x.append(t)
        tmpy = ()
        tmpy += (values[i + back-1, targ],)  # origin value for mapping
        for n in range(fore):  # waypoints
            tmpy += (values[i + back + n, targ],)
        y.append(tmpy)

    x, y = np.array(x), np.array(y)
    t_x, t_y = x[len_v_x + back:], y[len_v_x + back:]
    v_x, v_y = x[:len_v_x + back], y[:len_v_x + back]
    t_x, v_x = t_x.reshape(t_x.shape[0], back, t_x.shape[3]), v_x.reshape(v_x.shape[0], back, v_x.shape[3])
    # t_y,v_y = t_y.reshape(t_y.shape[0],1,1),v_y.reshape(v_y.shape[0],1,1)
    print(t_x.shape)
    print(t_y.shape)
    print(v_x.shape)
    print(v_y.shape)
    return x, y, t_x, t_y, v_x, v_y


def getDataD(file,back,fore,targ): # forex datahandler
    data = read_csv(file, header=0, index_col=0)
    print(data.head())
    values = data
    print(values.head())
    values = values.values
    len_v_x = int(.05 * len(values))

    x = []
    y = []
    for i in range(len(values) - back - 1 - fore):
        t = []
        cur  = (values[i + back-1, targ],)
        for j in range(0, back):
            t.append(values[[(i + j)], :]-cur)
        x.append(t)
        tmpy = ()
        tmpy += (values[i + back-1, targ],)  # origin value for mapping
        for n in range(fore):  # waypoints
            tmpy += (values[i + back + n, targ]-cur[targ],)
        y.append(tmpy)

    x, y = np.array(x), np.array(y)
    t_x, t_y = x[len_v_x + back:], y[len_v_x + back:]
    v_x, v_y = x[:len_v_x + back], y[:len_v_x + back]
    t_x, v_x = t_x.reshape(t_x.shape[0], back, t_x.shape[3]), v_x.reshape(v_x.shape[0], back, v_x.shape[3])
    # t_y,v_y = t_y.reshape(t_y.shape[0],1,1),v_y.reshape(v_y.shape[0],1,1)
    print(t_x.shape)
    print(t_y.shape)
    print(v_x.shape)
    print(v_y.shape)
    return x, y, t_x, t_y, v_x, v_y


def getDataDFX(file,back,fore,targ): # forex datahandler
    data = read_csv(file, header=0, index_col=0)
    print(data.head())
    values = data
    print(values.head())
    values = values.values
    len_v_x = int(.05 * len(values))
    mxtik = 25000.
    x = []
    y = []
    for i in range(len(values) - back - 1 - fore):
        t = []
        cur  = (values[i + back-1, targ],)
        for j in range(0, back):
            tmp=np.concatenate((values[[(i + j)], :4]-cur,values[[(i + j)], 3:4]/mxtik,values[[(i + j)], 5:]),axis=1)
            t.append(tmp)
        x.append(t[::-1]) # inverting timeframe provides markable improvment to learnin rate and gradient surface, tested on small 1e2 dataset .035 vs 7e-4 achived
        tmpy = ()
        tmpy += (values[i + back-1, targ],)  # origin value for mapping
        for n in range(fore):  # waypoints
            tmpy += (values[i + back + n, targ]-cur[0],)
        y.append(tmpy)

    x, y = np.array(x), np.array(y)
    t_x, t_y = x[len_v_x + back:], y[len_v_x + back:]
    v_x, v_y = x[:len_v_x + back], y[:len_v_x + back]
    t_x, v_x = t_x.reshape(t_x.shape[0], back, t_x.shape[3]), v_x.reshape(v_x.shape[0], back, v_x.shape[3])
    # t_y,v_y = t_y.reshape(t_y.shape[0],1,1),v_y.reshape(v_y.shape[0],1,1)
    print(t_x.shape)
    print(t_y.shape)
    print(v_x.shape)
    print(v_y.shape)
    return x, y, t_x, t_y, v_x, v_y

'''
#data = read_csv('1e2data.csv', header=0, index_col=0)
back = 5
fore = 2
x, y, t_x, t_y, v_x, v_y = getDataDFX('1e2data.csv',back,fore,1)
print(v_x[:1])
print(v_y[:1])

print(t_x.max())'''