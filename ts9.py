import numpy as np
from numpy import array
import time
# '''
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from numpy import concatenate
from matplotlib import pyplot  as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
#'''
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers

'''
import FP_data as fpd
import FP_config as fpc

#print(fpd.fetchData(5).dtypes) # all cols
fpd.fetchData(100).to_csv('1e2data.csv')
fpd.fetchData(1000).to_csv('1e3data.csv')
fpd.fetchData(2000).to_csv('2e3data.csv')
fpd.fetchData(5000).to_csv('5e3data.csv')
fpd.fetchData(10000).to_csv('1e4data.csv')
#'''

cols = ['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty', "bid_hl", "bid_cl", "bid_ho", "bid_co"]


def fitData(data,cols):
    high = max(data[cols[2]])
    low = min(data[cols[3]])
    for i in range(0,4):
        data[cols[i]] = (data[cols[i]]-low)/(high-low)
    for i in range(4,8):
        data[cols[i]] = data[cols[i]]/max(data[cols[i]])
    i = 8
    mx = max(max(data[cols[i]]), abs(min(data[cols[i]])))
    data[cols[i]] = data[cols[i]] / mx


def getData(file,back,fore):
    data = read_csv(file, header=0, index_col=0)
    print(data.head())
    values = data
    fitData(values,cols)
    print(data.head())
    for each in cols:
        print(each,' max ',max(data[each]),' min ',min(data[each]))
    values = values.values
    len_v_x = int(.05 * len(values))

    x = []
    y = []
    for i in range(len(values) - back - 1 - fore):
        t = []
        for j in range(0, back):
            t.append(values[[(i + j)], :])
        x.append(t[::-1])
        tmpy = ()
        tmpy += (values[i + back-1, 1],)  # origin value for mapping
        #tmpy += (values[i + back-1 + fore, 1],)  # forwards value
        for n in range(fore): #waypoints
            tmpy += (values[i + back + fore - n, 1],)
        y.append(tmpy)

    x, y = np.array(x), np.array(y)
    t_x, t_y = x[len_v_x + back:], y[len_v_x + back:]
    v_x, v_y = x[:len_v_x + back], y[:len_v_x + back]
    t_x, v_x = t_x.reshape(t_x.shape[0], back, len(cols)), v_x.reshape(v_x.shape[0], back, len(cols))
    # t_y,v_y = t_y.reshape(t_y.shape[0],1,1),v_y.reshape(v_y.shape[0],1,1)
    print(t_x.shape)
    print(t_y.shape)
    print(v_x.shape)
    print(v_y.shape)
    return x,y,t_x,t_y,v_x,v_y

back = 30
fore = 2
x,y,t_x,t_y,v_x,v_y = getData('1e4data.csv',back,fore)

nodes = 30
model = Sequential()
model.add(LSTM(nodes,return_sequences=True,activation='relu',input_shape=(t_x.shape[1],t_x.shape[2]),
               kernel_initializer='normal',kernel_constraint=maxnorm(3),
               ))
model.add(Dropout(rate=.2))
model.add(LSTM(nodes, return_sequences=True, activation='relu',
               kernel_initializer='normal', kernel_constraint=maxnorm(3)))
for i in range(3):
    model.add(LSTM(nodes,return_sequences=True,activation='relu'))
model.add(LSTM(nodes,activation='relu'))
#model.add(Dense(20))
model.add(Dense(t_y.shape[1]-1)) #foreward ty0 is guide value
model.summary()
optimizer = optimizers.adam()
model.compile(optimizer=optimizer,loss='mean_squared_error')


def results(v_x,v_y):
    print('eval : ',model.evaluate(v_x,v_y[:,1:]))
    pred = model.predict(v_x)

    npv = np.array(v_y)
    npp = np.array(pred)
    print(npv[0:5])
    print(npp[0:5])
    print('error')
    print(npv[0:5,1:]-npp[0:5])
    plt.figure()
    plt.plot(npv[:,0], color='black')
    #plt.plot(range(fore, fore + len(npp)), npp[:, 0])
    for i in range(npp.shape[1]):
        plt.plot(range(i+1,i+1+len(npp)),npp[:,i])
    plt.show()


def mfit(epochs, batch, groups=1):
    mxeval = 99999.
    n = 'modMAX{0}.h5'.format(round(time.time()))
    for i in range(epochs):
        print(i+1,' of ',epochs)
        model.fit(t_x,t_y[:,1:],epochs=groups,batch_size=batch,verbose=2)
        eval = model.evaluate(v_x, v_y[:,1:],verbose=0)
        if mxeval > eval:
            mxeval=eval
            print ('saved ',n)
            model.save(n)
        print('eval : ', eval,', maxeval : ',mxeval)
    results(v_x, v_y)
    #results(t_x, t_y)


# ts9.results(ts9.t_x,ts9.t_y)
# ts9.results(ts9.v_x,ts9.v_y)
# ts9.mfit(40,5)
# ts9.x,ts9.y,ts9.t_x,ts9.t_y,ts9.v_x,ts9.v_y = ts9.getData('1e4data.csv',ts9.back,ts9.fore)
# ts9.model.fit(ts9.t_x,ts9.t_y[:,1:],epochs=100,batch_size=1,verbose=2)
# ts9.model = ts9.load_model('modMAX1559282096.h5')
mfit(5,7,5)
def train(thresh,e=5,b=7,g=25):
    ag = []
    while model.evaluate(t_x, t_y[:,1:],verbose=0)>thresh:
        mag = model.evaluate(t_x, t_y[:,1:],verbose=0)
        ag.append(mag)
        print(mag)
        mfit(e,b,g)
    print(len(ag))
    return ag
train(0.0005,e=5,b=7,g=25)
#mfit(3,1)
#mfit(10,5)
#



