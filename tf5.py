import numpy as np
from numpy import array
import time
from random import randrange
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

from FP_datamanager import getDataDFX
from CosHotRestarts import CosHotRestart
from sgdr import SGDRScheduler

back = 20 # 40
fore = 4
x,y,t_x,t_y,v_x,v_y = getDataDFX('1e4data.csv',back,fore,1)

nodes = 10
nodes2 = 50
model = Sequential()
model.add(LSTM(nodes*5,return_sequences=True,activation='relu',input_shape=(t_x.shape[1],t_x.shape[2]),
               #kernel_initializer='normal',kernel_constraint=maxnorm(3),
               ))
'''
#model.add(Dropout(rate=.2))
model.add(LSTM(nodes*4,  activation='relu', return_sequences=True,
               #kernel_initializer='normal', kernel_constraint=maxnorm(3)
               ))
for i in range(1):
    model.add(LSTM(nodes*(3-i),return_sequences=True,activation='relu'))'''
model.add(LSTM(nodes*4,activation='relu'))
model.add(Dense(nodes2,activation='relu'))
for i in range(3):
    model.add(Dense(nodes2,activation='relu'))
model.add(Dense(t_y.shape[1]-1,activation='linear')) #foreward ty0 is guide value

model.summary()
optimizer = optimizers.adam(lr=1e-2,beta_2=.98,epsilon=1e-5) #.sgd(lr=.1, momentum=0.9) #)#
model.compile(optimizer=optimizer,loss='mean_squared_error')


def results(v_x,v_y, save=0):
    eval = model.evaluate(v_x,v_y[:,1:])
    print('eval : ',eval)
    pred = model.predict(v_x)

    npv = np.array(v_y)
    npp = np.array(pred)
    print(npv[0:5])
    print(npp[0:5])
    print('error')
    print(npv[0:5,1:]-npp[0:5])
    plt.figure(figsize=(16,8),frameon=False)
    plt.plot(npv[:,0], color='black')
    #plt.plot(range(fore, fore + len(npp)), npv[:,0]+npp[:, 0])
    for i in range(npp.shape[1]):
        plt.plot(range(i+1,i+1+len(npp)),npv[:,0]+npp[:,i])
    if save == 1:
        plt.savefig('validation-time{1}-eval{0}.png'.format(eval,round(time.time())))
    else:
        plt.show()

def mtrain(t_x,t_y, ep=250, cycles=25, bs=15, coshot=True):
    if coshot:
        CosHot = CosHotRestart(nb_epochs=ep, nb_cycles=cycles, gain=1.1, verbose=0)
        model.fit(t_x, t_y, epochs=ep, verbose=2,batch_size=bs,  callbacks=[CosHot]) #)#
    else:
        SGDR = SGDRScheduler(min_lr=1e-6,max_lr=1e-2,
                             steps_per_epoch=np.ceil(len(t_x) / 15),
                             lr_decay=0.9,cycle_length=5,mult_factor=1.2)
        model.fit(t_x, t_y, epochs=ep, verbose=2, batch_size=bs, callbacks=[SGDR])

def rtrain(min=0.001):
    mag = model.evaluate(t_x, t_y[:, 1:], verbose=0)
    var = True
    while mag > 0.001:
        mtrain(t_x, t_y[:,1:],coshot=var)
        var = not var #use erratic coshot to bounce moddel over humps and greedy SGDR to smoothe to lower loss
        mag = model.evaluate(t_x, t_y[:, 1:], verbose=0)
        n = 'modMAX{0}.h5'.format(round(time.time()))
        print('saved ', n, ' with eval ', mag)
        model.save(n)

def batcher(minl=0.003, minb = 600, bl=100, gain=1.1):
    lx = len(t_x)
    if lx < minb:
        rtrain(minl)
    else:
        a = randrange(0, lx-bl)
        n_x = t_x[a:a+bl]
        n_y = t_y[a:a+bl]
        print(t_x.shape, n_x.shape)
        tlen = int(minb/2)
        while len(n_x)<lx/(gain*1.1):
            while len(n_x)<tlen:
                a = randrange(0, lx-bl)
                n_x = np.concatenate((n_x,t_x[a:a+bl]),axis=0)
                n_y = np.concatenate((n_y,t_y[a:a+bl]),axis=0)
            print (t_x.shape, n_x.shape)
            mag = model.evaluate(n_x, n_y[:, 1:], verbose=0)
            var = True
            while mag > minl:
                mtrain(n_x, n_y[:, 1:], coshot=var)
                var = not var  # use erratic coshot to bounce moddel over humps and greedy SGDR to smoothe to lower loss
                mag = model.evaluate(n_x, n_y[:, 1:], verbose=0)
                print('eval ', mag)
            n = 'modMAX{0}.h5'.format(round(time.time()))
            print('saved ', n, ' with eval ', mag)
            model.save(n)
            tlen = min(len(n_x) * gain, lx / gain)
    rtrain(minl)


batcher()