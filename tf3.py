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

from FP_datamanager import getDataDFX
from sgdr import SGDRScheduler

back = 40
fore = 4
x,y,t_x,t_y,v_x,v_y = getDataDFX('1e2data.csv',back,fore,1)

schedule = SGDRScheduler(min_lr=1e-6,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(9415/15),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.2)

nodes = 48
model = Sequential()
model.add(LSTM(nodes*5,return_sequences=True,activation='relu',input_shape=(t_x.shape[1],t_x.shape[2]),
               kernel_initializer='normal',kernel_constraint=maxnorm(3),
               ))
model.add(Dropout(rate=.2))
model.add(LSTM(nodes*4,  activation='relu', #return_sequences=True,
               kernel_initializer='normal', kernel_constraint=maxnorm(3)
               ))
'''for i in range(0):
    model.add(LSTM(nodes*3-i,return_sequences=True,activation='relu'))
model.add(LSTM(nodes,activation='relu'))'''
model.add(Dense(60,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(60,activation='relu'))
model.add(Dense(t_y.shape[1]-1,activation='linear')) #foreward ty0 is guide value
model.summary()
optimizer = optimizers.sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #.adam(lr=1e-3,beta_2=.98,epsilon=1e-5) #)#
model.compile(optimizer=optimizer,loss='mean_squared_error')

def mtrain(ep=250):
    model.fit(t_x, t_y[:,1:], epochs=250, verbose=2,batch_size=15,  callbacks=[schedule]) #)#

def rtrain(min=0.001):
    mag = model.evaluate(t_x, t_y[:, 1:], verbose=0)
    while mag > 0.001:
        mtrain()
        mag = model.evaluate(t_x, t_y[:, 1:], verbose=0)
        n = 'modMAX{0}.h5'.format(round(time.time()))
        print('saved ', n, ' with eval ', mag)
        model.save(n)

#rtrain()

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


# tf3.results(tf3.t_x,tf3.t_y)
# tf3.results(tf3.v_x,tf3.v_y)
# ts9.mfit(40,5)
# tf3.x,tf3.y,tf3.t_x,tf3.t_y,tf3.v_x,tf3.v_y = tf3.getDataDFX('2e3data.csv',tf3.back,tf3.fore, 1)
# ts9.model.fit(ts9.t_x,ts9.t_y[:,1:],epochs=100,batch_size=1,verbose=2)
# tf3.model = tf3.load_model('modMAX1559523076-1e3max.h5')
# tf3.model.summary()



