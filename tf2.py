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

from snapshot import Snapshot

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

back = 60
fore = 4
x,y,t_x,t_y,v_x,v_y = getDataDFX('1e3data.csv',back,fore,1)

nodes = 20
model = Sequential()
model.add(LSTM(nodes*3,return_sequences=True,activation='relu',input_shape=(t_x.shape[1],t_x.shape[2]),
               kernel_initializer='normal',kernel_constraint=maxnorm(3),
               ))
model.add(Dropout(rate=.2))
model.add(LSTM(nodes*2, return_sequences=True, activation='relu',
               kernel_initializer='normal', kernel_constraint=maxnorm(3)))
for i in range(3):
    model.add(LSTM(nodes,return_sequences=True,activation='relu'))
model.add(LSTM(nodes,activation='relu'))
#model.add(Dense(20))
model.add(Dense(t_y.shape[1]-1)) #foreward ty0 is guide value
model.summary()
optimizer = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=["mse"])


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
        plt.savefig('validation-eval{0}-time{1}.png'.format(eval,round(time.time())))
    plt.show()


cbs = [Snapshot('snapshots', nb_epochs=11, verbose=2, nb_cycles=2)]
model.fit(t_x, t_y[:,1:], epochs=10,batch_size=7,callbacks=cbs,shuffle=True,verbose=2)
          #validation_data=(v_x,v_y[:,1:]))


# tf1.results(tf1.t_x,tf1.t_y)
# tf1.results(tf1.v_x,tf1.v_y)
# ts9.mfit(40,5)
# ts9.x,ts9.y,ts9.t_x,ts9.t_y,ts9.v_x,ts9.v_y = ts9.getData('1e4data.csv',ts9.back,ts9.fore)
# ts9.model.fit(ts9.t_x,ts9.t_y[:,1:],epochs=100,batch_size=1,verbose=2)




