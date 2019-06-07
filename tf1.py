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

back = 60
fore = 4
x,y,t_x,t_y,v_x,v_y = getDataDFX('1e3data.csv',back,fore,1)

nodes = 8
model = Sequential()
model.add(LSTM(nodes*4,return_sequences=True,activation='relu',input_shape=(t_x.shape[1],t_x.shape[2]),
               kernel_initializer='normal',kernel_constraint=maxnorm(3),
               ))
model.add(Dropout(rate=.2))
model.add(LSTM(nodes*3, return_sequences=True, activation='relu',
               kernel_initializer='normal', kernel_constraint=maxnorm(3)))
for i in range(2):
    model.add(LSTM(nodes*3-i,return_sequences=True,activation='relu'))
model.add(LSTM(nodes,activation='relu'))
#model.add(Dense(20,activation='relu'))
model.add(Dense(t_y.shape[1]-1,activation='linear')) #foreward ty0 is guide value
model.summary()
optimizer = optimizers.sgd() # .adam()#lr=1e-4,beta_2=.98,epsilon=1e-9) #
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
    results(v_x, v_y, 1)


# tf1.results(tf1.t_x,tf1.t_y)
# tf1.results(tf1.v_x,tf1.v_y)
# ts9.mfit(40,5)
# ts9.x,ts9.y,ts9.t_x,ts9.t_y,ts9.v_x,ts9.v_y = ts9.getData('1e4data.csv',ts9.back,ts9.fore)
# ts9.model.fit(ts9.t_x,ts9.t_y[:,1:],epochs=100,batch_size=1,verbose=2)
# tf1.model = tf1.load_model('modMAX1559354425.h5')

# mfit(5,50,50)
def train(thresh,e=5,b=7,g=25):
    ag = []
    mag = 999.
    while mag>thresh:
        ag.append(mag)
        print(mag)
        mfit(e,b,g)
        mag = model.evaluate(t_x, t_y[:, 1:], verbose=0)
    print('traintime : ',len(ag))
    return ag
#story = train(0.01,e=3,b=50,g=50)
story = train(0.001,e=3,b=7,g=50)

#tf1.story = tf1.train(0.0005)



