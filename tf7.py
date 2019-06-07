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
from keras.models import Sequential, load_model, Model, Input
from keras.layers import Dense, Dropout, Concatenate, Flatten, Activation
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
x,y,t_x,t_y,v_x,v_y = getDataDFX('1e3data.csv',back,fore,1)

nodes = 30
nodes2 = 30

LSTMin = Input(shape=(t_x.shape[1],t_x.shape[2]))

LSTM_1 = []
for i in range(3):
    LSTM_n = LSTM(nodes,return_sequences=True,activation='relu')(LSTMin)
    LSTM_1.append(LSTM_n)

LSTM_1_merge = Concatenate()(LSTM_1)
LSTM_2 = LSTM(nodes,activation='relu')(LSTM_1_merge)

flat = Flatten()(LSTM_1_merge)
flat = Dense(nodes2,activation='relu')(flat)

LSTM_2_merge = Concatenate()([LSTM_2, flat])

# amplifier layer (ex. tf6)
# uses multiple separate Dropout-Dense subnets to allow back propagation to be amplified
# demonstrable increase from 6->8 maximum trainable LSTM_merge layers
# with increase merg(LSTM layers) from 0(no subnet) to 1(2 subnets)
# to 2(8 subnets, unreliable without boosting via callbacks)
# uses principle proposed in [paper?] as each subnet acts as a concurrent micro batch
# diminishing return tested with up to 60 nets without significant improvements over 8
# kernel_constraint=maxnorm(3) is not beneficial unlike original dropout regularisation paper proposal
DRr=.4 # .4-.6 dropout ideal
amp = []
for i in range(2):
    amp_n = Dropout(rate=DRr)(LSTM_2_merge)
    amp_n = Dense(nodes2,activation='relu', kernel_initializer='normal',kernel_constraint=maxnorm(3))(amp_n)
    amp.append(amp_n)

activation = Concatenate()(amp)
activation = Dense(t_y.shape[1]-1,activation='linear')(activation)

model = Model(inputs=LSTMin,outputs=activation)

model.summary()
# (ex. ts8)modified adam hyperperamaters based on [paper?] reliable with lr=7e-6 on 2m+ params
optimizer = optimizers.adam(lr=1e-2,beta_2=.98,epsilon=1e-5) #.sgd(lr=.1, momentum=0.9) #)#
model.compile(optimizer=optimizer,loss='mean_squared_error')


#model.fit(t_x, t_y[:,1:],epochs=200,batch_size=15, verbose=2)

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

#(ex. tf2-4) CosHot and SGDR with automatic hot restarts
# cosine based learning rate annealing algorithms
# [paper https://arxiv.org/abs/1608.03983]
# CosHot reliable annealing algorithm with minimal backslide
# SGDR heavily modified batchwise annealing algorithm that generates erratic
# SGDR can backslide easily on large datasets
# achieves remarkable learning and generalisation statistics when successful and works well on smaller datasets
# merged with hot restarts in (ex. ts4)
def mtrain(t_x,t_y, ep=250, cycles=25, bs=15, coshot=True):
    if coshot:
        CosHot = CosHotRestart(nb_epochs=ep, nb_cycles=cycles, gain=1.1, verbose=0)
        model.fit(t_x, t_y, epochs=ep, verbose=2,batch_size=bs,  callbacks=[CosHot]) #)#
    else:
        SGDR = SGDRScheduler(min_lr=1e-6,max_lr=1e-2,
                             steps_per_epoch=np.ceil(len(t_x) / 15),
                             lr_decay=0.9,cycle_length=5,mult_factor=1.2)
        model.fit(t_x, t_y, epochs=ep, verbose=2, batch_size=bs, callbacks=[SGDR])

#(ex. tf1-tf4) rtrain recursive training manager
# automates model training and swaps between training callback modes
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

# (ex. tf5) progressive training set growth for training on large datasets
# useful for rapidly training on 1e4+ datasets with high randomness and complexity
# uses principle of knowledge transfer to maintain low loss while geometrical increasing dataset
# minl(target loss) 0.003 provides beter generalisation that 0.001 but chance of backslide still exists
# bl(random string of data) for sequential data, reduces risk of backsliding but also effects generalisation
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
    rtrain(minl) # try to polish the new model with basic rtrain to target loss on dataset


#batcher()
#mtrain(t_x,t_y[:,1:], ep=250, cycles=25, bs=15, coshot=True)
#rtrain(min=0.001)


# ex tf7 input architecture analysis (to evaluate training and accuracy )
# all models trained on rtrain(min=0.001) with 1e3 dataset and 30 nodes/layer
# echo3 = 3 subnets in layer, uses LSTM-LSTM-Dense-Dense layering
# uses minimal node net w30nodes/layer incapable of fully learning dataset
# use for tf9 transfer learning experiment
#
# model = load_model('modMAX1559676117-1e3-0.18.h5')
# no input LSTM echo 0.0018 loss max, poor gen ~1500ep
# model = load_model('modMAX1559679704-1e3-0.17.h5')
# no input LSTM echo3 0.0017 loss max,
# poor gen, identical endpoint to 1559676117, lower training time ~1100ep
# model = load_model('modMAX1559686229-1e3-0.19.h5')
# input LSTM-LSTM+input-LSTM+input echo3, 0.0019 loss max ~1500ep
# notable inaccuracy on v_x with sections loosing the pred line
# model = load_model('modMAX1559690146-1e3-0.22.h5')
# echo3, LSTM2+input layer hopping inputs 0.0022 loss max ~700ep
# trains fast but limited accuracy and predictive power

# echo3, echo3,echo2 experiment
# failure to converge, tested wDropout to no gain

# model = load_model('modMAX1559693980-1e3-0.21.h5') # modMAX1559708703-1e3-0.09-.h5
# echo3-lstm+flat/dense 0.0021 loss max ~700 ep
# also achived 0.0009 with 12500ep+ simulated overnight to show max
# 0.0009 demonstrates lower error on v_x

# echo3-lstm+flat/dense w200back can not sim error nan loss
# exploding gradient limit identified as +90back
# 90back does mot converge
# model = load_model('modMAX1559742989-1e3-70b-0.2.h5')
# 70back converges with same aprox loss, ~2000 ep
# very low generalisation
# model = load_model('modMAX1559752297-1e3-10b-0.2.h5')
# 5back model, faster to train, for this use case provides similar generalisation to 70back
# tf8 to try applying ResNet blocks to problem