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
# '''
from keras.models import Sequential, load_model, Model, Input
from keras.layers import Dense, Dropout, Concatenate, Flatten, Activation, Add
from keras.layers import LSTM
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from keras import backend as K  # to set LR

# print(K.get_value(model.optimizer.lr))
# K.set_value(model.optimizer.lr, 0.001)

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

back = 5  # 40
fore = 4
x, y, t_x, t_y, v_x, v_y = getDataDFX('1e3data.csv', back, fore, 1)

nodes = 30
nodes2 = 30

LSTMin = Input(shape=(t_x.shape[1], t_x.shape[2]))

# temp flatten to dense for quick experiments
flat = Flatten()(LSTMin)
# network block of 2 dense nets max
'''
flat = Dense(nodes2,activation='relu')(flat)
flat = Dense(nodes2,activation='relu')(flat)
flat = Dense(nodes2,activation='relu')(flat)
'''

flat = Dense(nodes2, activation='relu')(flat)
flat = Dense(nodes2, activation='relu')(flat)  # convergence of 3 layer net hampered by 2x dense

'''converted to [for i] system for testing
bypass_2 = flat
D2 = Dense(nodes2,activation='relu')(bypass_2)
D2 = Dense(nodes2,activation='relu')(D2)
merge1 = Add()([D2,bypass_2])
merge1 = Activation('relu')(merge1)

bypass_1 = merge1
D1 = Dense(nodes2,activation='relu')(bypass_1)
D1 = Dense(nodes2,activation='relu')(D1)
merge = Add()([D1,bypass_1])
merge = Activation('relu')(merge)
# adding activation smoothed learning loss but does not improve loss, test with multiple layers
'''

''' # double bypas i=16,n=5 175dense net trainable
BigBypass = flat
for n in range(5): # double bypass
    D2 = Dense(nodes2, activation='relu')(BigBypass)
    bypass = D2
    for i in range(16): #single bypass
        D1 = Dense(nodes2, activation='relu')(bypass)
        # D1 = Activation('relu')(D1)
        D1 = Dense(nodes2, activation='relu')(D1)
        merge = Add()([D1, bypass])
        merge = Activation('relu')(merge)
        bypass = merge
    merge = Dense(nodes2, activation='relu')(bypass)
    merge = Add()([merge, BigBypass])
    BigBypass = merge
'''

'''
bypass = flat # clasic ResNet
for i in range(16): #single bypass
    D1 = Dense(nodes2, activation='relu')(bypass)
    # D1 = Activation('relu')(D1)
    D1 = Dense(nodes2, activation='relu')(D1)
    merge = Add()([D1, bypass])
    merge = Activation('relu')(merge)
    bypass = merge
'''

'''
# implementation of https://arxiv.org/pdf/1608.06993.pdf
agregate = [flat]
adds = flat # for first element
for i in range(48):
    D1 = Dense(nodes2, activation='relu')(adds)
    # D1 = Activation('relu')(D1)
    D1 = Dense(nodes2, activation='relu')(D1)
    agregate.append(D1) # add to cumulative list
    adds = Add()(agregate)  # add cumulative list for bypass
    adds = Activation('relu')(adds)

merge = adds
'''

'''
# Double bypass deep cross net
BigBypass = flat
for n in range(6):
    D2 = Dense(nodes2, activation='relu')(BigBypass)
    agregate = [flat]
    adds = flat # for first element
    for i in range(16):
        D1 = Dense(nodes2, activation='relu')(adds)
        D1 = Dense(nodes2, activation='relu')(D1)
        agregate.append(D1) # add to cumulative list
        adds = Add()(agregate)  # add cumulative list for bypass
        adds = Activation('relu')(adds)
    merge = Dense(nodes2, activation='relu')(adds)
    merge = Add()([merge, BigBypass])
    BigBypass = merge
'''

'''
# Double Deep Cross Net
Bigagregate = [flat]
Bigadds = flat
for n in range(6):
    D2 = Dense(nodes2, activation='relu')(Bigadds)

    agregate = [D2]
    adds = D2 # for first element
    for i in range(16):
        D1 = Dense(nodes2, activation='relu')(adds)
        D1 = Dense(nodes2, activation='relu')(D1)
        agregate.append(D1) # add to cumulative list
        adds = Add()(agregate)  # add cumulative list for bypass
        adds = Activation('relu')(adds)

    D2 = Dense(nodes2, activation='relu')(adds)
    Bigagregate.append(D2)
    Bigadds = Add()(Bigagregate)
    Bigadds = Activation('relu')(Bigadds)
'''

# Nested Bypass Method
def nestedBypass(input,nests,eggs):
    BigAg = [input]
    BigAdd = input
    for i in range(eggs):
        D2 = Dense(nodes2, activation='relu')(BigAdd)
        if nests > 0:
            Egg = nestedBypass(D2,nests-1,eggs)
        else:
            Egg = D2
        D2 = Dense(nodes2, activation='relu')(Egg)
        BigAg.append(D2)
        BigAdd = Add()(BigAg)
        BigAdd = Activation('relu')(BigAdd)
    return BigAdd

merge = nestedBypass(flat,3,3)


# amplifier layer (ex. tf6)
# uses multiple separate Dropout-Dense subnets to allow back propagation to be amplified
# demonstrable increase from 6->8 maximum trainable LSTM_merge layers
# with increase merg(LSTM layers) from 0(no subnet) to 1(2 subnets)
# to 2(8 subnets, unreliable without boosting via callbacks)
# uses principle proposed in [paper?] as each subnet acts as a concurrent micro batch
# diminishing return tested with up to 60 nets without significant improvements over 8
# kernel_constraint=maxnorm(3) is not beneficial unlike original dropout regularisation paper proposal
DRr = .4  # .4-.6 dropout ideal
amp = []
for i in range(2):
    amp_n = Dropout(rate=DRr)(merge)
    amp_n = Dense(nodes2, activation='relu', kernel_initializer='normal', kernel_constraint=maxnorm(3))(amp_n)
    amp.append(amp_n)

activation = Concatenate()(amp)
activation = Dense(t_y.shape[1] - 1, activation='linear')(activation)

model = Model(inputs=LSTMin, outputs=activation)

model.summary()
# (ex. ts8)modified adam hyperperamaters based on [paper?] reliable with lr=7e-6 on 2m+ params
optimizer = optimizers.adam(lr=1e-2, beta_2=.98, epsilon=1e-5)  # , clipvalue=0.5) #.sgd(lr=.1, momentum=0.9) #)#
model.compile(optimizer=optimizer, loss='mean_squared_error')


# model.fit(t_x, t_y[:,1:],epochs=500,batch_size=15, verbose=2)

def results(v_x, v_y, save=0):
    eval = model.evaluate(v_x, v_y[:, 1:])
    print('eval : ', eval)
    pred = model.predict(v_x)

    npv = np.array(v_y)
    npp = np.array(pred)
    print(npv[0:5])
    print(npp[0:5])
    print('error')
    print(npv[0:5, 1:] - npp[0:5])
    plt.figure(figsize=(16, 8), frameon=False)
    plt.plot(npv[:, 0], color='black')
    # plt.plot(range(fore, fore + len(npp)), npv[:,0]+npp[:, 0])
    for i in range(npp.shape[1]):
        plt.plot(range(i + 1, i + 1 + len(npp)), npv[:, 0] + npp[:, i])
    if save == 1:
        plt.savefig('validation-time{1}-eval{0}.png'.format(eval, round(time.time())))
    else:
        plt.show()


# (ex. tf2-4) CosHot and SGDR with automatic hot restarts
# cosine based learning rate annealing algorithms
# [paper https://arxiv.org/abs/1608.03983]
# CosHot reliable annealing algorithm with minimal backslide
# SGDR heavily modified batchwise annealing algorithm that generates erratic
# SGDR can backslide easily on large datasets
# achieves remarkable learning and generalisation statistics when successful and works well on smaller datasets
# merged with hot restarts in (ex. ts4)
def mtrain(t_x, t_y, ep=250, cycles=25, bs=15, coshot=True):
    if coshot:
        CosHot = CosHotRestart(nb_epochs=ep, nb_cycles=cycles, gain=1.1, verbose=0)
        model.fit(t_x, t_y, epochs=ep, verbose=2, batch_size=bs, callbacks=[CosHot])  # )#
    else:
        SGDR = SGDRScheduler(min_lr=1e-6, max_lr=1e-2,
                             steps_per_epoch=np.ceil(len(t_x) / 15),
                             lr_decay=0.9, cycle_length=5, mult_factor=1.2)
        model.fit(t_x, t_y, epochs=ep, verbose=2, batch_size=bs, callbacks=[SGDR])


# (ex. tf1-tf4) rtrain recursive training manager
# automates model training and swaps between training callback modes
def rtrain(min=0.001):
    mag = model.evaluate(t_x, t_y[:, 1:], verbose=0)
    var = True
    while mag > 0.001:
        mtrain(t_x, t_y[:, 1:], coshot=var)
        var = not var  # use erratic coshot to bounce moddel over humps and greedy SGDR to smoothe to lower loss
        mag = model.evaluate(t_x, t_y[:, 1:], verbose=0)
        n = 'modMAX{0}.h5'.format(round(time.time()))
        print('saved ', n, ' with eval ', mag)
        model.save(n)


# (ex. tf5) progressive training set growth for training on large datasets
# useful for rapidly training on 1e4+ datasets with high randomness and complexity
# uses principle of knowledge transfer to maintain low loss while geometrical increasing dataset
# minl(target loss) 0.003 provides beter generalisation that 0.001 but chance of backslide still exists
# bl(random string of data) for sequential data, reduces risk of backsliding but also effects generalisation
def batcher(minl=0.003, minb=600, bl=100, gain=1.1):
    lx = len(t_x)
    if lx < minb:
        rtrain(minl)
    else:
        a = randrange(0, lx - bl)
        n_x = t_x[a:a + bl]
        n_y = t_y[a:a + bl]
        print(t_x.shape, n_x.shape)
        tlen = int(minb / 2)
        while len(n_x) < lx / (gain * 1.1):
            while len(n_x) < tlen:
                a = randrange(0, lx - bl)
                n_x = np.concatenate((n_x, t_x[a:a + bl]), axis=0)
                n_y = np.concatenate((n_y, t_y[a:a + bl]), axis=0)
            print(t_x.shape, n_x.shape)
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
    rtrain(minl)  # try to polish the new model with basic rtrain to target loss on dataset


# batcher()
#mtrain(t_x, t_y[:, 1:], ep=250, cycles=25, bs=15, coshot=True)
# rtrain(min=0.001)

# ex tf8 ResNet block style residual network
# testing done with 200 ep adam on 1e3 dataset for consistent analysis
# .0112 loss is failure to converge for 1e3 dataset
# 2 Dense flatten converged 0.0099 loss
# 3 Dense flatten failed to converge
# with 2 Dense [bypass,2dense] converged to .0097 loss
# with 2 dense,[bypass,2dense],[bypass,2dense] achieved loss .0097 with activation
# with 2 dense,[bypass,2dense],[bypass,2dense] achieved loss .0099 without activation
# 8 layers total trained on adam successfully before LR annealing required
# using for i residual net builder CosHot method used to anneal LR and converge nets in 250ep
# range(3) .0097
# range(4) .0081
# range(5) .0095
# range(8) .0089
# range(16) .0101
# range(32) .0092 model is shaky in training
# range(64) .0094 very shaky and gradient explosion symptoms early on
# range(128) nan gradient explosion retry with clipping 261Dense net
# theoretically trainable with optimizers.adam(lr=1e-7,beta_2=.98,epsilon=1e-5, clipvalue=0.2)
# achieved loss 8.6084 after 900 ep using adam->CosHot
# implementation tests of BigBypass method with i=32, n=2 137dense model converges
# i=32, n=2 (137dense) .0096 approaching exploding grad
# i=16, n=4 (141dense) .0090 approaching exploding grad
# model = load_model('modMAX1559762420.h5') # for future training
# did not exceed range(64)
# implementation of https://arxiv.org/pdf/1608.06993.pdf
# performance on i = 16 0.0084, trains faster and performs better
# i=48 101dense net 0.0094 near max for architecture with current LR
# Double bypass deep cross net method implemented
# tested i=16 n=3 (dense_107) 0.0084, exceptional convergence performance and train speed
# model = load_model('modMAX1559765612.h5') # for future training
# i=16 n=6 (dense_209) converges with issues is approaching gradient explosion with current LR
# model = load_model('modMAX1559767513.h5') # for future training use 500ep CosHot
# Double Deep Cross Net method developed
# smoother convergence on i=16 n=6 (dense_209) 0.0071 with 500ep CosHot
# model = load_model('modMAX1559769528.h5')
# Nested Bypass method developed
# nestedBypass(flat,3,3) dense_245 0.0082 reliably slowly trainable super deep net
# model = load_model('modMAX1559770750.h5')



# ex tf7 input architecture analysis (to evaluate training and accuracy )
#############################################################################
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
###############################################################################
