import numpy as np
import time
import pandas as pd

'''
import os
import sys
import pickle
import logging
import pickle
#'''
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
#'''

leng=5
transamples=5
data = [[i+j for j in range(leng)] for i in range(transamples)]
data = np.array(data, dtype=np.float32)
target = [[i+j+1 for j in range(leng)] for i in range(1,transamples+1)]
target = np.array(data, dtype=np.float32)

for i in range(2):
    print('x : ',data[i])
    print('y : ',target[i])

data = data.reshape(transamples,1,leng)
target = target.reshape(transamples,1,leng)

print('transformed')
for i in range(2):
    print('x : ',data[i])
    print('y : ',target[i])

def g_data(amm):
    x = np.zeros((amm,1))
    for i in range(amm):
        n = 0
        m=1
        p=1
        '''
        if i > 0:
            n = x[i-1][4]+i
            p = x[i-1][2]*2
        if i>1:
            m=  x[i-1][3]+x[i-2][3]
            # '''
        x[i]=[i] #norm,norm,geometric,fibinacci,exponential
    return x

norm=1000
da1 = g_data(1000)
da1/=norm*2
for i in range(3):
    print(da1[i])

def datasets(data, back,fore,target,mode):
    r1 = len(data)-(back+fore)
    x = np.zeros((r1,len(data[0]),back))
    if mode==1:
        y = np.zeros((r1, 2))
        # gen data
        for i in range(r1):
            tx = np.array((data[i:i + back])[::-1]).reshape(1,back)
            dx = 0
            x[i] = tx
            # determine target
            cur = data[i+back-1][target]
            next = data[i + back-1+fore][target]

            y[i] = [cur,next]
    #x.reshape(r1,1,back)
    return x,y


back=3
fore=1
target=0
mode=1
batch=10
cols=2
t_x, t_y= datasets(da1[0:400], back, fore, target, mode)
v_x, v_y= datasets(da1[400:500], back,fore, target, mode)
print(t_x.shape)
print(t_y.shape)

for i in range(2):
    print('x : ',v_x[i])
    print('y : ',v_y[i])


#'''
lstm_1=2
dense_layers=1
out_layers=1
epochs=200
model = Sequential()
for i in range(lstm_1):
    model.add(LSTM(32,input_shape=(1,back),
                   activation='sigmoid',return_sequences=True,
                   dropout=0.6, kernel_initializer='random_uniform'
                   ))
model.add(LSTM(32,#input_shape=(back,5),
               activation='sigmoid',
               dropout=0.6, kernel_initializer='random_uniform'
               ))
for i in range(dense_layers):
    model.add(Dense(20,
                    activation='relu'))
for i in range(out_layers):
    model.add(Dense(2, activation='sigmoid'))

optimizer = optimizers.adam(lr=1e-2) #RMSprop(lr=1e-2)
model.compile(loss='mean_squared_error'
              ,optimizer=optimizer #'adam'
              ,metrics=['accuracy'])

model.fit(t_x,t_y,epochs=epochs
          ,batch_size=batch
          ,validation_data=(v_x,v_y))

print("accuracy = ", model.evaluate(v_x, v_y))
pred = model.predict(v_x[0:10])
pred *= norm
v_yn =v_y*norm
v_xn =v_x*norm
for i in range (10):
    print(v_xn[i])
    print ("testset = ",pred[i], ":::", v_yn[i]," = real")




#'''