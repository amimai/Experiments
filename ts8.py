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

data = read_csv('1e3data.csv', header=0, index_col=0)
print(data.head())

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


values = data
fitData(values,cols)
print(data.head())
for each in cols:
    print(each,' max ',max(data[each]),' min ',min(data[each]))

values = values.values

back = 10
fore = 5
len_v_x = int(.3*len(values))
x = []
y = []
for i in range(len(values)-back-1-fore):
    t=[]
    for j in range(0,back):
        t.append(values[[(i+j)],:])
    x.append(t)
    y.append((values[i+back+fore,1],values[i+back+fore-1,1],values[i+back+fore-2,1]))

x,y = np.array(x), np.array(y)
t_x,t_y = x,y
v_x,v_y = x[:len_v_x+back],y[:len_v_x+back]
t_x,v_x = t_x.reshape(t_x.shape[0],back,len(cols)),v_x.reshape(v_x.shape[0],back,len(cols))
#t_y,v_y = t_y.reshape(t_y.shape[0],1,1),v_y.reshape(v_y.shape[0],1,1)
print(t_x.shape)
print(t_y.shape)
print(v_x.shape)
print(v_y.shape)


model = Sequential()
model.add(LSTM(120,return_sequences=True,input_shape=(t_x.shape[1],t_x.shape[2])))
model.add(LSTM(120,return_sequences=True))
model.add(LSTM(120,return_sequences=True))
model.add(LSTM(120))
model.add(Dense(3))
model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')


model.fit(t_x,t_y,epochs=150,batch_size=1)
'''
for i in range(5):
    pred = model.predict(v_x[i].reshape(1,10,9))
    print(pred,' : ', v_y[i])
'''
print('eval : ',model.evaluate(v_x,v_y))
pred = model.predict(v_x)

npv = np.array(v_y)
npp = np.array(pred)
print(npv[0:5])
print(npp[0:5])
print('error')
print(npv[0:5]-npp[0:5])
plt.plot(npv[:,0], color='red')
plt.plot(npp[:,0],color='purple')
plt.plot(npp[:,1],color='blue')
plt.plot(npp[:,2],color='green')
plt.show()




