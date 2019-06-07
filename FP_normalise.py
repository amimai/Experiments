from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import FP_timeseries as fpt
import numpy as np

import FP_config as FPc

print("checking if any null values are present\n", df_ge.isna().sum())

train_cols = FPc.train_cols
df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])

BATCH_SIZE = FPc.BATCH_SIZE


x_t, y_t = fpt.build_timeseries(x_train, 1)
x_t = fpt.trim_dataset(x_t, BATCH_SIZE)
y_t = fpt.trim_dataset(y_t, BATCH_SIZE)
x_temp, y_temp = fpt.build_timeseries(x_test, 1)
x_val, x_test_t = np.split(fpt.trim_dataset(x_temp, BATCH_SIZE),2)
y_val, y_test_t = np.split(fpt.trim_dataset(y_temp, BATCH_SIZE),2)