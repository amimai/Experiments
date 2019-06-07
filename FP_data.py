import FP_config as FPc

import fxcmpy

TOKEN = FPc.TOKEN
con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')

# instruments = con.get_instruments_for_candles()
inst = FPc.instrument

# m1,m5,m15,m30,H1,H2,H3,H4,H6,H8,D1,W1,M1.
peri = FPc.period

# ['bidclose', 'bidhigh', 'bidlow', 'askopen', 'askclose','askhigh', 'asklow', 'tickqty']
cols = FPc.colums

numb = FPc.Train_Data_size


def ticks(instrument=inst, columns=cols, period=peri, number=numb):
    r = con.get_candles(instrument, columns=columns, period=period, number=number)
    return r


def fetchData(number=numb):
    a = ticks(number=number)

    a["bid_hl"] = a["bidhigh"] - a["bidlow"]  # range
    a["bid_cl"] = a["bidclose"] - a["bidlow"]  # downforce
    a["bid_ho"] = a["bidhigh"] - a["bidopen"]  # upforce
    a["bid_co"] = a["bidclose"] - a["bidopen"]  # direction

    print('got data')
    return a


def neatData():
    a = fetchData()
    ma = 0.0
    mi = 0.0
    ma = a["bidhigh"].max()
    mi = a["bidlow"].min()
    n = mi - (ma-mi)
    a["bidopen"] = a["bidopen"] - n
    a["bidclose"] = a["bidclose"] - n
    a["bidhigh"] = a["bidhigh"] - n
    a["bidlow"] = a["bidlow"] - n
    print(n)
    return a