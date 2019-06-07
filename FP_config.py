
TOKEN = '943bfa34a2e8644a5614629836220d6f87f2bbad'
#instrument = 'EUR/USD'
instrument = 'USD/JPY'
period = 'H1'
colums = ['bidopen', 'bidclose', 'bidhigh', 'bidlow', 'tickqty']
Train_Data_size = 10000
train_cols = ["bidopen", "bidclose", "bidhigh", "bidlow", "tickqty", "bid_hl", "bid_cl", "bid_ho", "bid_co"]
TIME_STEPS = 60
BATCH_SIZE = 5

OUTPUT_PATH = 'C:/Users/Sophie/Downloads/PyOut'