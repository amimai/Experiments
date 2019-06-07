import FP_config as FPc

TIME_STEPS = FPc.TIME_STEPS
BATCH_SIZE = FPc.BATCH_SIZE

lstm_model = Sequential()
lstm_model.add(LSTM(100,
                    batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                    dropout=0.0, recurrent_dropout=0.0,
                    stateful=True,
                    kernel_initializer='random_uniform'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(20,activation='relu'))
lstm_model.add(Dense(1,activation='sigmoid'))
optimizer = optimizers.RMSprop(lr=lr)
lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)



csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'your_log_name' + '.log'), append=True)

history = model.fit(x_t, y_t, epochs=your_epochs, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
trim_dataset(y_val, BATCH_SIZE)), callbacks=[csv_logger])