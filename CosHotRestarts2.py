import math
from keras.layers import *
from keras.callbacks import Callback

class CosHotRestart(Callback):

    def __init__(self, nb_epochs, nb_cycles=5, gain=1.2, verbose=0):
        if nb_cycles > nb_epochs:
            raise ValueError('nb_epochs has to be lower than nb_cycles.')

        super(CosHotRestart, self).__init__()
        self.verbose = verbose
        self.nb_epochs = nb_epochs
        self.nb_cycles = nb_cycles
        self.period = self.nb_epochs // self.nb_cycles
        self.nb_digits = len(str(self.nb_cycles))

        self.gain = gain
        self.prev_epoch = 0
        self.minloss = 9999.
        self.stored_weights = 0
        self.delta_p = 0
        self.LRhist = []

    def on_epoch_end(self, epoch, logs=None):
        if self.minloss > logs.get('loss'): # save best model for restarts
            print('newloss ', logs.get('loss'))
            self.stored_weights = self.model.get_weights()
            self.minloss = logs.get('loss')
            self.period += 1 # slow down decay if its working
            self.delta_p +=1
            self.LRhist.append(K.get_value(self.model.optimizer.lr))

        if epoch == 0 or (epoch + 1 - self.prev_epoch) % self.period != 0: return
        cycle = int(epoch / self.period)
        cycle_str = str(cycle).rjust(self.nb_digits, '0')
        print('cycle = %s' % cycle_str)

        self.period = int((self.period-self.delta_p) * self.gain)
        self.delta_p = 0
        self.prev_epoch = epoch
        print('period = %d' % self.period)

        #self.period = int(self.period * self.gain)
        #print('setting period to %d' % self.period)

        # Resetting the learning rate
        K.set_value(self.model.optimizer.lr, self.base_lr)

        # restore best weights achived in cycle
        self.model.set_weights(self.stored_weights)
        print('restoring best model')


    def on_epoch_begin(self, epoch, logs=None):
        if epoch <= 0: return

        lr = self.schedule(epoch)
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: Snapchot modifying learning '
                  'rate to %s.' % (epoch + 1, lr))

    def schedule(self, epoch):
        lr = math.pi * (epoch % self.period) / self.period
        lr = self.base_lr / 2 * (math.cos(lr) + 1)
        return lr

    def set_model(self, model):
        self.model = model
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get initial learning rate
        self.base_lr = float(K.get_value(self.model.optimizer.lr))

    def on_train_end(self, logs={}):
        # Set weights to the values from the end of the best cycle
        if len(self.LRhist)>0:
            K.set_value(self.model.optimizer.lr, 5*sum(self.LRhist)/len(self.LRhist))
        print(K.get_value(self.model.optimizer.lr))
        self.model.set_weights(self.stored_weights)
        self.minloss = 9999.
        self.stored_weights = 0
