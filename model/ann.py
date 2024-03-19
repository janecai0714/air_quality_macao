import pandas as pd
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, Activation, LSTM, GRU
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from tensorflow import keras

nbf = 64 		# No. Conv Filters
flen = 16 		# Conv Filter length
nlstm = 100 	# No. LSTM layers
ndrop = 0.2     # LSTM layer dropout
nbatch = 32 	# Fit batch No.
nepochs = 500
nstep  = 35
input_size = 1# No. training rounds"same"
class air_ann(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu', input_shape=(35, ))
        self.dense2= tf.keras.layers.Dense(64, activation= 'relu')
        self.dense3 = tf.keras.layers.Dense(32, activation= 'relu')
        self.dropout1 = Dropout(0.5)

        self.dense4 = Dense(1, activation='linear')
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout1(x)
        x = self.dense4(x)
        return x