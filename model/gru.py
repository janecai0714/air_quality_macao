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
class air_gru(keras.Model):
    def __init__(self):
        super().__init__()
        self.gru1 = GRU(units=32, return_sequences=True, input_shape=(nstep, input_size))
        self.dropout1 = Dropout(0.2)

        self.gru2 = GRU(units=32, return_sequences=True)
        self.dropout2 = Dropout(0.2)

        self.gru3 = GRU(units=32, return_sequences=True)
        self.dropout3 = Dropout(0.2)

        self.dense1 = Dense(1, activation='linear')
    def call(self, inputs):
        x = self.gru1(inputs)
        x = self.dropout1(x)
        x = self.gru2(x)
        x = self.dropout2(x)
        x = self.gru3(x)
        x = self.dropout3(x)
        x = self.dense1(x)
        return x