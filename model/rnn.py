import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, Activation, LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
import matplotlib.pyplot as plt
from tensorflow import keras

nbf = 64 		# No. Conv Filters
flen = 16 		# Conv Filter length
nlstm = 100 	# No. LSTM layers
ndrop = 0.1     # LSTM layer dropout
nbatch = 32 	# Fit batch No.
nepochs = 500    # No. training rounds"same"

class air_rnn(keras.Model):
    def __init__(self):
        super().__init__()
        self.rnn = tf.keras.Sequential()
        self.dropout = tf.keras.layers.Dropout(0.8)
        self.dense1 = Dense(32, activation='relu')
        self.dense2 = Dense(1, activation='linear')
    def call(self, inputs):
        x = self.rnn(inputs)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

