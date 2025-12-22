### PINN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


### Load preprocessed data (from COVID_Data.py script)
### These data are arrays
t_data = np.load("t_data_2020.npy")       ### time points 
I_data = np.load("I_data_2020.npy")       ### observed infections
t_col  = np.load("t_col.npy")        ### collocation points for physics loss

### Store the max time for scaling
t_max = t_data.max()

### Convert to TensorFlow tensors (so they can be used for model training)
### tensor = multi-dimensional list of numbers
t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
I_tensor = tf.convert_to_tensor(I_data, dtype=tf.float32)

def create_model():
    t_input = Input(shape=(1,), name='time_input')

    x = Dense(32, activation='tanh')(t_input)
    x = Dense(64, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    x = Dense(32, activation='tanh')(x)

    S = Dense(1, activation='sigmoid', name='S')(x)
    E = Dense(1, activation='sigmoid', name='E')(x)
    I_out = Dense(1, activation='sigmoid', name='I')(x)
    R = Dense(1, activation='sigmoid', name='R')(x)

    model = Model(inputs=t_input, outputs=[S, E, I_out, R])
    return model

NN = create_model()
NN.summary()

NN.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'S': 'mse',
        'E': 'mse',
        'I': 'mse',
        'R': 'mse'
    },
    metrics={
        'I': 'mae'  # track mean absolute error on infections
    }
)
