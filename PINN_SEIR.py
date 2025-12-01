import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

### So far, framework for PINN to solve SEIR
### Need to input observed data (collocation points)
### tf.gradientTape to compute derivatives
### Compute total loss (data informed + physics informed)
### Initialise SEIR parameters
### Add full training loop

### Create PINN
def create_pinn_model():
    ### Input layer - time (shape = 1 because time is 1D)
    t_input = Input(shape=(1,), name='time_input')
    
    ### Hidden layer 1 = 32 neurons, tanh activation
    x = Dense(32, activation='tanh')(t_input)
    
    ### Hidden layers 2 + 3 = 64 neurons, tanh activation  
    x = Dense(64, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    
    ### Hidden layer 4 = 32 neurons, tanh activation       
    x = Dense(32, activation='tanh')(x)
    
    ### Output layers for S, E, I, R
    S = Dense(1, activation=None, name='S')(x)
    E = Dense(1, activation=None, name='E')(x)
    I = Dense(1, activation=None, name='I')(x)
    R = Dense(1, activation=None, name='R')(x)
    
    ### Create the model - inputs = time, outputs = SEIR compartments
    model = Model(inputs=t_input, outputs=[S, E, I, R])
    return model

### Physics informed loss function
@tf.function
def physics_loss(pinn_dynamics, t_collocation, beta, sigma, gamma):
    ### Get predictions and derivatives from PINN
    S, E, I, R, dS_dt, dE_dt, dI_dt, dR_dt = pinn_dynamics(t_collocation)
    
    ### SEIR equations (N = 1 in scaled form)
    N = 1.0
    dS_dt_physics = -beta * S * I / N
    dE_dt_physics = beta * S * I / N - sigma * E
    dI_dt_physics = sigma * E - gamma * I
    dR_dt_physics = gamma * I
    
    ### Physics loss (calculated by MSE) 
    S_loss = tf.reduce_mean(tf.square(dS_dt - dS_dt_physics))
    E_loss = tf.reduce_mean(tf.square(dE_dt - dE_dt_physics))
    I_loss = tf.reduce_mean(tf.square(dI_dt - dI_dt_physics))
    R_loss = tf.reduce_mean(tf.square(dR_dt - dR_dt_physics))
    
    ### Total physics-informed loss
    return S_loss + E_loss + I_loss + R_loss

### Define optimizer
optimizer = Adam(learning_rate=0.001)

### Training model
@tf.function
def train_step(pinn_model, total_loss, beta, sigma, gamma):
    with tf.GradientTape() as tape:
        loss = total_loss()
    
    ### Get gradients and perform optimization
    gradients = tape.gradient(loss, pinn_model.trainable_variables + [beta, sigma, gamma])
    optimizer.apply_gradients(zip(gradients, pinn_model.trainable_variables + [beta, sigma, gamma]))
    
    return loss


### Plotting
def plot_seir(t, S, E, I, R):
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population Fraction')
    plt.title('SEIR Model using PINNs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
