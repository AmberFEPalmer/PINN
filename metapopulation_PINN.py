import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import pandas as pd

data = pd.read_csv("metapopulation_results.csv")   
t_data = data["time"].values.reshape(-1, 1)
I_data = data["I"].values.reshape(-1, 1)    

### Train/test split
N_obs = len(I_data)
t_data = t_data[:N_obs].reshape(-1, 1)
I_data = I_data.reshape(-1, 1)

### Generate training and testing data - takes first 80% of datasets
split = int(0.8 * N_obs) 

t_train = t_data[:split] ### take all elements from 0 up to "split"
I_train = I_data[:split]

t_test  = t_data[split:] ### take all elements from "split" to the end
I_test  = I_data[split:]

### Convert to tensors
t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
I_train_tensor = tf.convert_to_tensor(I_train, dtype=tf.float32)

### Define PINN
### L2 regularisation for hidden layers 
### https://keras.io/api/layers/regularizers/
### https://developers.google.com/machine-learning/crash-course/overfitting/regularization
def create_pinn_model():
    ### Input layer = time 
    t_input = Input(shape=(1,), name='time_input')

    ### 3 Hidden layers, 50 neurons each , tanh activation (tanh = non-linear + smooth)
    ### 3 hidden layers with 50 neurons to match Qian et al. 2025
    x_seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    x_seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(x_seir)
    x_seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(x_seir)

    ### SEIR outputs 
    S = Dense(1, activation=None, name='S')(x_seir)
    I1 = Dense(1, activation=None, name='I1')(x_seir)
    I2 = Dense(1, activation=None, name='I2')(x_seir)
    I3 = Dense(1, activation=None, name='I3')(x_seir)
    R = Dense(1, activation=None, name='R')(x_seir)

    ### Time-varying beta 
    ### beta = softplus activation -> allows it to be greater than 1
    beta_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    beta_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_hidden)

    beta = Dense(1, activation=None, name='beta')(beta_hidden) 

    model = Model(inputs=t_input, outputs=[S, I1, I2, I3, R])
    return model

model = create_pinn_model()
### Print model architecture
model.summary()

### Define initial conditions
S0 = tf.constant(0.999, dtype=tf.float32)
I0 = tf.constant([0.001, 0.0, 0.0], dtype=tf.float32)
R0 = tf.constant(0.0, dtype=tf.float32)

### Define physics informed loss
def metapopulation_ode_loss(t_col, net, lam, gamma_list, S0_fixed, I0_fixed, R0_fixed, t_data_loss, I_data_loss):

    ### if t_col is a 1D array it is reshaped to a column vector
    if len(t_col.shape) == 1:t_col = tf.reshape(t_col, (-1, 1))
    
    ### Convert data to tensors 
    if not isinstance(t_data_loss, tf.Tensor):t_data_loss = tf.convert_to_tensor(t_data_loss, dtype=tf.float32)
    if not isinstance(I_data_loss, tf.Tensor):I_data_loss = tf.convert_to_tensor(I_data_loss, dtype=tf.float32)

    ### if t_data_loss is a 1D array it is reshaped to a column vector
    if len(t_data_loss.shape) == 1:t_data_loss = tf.reshape(t_data_loss, (-1, 1))
    
    ### if I_data_loss is a 1D array it is reshaped to a column vector
    if len(I_data_loss.shape) == 1:I_data_loss = tf.reshape(I_data_loss, (-1, 1))
    
    ### Physics loss at collocation points
    ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, I1, I2, I3, R = net(t_col)
        
    ### Compute derivatives e.g. dS/dt
    dS_dt = tape.gradient(S, t_col) 
    dI1_dt = tape.gradient(I1, t_col) 
    dI2_dt = tape.gradient(I2, t_col) 
    dI3_dt = tape.gradient(I3, t_col) 
    dR_dt = tape.gradient(R, t_col) 
    del tape

    ### SEIR equations - these are in real time not normalised time
    dS_dt_true = -lam * S
    dI1_dt_true = lam * S - gamma_list[0] * I1
    dI2_dt_true = gamma_list[0] * I1 - gamma_list[1] * I2
    dI3_dt_true = gamma_list[1] * I2 - gamma_list[2] * I3
    dR_dt_true = gamma_list[2] * I3
    
    ### divide the gradients by T 
    ### This ensures physics loss is on the same scale as data loss
    days = 50.0
    T = tf.constant(days, dtype=tf.float32)
    
    dS_dt_normalised = dS_dt / T
    dI1_dt_normalised = dI1_dt / T
    dI2_dt_normalised = dI2_dt / T
    dI3_dt_normalised = dI3_dt / T
    dR_dt_normalised = dR_dt / T
    
    ### Physics-informed loss - mean squared error
    loss_S = tf.reduce_mean(tf.square(dS_dt_normalised - dS_dt_true))
    loss_I1 = tf.reduce_mean(tf.square(dI1_dt_normalised - dI1_dt_true))
    loss_I2 = tf.reduce_mean(tf.square(dI2_dt_normalised - dI2_dt_true))
    loss_I3 = tf.reduce_mean(tf.square(dI3_dt_normalised - dI3_dt_true))
    loss_R = tf.reduce_mean(tf.square(dR_dt_normalised - dR_dt_true))

    physics_loss = loss_S + loss_I1 + loss_I2 + loss_I3 + loss_R

    ### Initial condition loss (evaluate at t=0)
    t_zero = tf.constant([[0.0]], dtype=tf.float32) 
    S_0, I1_0, I2_0, I3_0, R_0 = net(t_zero)
    
    IC_loss = tf.reduce_mean(
        tf.square(S_0 - S0_fixed) +
        tf.square(I1_0 - I0_fixed[0]) +
        tf.square(I2_0 - I0_fixed[1]) +
        tf.square(I3_0 - I0_fixed[2]) +
        tf.square(R_0 - R0_fixed)
)

    ### constrain SEIR equations to equal 1
    S, I1, I2, I3, R = net(t_col)
    conservation_loss = tf.reduce_mean(tf.square(S + I1 + I2 + I3 + R - 1.0))
    
    ### Data loss 
    S_pred, I1_pred, I2_pred, I3_pred, R_pred = net(t_data_loss)
    I_pred = I1_pred + I2_pred + I3_pred
    data_loss = tf.reduce_mean(tf.square(I_pred - I_data_loss))

    ### Total loss
    ### the scale for physics loss is bigger than data loss which is why data loss needs to be much higher weighted
    ### (the derivatives are bigger numbers than the data)
    total_loss = 1.0 * data_loss + 1.0 * IC_loss +0.1*physics_loss + 1.0*conservation_loss
    
    return total_loss

### Define parameters which don't vary over time
lam = tf.constant(0.1, dtype=tf.float32, name='lam_raw')
gamma_list = [0.2, 0.2, 0.2]

S0_fixed = S0
I0_fixed = I0
R0_fixed = R0

### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
### learning rate scheduler added (not in original paper)
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=6000,
    decay_rate=0.95,
    staircase=False
)
optm = Adam(learning_rate=lr_schedule)

### Collocation points for physics loss
n_collocation = 50
t_col_uniform = np.linspace(0, 1, n_collocation).reshape(-1, 1)
t_col_tensor = tf.convert_to_tensor(t_col_uniform, dtype=tf.float32)

### ensure all inputs are float32 for training
t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)
I_train = tf.convert_to_tensor(I_train, dtype=tf.float32)
t_test = tf.convert_to_tensor(t_test, dtype=tf.float32)
I_test = tf.convert_to_tensor(I_test, dtype=tf.float32)

### Training loop
train_loss_record = []
test_loss_record = []  

trainable_vars = model.trainable_variables 

@tf.function
def train_step(t_col, t_data, I_data):
    with tf.GradientTape() as tape:
        loss = metapopulation_ode_loss(
            t_col, model, lam, gamma_list, S0_fixed, I0_fixed, R0_fixed,
            t_data, I_data
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optm.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def test_step(t_col, t_data, I_data):
    return metapopulation_ode_loss(t_col, model, lam, gamma_list, S0_fixed, I0_fixed, R0_fixed, t_data, I_data)

print("Starting training...")
for itr in range(500000):
    train_loss = train_step(t_col_tensor, t_train, I_train)
    train_loss_record.append(float(train_loss))

    if itr % 1000 == 0:
        test_loss = test_step(t_col_tensor, t_test, I_test)
        test_loss_record.append(float(test_loss))

    if itr % 10000 == 0:
        print(
            f"Iteration {itr}, "
            f"Train Loss: {float(train_loss):.6f}, "
            f"Test Loss: {float(test_loss):.6f}"
            )

### Plot training loss
t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
_, _, I_pred, _, _ = model(t_tensor)
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.yscale('log')  
plt.grid(True)
plt.savefig('PINN_training_loss.png') ### savefig has to be before show
plt.show()

### Make sure data shapes are compatible with matplotlib
def to_numpy_flat(arr):
    if hasattr(arr, 'numpy'):  
        return arr.numpy().flatten()
    else:  
        return arr.flatten()

t_data_np  = to_numpy_flat(t_data)         
I_pred_np  = to_numpy_flat(I_pred)
t_train_np = to_numpy_flat(t_train)
I_train_np = to_numpy_flat(I_train)
t_test_np  = to_numpy_flat(t_test)
I_test_np  = to_numpy_flat(I_test)

### Plot PINN training and forecasting
plt.figure(figsize=(14, 6))
plt.plot(t_data_np, I_pred_np,'b-', linewidth=2, label='I (PINN prediction)')
plt.plot(t_train_np, I_train_np,'r-', linewidth=2, label='I (observed – train)')
plt.plot(t_test_np, I_test_np,'r-', linewidth=2, label='I (observed – test)')
plt.axvline(x=t_train_np[-1],color='gray', linestyle='--', label='Train/Test Split')

plt.xlabel('Normalized Time')
plt.ylabel('Infected (normalized)')
plt.title('SEIR PINN on simulated data: Training vs Forecasting')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('PINN_output.png')
plt.show()

### Model evaluation - mean absolute error
mae_test = tf.keras.losses.MeanAbsoluteError()(I_test, I_pred[split:]).numpy()
print("Mean Absolute Error:", mae_test)

### Model evaluation - mean sqaured error
mse_test = tf.keras.losses.MeanSquaredError()(I_test, I_pred[split:]).numpy()
print("Mean Squared Error:", mse_test)

### Model evaluation - mean absolute percentage error
mape_test = tf.keras.losses.MeanAbsolutePercentageError()(I_test, I_pred[split:]).numpy()
print("Mean Absolute Percentage Error:", mape_test)