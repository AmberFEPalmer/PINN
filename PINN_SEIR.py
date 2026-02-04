import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import pandas as pd

### Main websites used 
### https://i-systems.github.io/tutorial/KSNVE/220525/01_PINN.html
### https://vitalitylearning.medium.com/solving-a-first-order-ode-with-physics-informed-neural-networks-22e385f09d35
### code from seminal paper https://github.com/maziarraissi/PINNs
### https://www.tensorflow.org/tutorials/customization/basics

### Load preprocessed data as arrays (from COVID_Data.py script)
t_data = np.load("data/t_data_2020.npy")     
### Store the max time for scaling
t_max = t_data.max()
I_data = np.load("data/I_data_2020.npy")    

### Collocation points are random therefore collocation points are only created in split_covid_data_by_month.py
### this works because time is normalised from 0-1 and the PINN is trained with this normalised time  
t_col  = np.load("data/t_col.npy")       

### Train/test split
N_obs = len(I_data)
t_data = t_data[:N_obs].reshape(-1, 1)
I_data = I_data.reshape(-1, 1)

### Generate training and testing data - takes first 80% of datasets
split = int(0.9 * N_obs) 

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
    E = Dense(1, activation=None, name='E')(x_seir)
    I = Dense(1, activation=None, name='I')(x_seir)
    R = Dense(1, activation=None, name='R')(x_seir)

    ### Time-varying beta 
    ### beta = softplus activation -> allows it to be greater than 1
    beta_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    beta_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_hidden)
    beta_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_hidden)

    beta = Dense(1, activation=None, name='beta')(beta_hidden) 

    model = Model(inputs=t_input, outputs=[S, E, I, R, beta])
    return model

model = create_pinn_model()
### Print model architecture
model.summary()

### Define initial conditions
S0 = tf.constant(0.9, dtype=tf.float32)
E0 = tf.constant(0.05, dtype=tf.float32)
I0 = tf.constant(0.05, dtype=tf.float32)
R0 = tf.constant(0.0, dtype=tf.float32)

### Define physics informed loss
def seir_ode_loss(t_col, t_data_loss, I_data_loss, net):
    
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
        S, E, I, R, beta = net(t_col)
       
    ### Define parameters which don't vary over time
    ### Following what was done in Qian et al. 2025
    sigma = tf.constant(0.3, dtype=tf.float32, name='sigma')
    gamma = tf.constant(0.3, dtype=tf.float32, name='gamma')   
     
    ### Compute derivatives e.g. dS/dt
    dS_dt = tape.gradient(S, t_col) 
    dE_dt = tape.gradient(E, t_col) 
    dI_dt = tape.gradient(I, t_col) 
    dR_dt = tape.gradient(R, t_col) 
    
    d_beta_dt = tape.gradient(beta, t_col)
    beta_smooth_loss = tf.reduce_mean(tf.square(d_beta_dt))
    
    del tape

    ### SEIR equations - these are in real time not normalised time
    T = tf.constant(t_max, dtype=tf.float32)
    dS_dt_true_scaled = T * (-beta * S * I)
    dE_dt_true_scaled = T * (beta * S * I - sigma * E)
    dI_dt_true_scaled = T * (sigma * E - gamma * I)
    dR_dt_true_scaled = T * (gamma * I)
 
    ### Physics-informed loss - mean squared error
    loss_S = tf.reduce_mean(tf.square(dS_dt - dS_dt_true_scaled))
    loss_E = tf.reduce_mean(tf.square(dE_dt - dE_dt_true_scaled))
    loss_I = tf.reduce_mean(tf.square(dI_dt - dI_dt_true_scaled))
    loss_R = tf.reduce_mean(tf.square(dR_dt - dR_dt_true_scaled))

    physics_loss = (
        1.0 * loss_S +
        1.0 * loss_E +
        1.0 * loss_I +
        1.0 * loss_R 
    )

    ### Initial condition loss (evaluate at t=0)
    t_zero = tf.constant([[0.0]], dtype=tf.float32) 
    S_0, E_0, I_0, R_0, _ = net(t_zero)
    
    IC_loss = tf.reduce_mean(
        tf.square(S_0 - S0_fixed) +
        tf.square(E_0 - E0_fixed) +
        tf.square(I_0 - I0_fixed) +
        tf.square(R_0 - R0_fixed) )
    
    ### Data loss 
    t_data_normalized = t_data_loss 
    _, _, I_pred, _, _ = net(t_data_normalized)
    data_loss = tf.reduce_mean(tf.square(I_pred - I_data_loss))
    
    ### Total loss
    ### the scale for physics loss is bigger than data loss which is why data loss needs to be much higher weighted
    ### (the derivatives are bigger numbers than the data)
    total_loss = 1.0 * data_loss + 0.0 * IC_loss + 0.001*physics_loss + 0* beta_smooth_loss
    
    return total_loss

S0_fixed = S0
E0_fixed = E0
I0_fixed = I0
R0_fixed = R0

### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
### learning rate scheduler added (not in original paper)
### https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=6000,
    decay_rate=0.95,
    staircase=False
)
optm = Adam(learning_rate=lr_schedule)

### Collocation points for physics loss
n_collocation = 30
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

### @tf.function increased TensorFlow speed 
### https://www.tensorflow.org/guide/function
@tf.function
def train_step(t_col, t_data, I_data):
    with tf.GradientTape() as tape:
        loss = seir_ode_loss(t_col, t_data, I_data, model)
    grads = tape.gradient(loss, model.trainable_variables)
    optm.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def test_step(t_col, t_data, I_data):
    return seir_ode_loss(t_col, t_data, I_data, model)

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
plt.title('SEIR PINN on real data: Training vs Forecasting')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('PINN_output.png')
plt.show()

### Plot actual infection counts vs PINN predictions rather than normalised
N = 55000000
### Convert tensors to numpy arrays
t_data_np  = to_numpy_flat(t_data)         
I_pred_np  = to_numpy_flat(I_pred) * N
I_data_np  = to_numpy_flat(I_data) * N         
t_train_np = to_numpy_flat(t_train)
I_train_np = to_numpy_flat(I_train) * N
t_test_np  = to_numpy_flat(t_test)
I_test_np  = to_numpy_flat(I_test) * N

plt.figure(figsize=(12, 6))
plt.plot(t_data_np, I_pred_np, 'b-', linewidth=2, label='PINN Predicted I')
plt.plot(t_train_np, I_train_np, 'r-', linewidth=2, label='I (Observed – train)')
plt.plot(t_test_np, I_test_np,'r-', linewidth=2, label='I (Observed – test)')
plt.axvline(x=t_train_np[-1],color='gray', linestyle='--', label='Train/Test Split')
plt.xlabel('Time (days or normalized units)')
plt.ylabel('Infected Individuals')
plt.title('PINN Prediction vs Actual Infection Counts')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('PINN_vs_actual_infections_counts.png')
plt.show()

### Plot beta over time
t_plot = np.linspace(0.0, 1.0, 500)
t_plot_tensor = tf.convert_to_tensor(t_plot.reshape(-1, 1), dtype=tf.float32)
_, _, _, _, beta = model.predict(t_plot_tensor)

plt.figure(figsize=(8,5))
plt.plot(t_plot, beta.flatten(), 'g-', linewidth=2)
plt.xlabel('Normalised time')
plt.ylabel('β(t)')
plt.ylim(0, 1)   # <--- force y-axis from 0 to 1
plt.grid(True)
plt.show()
plt.savefig('Beta_over_time.png', dpi=300)

### Model evaluation - mean absolute error
mae_test = tf.keras.losses.MeanAbsoluteError()(I_test, I_pred[split:]).numpy()
print("Mean Absolute Error:", mae_test)

### Model evaluation - mean sqaured error
mse_test = tf.keras.losses.MeanSquaredError()(I_test, I_pred[split:]).numpy()
print("Mean Squared Error:", mse_test)

### Model evaluation - mean absolute percentage error
mape_test = tf.keras.losses.MeanAbsolutePercentageError()(I_test, I_pred[split:]).numpy()
print("Mean Absolute Percentage Error:", mape_test)