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
t_data = np.load("../../data/t_data_2021.npy")     
### Store the max time for scaling
t_max = t_data.max()
I_data = np.load("../../data/I_data_2021.npy")    

### Collocation points are random therefore collocation points are only created in split_covid_data_by_month.py
### this works because time is normalised from 0-1 and the PINN is trained with this normalised time  
t_col  = np.load("../../data/t_col.npy")       

# Add checks for normalization
print(f"I_data shape: {I_data.shape}")
print(f"I_data min: {I_data.min():.6f}, max: {I_data.max():.6f}")
print(f"I_data mean: {I_data.mean():.6f}, std: {I_data.std():.6f}")
if I_data.min() < 0 or I_data.max() > 1:
    print("WARNING: I_data is not in [0, 1] range. It may need normalization.")
else:
    print("I_data appears normalized (in [0, 1]).")

print(f"t_data min: {t_data.min():.6f}, max: {t_data.max():.6f}")

plt.figure()
plt.plot(t_data.flatten(), I_data.flatten())
plt.title("Raw I_data vs Time")
plt.xlabel("Normalized Time")
plt.ylabel("I (fraction)")
plt.show()

### Train/test split
N_obs = len(I_data)
t_data = t_data[:N_obs].reshape(-1, 1)
I_data = I_data.reshape(-1, 1)
### Generate training and testing data 
split = int(0.9 * N_obs) 

t_train = t_data[:split]
I_train = I_data[:split]
t_test = t_data[split:]
I_test = I_data[split:]

### Convert to tensors
t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
I_train_tensor = tf.convert_to_tensor(I_train, dtype=tf.float32)
t_test_tensor = tf.convert_to_tensor(t_test, dtype=tf.float32)
I_test_tensor = tf.convert_to_tensor(I_test, dtype=tf.float32)

### Define PINN
### L2 regularisation for hidden layers 
### https://keras.io/api/layers/regularizers/
### https://developers.google.com/machine-learning/crash-course/overfitting/regularization
def create_pinn_model():
    ### Input layer = time 
    t_input = Input(shape=(1,), name='time_input')

    ### 3 Hidden layers, 50 neurons each , tanh activation (tanh = non-linear + smooth)
    ### 3 hidden layers with 50 neurons to match Qian et al. 2025
    seir = Dense(100, activation='tanh')(t_input)
    seir = Dense(100, activation='tanh')(seir)
    seir = Dense(100, activation='tanh')(seir)
    
    ### SEIR outputs 
    ### 4 output layers
    ### No activation function
    S = Dense(1, activation='softplus', name='S')(seir)
    E = Dense(1, activation='softplus', name='E')(seir)
    I = Dense(1, activation='softplus', name='I')(seir)
    R = Dense(1, activation='softplus', name='R')(seir)

    ### Time-varying beta 
    ### beta = softplus activation -> allows it to be greater than 1
    ### 3 hidden layers, 50 neurons, tanh activation
    beta_hidden = Dense(100, activation = 'tanh')(t_input)
    beta_hidden = Dense(100, activation = 'tanh')(beta_hidden)
    beta_hidden = Dense(100, activation = 'tanh')(beta_hidden)
    
    ### Beta outputs
    ### One output layer
    ### No activation function
    beta = Dense(1, activation="softplus", name='beta')(beta_hidden)

    model = Model(inputs=t_input, outputs=[S, E, I, R, beta])
    return model


model = create_pinn_model()
### Print model architecture
model.summary()

### Define initial conditions
I0 = I_data[0]
E0 = 2 * I0
S0 = 1 - E0 - I0
R0 = tf.constant(0.0, dtype=tf.float32)

### Define physics informed loss
def physics_loss(t_col, t_data_loss, I_data_loss, net):
    
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
    dS_dt_physics = T * (-beta * S * I)
    dE_dt_physics = T * (beta * S * I - sigma * E)
    dI_dt_physics = T * (sigma * E - gamma * I)
    dR_dt_physics = T * (gamma * I)
 
    ### Physics-informed loss - mean squared error
    loss_S = tf.reduce_mean(tf.square(dS_dt - dS_dt_physics))
    loss_E = tf.reduce_mean(tf.square(dE_dt - dE_dt_physics))
    loss_I = tf.reduce_mean(tf.square(dI_dt - dI_dt_physics))
    loss_R = tf.reduce_mean(tf.square(dR_dt - dR_dt_physics))

    physics_loss = (
        1.0 * loss_S +
        1.0 * loss_E +
        1.0 * loss_I +
        1.0 * loss_R 
    )

    ### Initial condition loss (evaluate at t=0)
    t_zero = tf.constant([[0.0]], dtype=tf.float32) 
    S_0, E_0, I_0, R_0, _ = net(t_zero)
    
    Initial_condition_loss = tf.reduce_mean(
    tf.square(S_0 - S0) +
    tf.square(E_0 - E0) +
    tf.square(I_0 - I0) +
    tf.square(R_0 - R0)
)
    
    ## constrain SEIR equations to equal 1
    S, E, I, R, beta = net(t_col)
    conservation_loss = tf.reduce_mean(tf.square(S + E + I + R - 1.0))
    
    ### Data loss 
    t_data_normalized = t_data_loss 
    _, _, I_pred, _, _ = net(t_data_normalized)
    data_loss = tf.reduce_mean(tf.square((I_pred - I_data_loss)))
    
    ### Total loss
    ### the scale for physics loss is bigger than data loss which is why data loss needs to be much higher weighted
    total_loss =  1.0*data_loss 
    return total_loss, {
        "data_loss": data_loss,
        "IC_loss": Initial_condition_loss,
        "conservation_loss": conservation_loss,
        "ODE_loss": physics_loss,
        "beta_smooth_loss": beta_smooth_loss,
    }

S0_fixed = S0
E0_fixed = E0
I0_fixed = I0
R0_fixed = R0

### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
### learning rate scheduler added (not in original paper)
### https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/
initial_lr = 0.01
optm = Adam(learning_rate=initial_lr)

### Collocation points for physics loss
n_collocation = 1000
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
        total_loss, loss_dict = physics_loss(t_col, t_data, I_data, model)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optm.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_dict

@tf.function
def test_step(t_col, t_data, I_data):
    total_loss, loss_dict = physics_loss(t_col, t_data, I_data, model)
    return total_loss, loss_dict

print("Starting training...")
### 50,000 iterations
for itr in range(50_000):
    train_loss, train_loss_dict = train_step(t_col_tensor, t_train, I_train)
    train_loss_record.append(float(train_loss))

    test_loss, test_loss_dict = test_step(t_col_tensor, t_test, I_test)

    if itr % 1000 == 0:
        print(
            f"Iteration {itr}\n"
            f"Train Loss: {float(train_loss):.6f}, "
            f"Test Loss: {float(test_loss):.6f}\n"
            f"Data: {float(train_loss_dict['data_loss']):.6f}, "
            f"IC: {float(train_loss_dict['IC_loss']):.6f}, "
            f"Conservation: {float(train_loss_dict['conservation_loss']):.6f}, "
            f"ODE: {float(train_loss_dict['ODE_loss']):.6f}, "
            f"Beta Smooth: {float(train_loss_dict['beta_smooth_loss']):.6f}"
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
N = 56_000_000  # UK population for denormalization

### Plot PINN training and forecasting
t_pred_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
_, _, I_pred_train, _, _ = model(t_pred_tensor)
I_pred_train_np = to_numpy_flat(I_pred_train)

t_test_tensor = tf.convert_to_tensor(t_test, dtype=tf.float32)
_, _, I_pred_test, _, _ = model(t_test_tensor)
I_pred_test_np = to_numpy_flat(I_pred_test)

plt.figure(figsize=(14, 6))
plt.plot(t_train_np, I_pred_train_np * N, color="#ff7ee3", linewidth=1, label='I (PINN prediction - train)')
plt.plot(t_train_np, I_train_np * N, color="#004F94", linewidth=1, label='I (observed – train)')
plt.plot(t_test_np, I_test_np * N, color="#004F94", linewidth=1, label='I (observed – test)')
plt.plot(t_test_np, I_pred_test_np * N, color='#ff7ee3', label='I (PINN prediction - test)')
plt.axvline(x=t_train_np[-1], color='gray', linestyle='--', label='Train/Test Split')
plt.xlabel('Time')
plt.ylabel('Infected (actual)')
plt.title('SEIR PINN: Actual Infections')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Zoom in on actual data range
plt.ylim((I_data.min() * N) - 1000, (I_data.max() * N) + 1000)

plt.savefig('PINN_output.png')
plt.show()

# Plot prediction error on training data (in actual units)
plt.figure(figsize=(14, 6))
error = I_pred_train_np - I_train_np
plt.plot(t_train_np, error * N, label='Prediction Error (actual)', color='red')
plt.axhline(y=0, color='black', linestyle='--', label='Zero Error')
plt.xlabel('Time')
plt.ylabel('Error (actual infections)')
plt.title('Prediction Error on Training Data')
plt.legend()
plt.grid(True)
plt.savefig('prediction_error.png')
plt.show()

### Plot beta over time
t_plot = np.linspace(0.0, 1.0, 500)
t_plot_tensor = tf.convert_to_tensor(t_plot.reshape(-1, 1), dtype=tf.float32)
_, _, _, _, beta = model.predict(t_plot_tensor)

plt.figure(figsize=(8,5))
plt.plot(t_plot, beta.flatten(), 'g-', linewidth=2)
plt.xlabel('Normalised time')
plt.ylabel('β(t)')
plt.ylim(0, 1)  
plt.grid(True)
plt.show()
plt.savefig('Beta_over_time.png', dpi=300)

### Model evaluation - mean absolute error - test error
mae_test = tf.keras.losses.MeanAbsoluteError()(I_test, I_pred[split:]).numpy()
print("Mean Absolute Error:", mae_test)

### Model evaluation - mean sqaured error - test error
mse_test = tf.keras.losses.MeanSquaredError()(I_test, I_pred[split:]).numpy()
print("Mean Squared Error:", mse_test)