import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import pandas as pd
import os

data_folder = os.path.join("..", "synthetic_data_generation")
output_dir  = os.path.join("..", "..", "png_files")
os.makedirs(output_dir, exist_ok=True)

data_path = os.path.join(data_folder, "SEIR_metapopulation_two_patch.csv")
data = pd.read_csv(data_path)

t_data = data["time"].values.reshape(-1, 1)
I1_data = data["I1"].values.reshape(-1, 1)
I2_data = data["I2"].values.reshape(-1, 1) 

t_data = data["time"].values.reshape(-1, 1)
t_data = t_data / t_data.max()  # normalise to [0, 1]

### Train/test split
N_obs = len(t_data)
t_data = t_data[:N_obs].reshape(-1, 1)

### Generate training and testing data 
split = int(0.9 * N_obs) 
I1_train = I1_data[:split]
I1_test  = I1_data[split:]
I2_train = I2_data[:split]
I2_test  = I2_data[split:]
t_train = t_data[:split]
t_test  = t_data[split:]

I_scale = tf.constant(float(I1_train.max()), dtype=tf.float32)

### Convert to tensors (multi dimensional arrays)
### Array = objects all of the same type
### Need to convert from an array to a tensor for neural network
I1_train = tf.convert_to_tensor(I1_train, dtype=tf.float32)
I1_test  = tf.convert_to_tensor(I1_test,  dtype=tf.float32)
I2_train = tf.convert_to_tensor(I2_train, dtype=tf.float32)
I2_test  = tf.convert_to_tensor(I2_test,  dtype=tf.float32)

### Define PINN
### L2 regularisation for hidden layers -> helps to prevent overfitting
### Add penalty proportional to the sum of squared coefficients to the loss function
### Reduce model complexity, penalise large weights
### https://keras.io/api/layers/regularizers/
### https://developers.google.com/machine-learning/crash-course/overfitting/regularization
def create_pinn_model():
    t_input = Input(shape=(1,), name='time_input')

    ### SEIR subnet — 3 hidden layers, 50 neurons, tanh (Qian et al. 2025)
    seir1 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    seir1 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir1)
    seir1 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir1)
    
    seir2 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    seir2 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir2)
    seir2 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir2)

    ### SEIR outputs 
    ### 4 output layers
    ### No activation function
    S1 = Dense(1, activation="softplus", name='S1')(seir1)
    E1 = Dense(1, activation="softplus", name='E1')(seir1)
    I1 = Dense(1, activation="softplus", name='I1')(seir1)
    R1 = Dense(1, activation="softplus", name='R1')(seir1)
    
    S2 = Dense(1, activation="softplus", name='S2')(seir2)
    E2 = Dense(1, activation="softplus", name='E2')(seir2)
    I2 = Dense(1, activation="softplus", name='I2')(seir2)
    R2 = Dense(1, activation="softplus", name='R2')(seir2)

    ### Time-varying beta 
    ### beta = softplus activation -> allows it to be greater than 1
    ### 3 hidden layers, 50 neurons, tanh activation

    ### Beta outputs
    ### One output layer
    ### No activation function on log_beta, then exponentiate to ensure positivity
    ### Beta patch 1 subnet 
    beta1_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    beta1_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta1_hidden)
    beta1_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta1_hidden)

    ### Beta patch 2 subnet 
    beta2_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    beta2_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta2_hidden)
    beta2_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta2_hidden)

    ### Beta outputs 
    log_beta1 = Dense(1, activation=None, name='log_beta1')(beta1_hidden)
    log_beta2 = Dense(1, activation=None, name='log_beta2')(beta2_hidden)
    beta1 = Lambda(lambda x: tf.exp(x), name='beta1')(log_beta1)
    beta2 = Lambda(lambda x: tf.exp(x), name='beta2')(log_beta2)
    
    return Model(inputs=t_input, outputs=[S1, E1, I1, R1, beta1, S2, E2, I2, R2, beta2])
    
model = create_pinn_model()
### Print model architecture
model.summary()

### Define initial conditions
### Patch 1 
S0_p1 = tf.constant(49999/50000, dtype=tf.float32)
E0_p1 = tf.constant(0.0,         dtype=tf.float32)
I0_p1 = tf.constant(1/50000,     dtype=tf.float32)
R0_p1 = tf.constant(0.0,         dtype=tf.float32)

### Patch 2
S0_p2 = tf.constant(1, dtype=tf.float32)
E0_p2 = tf.constant(0.0,         dtype=tf.float32)
I0_p2 = tf.constant(0,     dtype=tf.float32)
R0_p2 = tf.constant(0.0,         dtype=tf.float32)

### Loss function
def loss_function(t_col, t_data_loss, I1_data_loss, I2_data_loss, net, I_scale, smooth_beta=True):

    ### if t_col is a 1D array it is reshaped to a column vector
    if len(t_col.shape) == 1:t_col = tf.reshape(t_col, (-1, 1))
    
    ### Convert data to tensors 
    ### Convert to column shape
    if not isinstance(I1_data_loss, tf.Tensor): I1_data_loss = tf.convert_to_tensor(I1_data_loss, dtype=tf.float32)
    if not isinstance(I2_data_loss, tf.Tensor): I2_data_loss = tf.convert_to_tensor(I2_data_loss, dtype=tf.float32)
    if len(I1_data_loss.shape) == 1: I1_data_loss = tf.reshape(I1_data_loss, (-1, 1))
    if len(I2_data_loss.shape) == 1: I2_data_loss = tf.reshape(I2_data_loss, (-1, 1))
    
    if not isinstance(t_data_loss, tf.Tensor): t_data_loss = tf.convert_to_tensor(t_data_loss, dtype=tf.float32)
    if len(t_data_loss.shape) == 1: t_data_loss = tf.reshape(t_data_loss, (-1, 1))
    
    ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
    ### Gradient tape is used to record operations for automatic differentiation
    ### Calculate the gradients of a computation
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S1, E1, I1, R1, beta1, S2, E2, I2, R2, beta2 = net(t_col)

    ### Define parameters which don't vary over time
    ### Following what was done in Qian et al. 2025
    sigma = tf.constant(0.25, dtype=tf.float32)
    gamma = tf.constant(0.25, dtype=tf.float32)
    
    ### Population sizes for each patch
    N1 = tf.constant(50000.0, dtype=tf.float32)
    N2 = tf.constant(50000.0, dtype=tf.float32)

    ### Compute derivatives e.g. dS/dt 
    ### Use automatic differentiation
    dS1_dt = tape.gradient(S1, t_col)
    dE1_dt = tape.gradient(E1, t_col)
    dI1_dt = tape.gradient(I1, t_col)
    dR1_dt = tape.gradient(R1, t_col)
    dS2_dt = tape.gradient(S2, t_col)
    dE2_dt = tape.gradient(E2, t_col)
    dI2_dt = tape.gradient(I2, t_col)
    dR2_dt = tape.gradient(R2, t_col)
    d_beta1_dt = tape.gradient(beta1, t_col)
    d_beta2_dt = tape.gradient(beta2, t_col)
    beta_smooth_loss = tf.reduce_mean(tf.square(d_beta1_dt)) + tf.reduce_mean(tf.square(d_beta2_dt))

    del tape

    ### SEIR equations with migration
    m = 0.01
    
    lambda1 = beta1 * I1 / N1
    lambda2 = beta2 * I2 / N2

    T = tf.constant(100.0, dtype=tf.float32)  # total days

    dS1 = T * (-lambda1 * S1 + m * (S2 - S1))
    dE1 = T * (lambda1 * S1 - sigma * E1 + m * (E2 - E1))
    dI1 = T * (sigma * E1 - gamma * I1 + m * (I2 - I1))
    dR1 = T * (gamma * I1 + m * (R2 - R1))

    dS2 = T * (-lambda2 * S2 + m * (S1 - S2))
    dE2 = T * (lambda2 * S2 - sigma * E2 + m * (E1 - E2))
    dI2 = T * (sigma * E2 - gamma * I2 + m * (I1 - I2))
    dR2 = T * (gamma * I2 + m * (R1 - R2))

    ### Physics informed loss
    ODE_loss = (
    tf.reduce_mean(tf.square(dS1_dt - dS1)) +
    tf.reduce_mean(tf.square(dE1_dt - dE1)) +
    tf.reduce_mean(tf.square(dI1_dt - dI1)) +
    tf.reduce_mean(tf.square(dR1_dt - dR1)) +
    tf.reduce_mean(tf.square(dS2_dt - dS2)) +
    tf.reduce_mean(tf.square(dE2_dt - dE2)) +
    tf.reduce_mean(tf.square(dI2_dt - dI2)) +
    tf.reduce_mean(tf.square(dR2_dt - dR2))
)

    ### Initial condition loss (evaluate at t=0)
    t_zero = tf.constant([[0.0]], dtype=tf.float32)
    S1_0, E1_0, I1_0, R1_0, _, S2_0, E2_0, I2_0, R2_0, _ = net(t_zero)
    Initial_condition_loss = tf.reduce_mean(
    tf.square(S1_0 - S0_p1) + tf.square(E1_0 - E0_p1) +
    tf.square(I1_0 - I0_p1) + tf.square(R1_0 - R0_p1) +
    tf.square(S2_0 - S0_p2) + tf.square(E2_0 - E0_p2) +
    tf.square(I2_0 - I0_p2) + tf.square(R2_0 - R0_p2)
)

    ### constrain SEIR equations to equal 1
    conservation_loss = (
    tf.reduce_mean(tf.square(S1 + E1 + I1 + R1 - 1.0)) +
    tf.reduce_mean(tf.square(S2 + E2 + I2 + R2 - 1.0))
)

    ### Data loss 
    _, _, I1_pred, _, _, _, _, I2_pred, _, _ = net(t_data_loss)
    data_loss = (
    tf.reduce_mean(tf.square((I1_pred - I1_data_loss) / I_scale)) +
    tf.reduce_mean(tf.square((I2_pred - I2_data_loss) / I_scale))
)

    ### Total loss
    total_loss = (
            1.0 * data_loss +
            0.1 * ODE_loss +
            1.0 * Initial_condition_loss +
            1.0 * conservation_loss +
            0.1 * beta_smooth_loss
        )

    return total_loss, {
        "data_loss":         data_loss,
        "IC_loss":           Initial_condition_loss,
        "conservation_loss": conservation_loss,
        "ODE_loss":          ODE_loss,
        "Smooth_beta_loss":  beta_smooth_loss
    }

### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
### Optimiser and collocation points 
optm = Adam(learning_rate=0.001)

### Collocation points for physics
n_collocation  = 1000
t_col_tensor   = tf.convert_to_tensor(
    np.linspace(0, 1, n_collocation).reshape(-1, 1), dtype=tf.float32
)
 
### ensure all inputs are float32 for training
t_train  = tf.convert_to_tensor(t_train,  dtype=tf.float32)
t_test   = tf.convert_to_tensor(t_test,   dtype=tf.float32)
I1_train = tf.convert_to_tensor(I1_train, dtype=tf.float32)
I1_test  = tf.convert_to_tensor(I1_test,  dtype=tf.float32)
I2_train = tf.convert_to_tensor(I2_train, dtype=tf.float32)
I2_test  = tf.convert_to_tensor(I2_test,  dtype=tf.float32)

### Training loop
train_loss_record = []
test_loss_record = []  

@tf.function
def train_step(t_col, t_data, I1_data, I2_data):
    with tf.GradientTape() as tape:
        total_loss, loss_dict = loss_function(t_col, t_data, I1_data, I2_data, model, I_scale, smooth_beta=True)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optm.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_dict

@tf.function
def test_step(t_col, t_data, I1_data, I2_data):
    total_loss, loss_dict = loss_function(t_col, t_data, I1_data, I2_data, model, I_scale, smooth_beta=True)
    return total_loss, loss_dict

print("Starting training...")
### 50,000 iterations (Qian et al. 2025)
for itr in range(30000):
    train_loss, train_loss_dict = train_step(t_col_tensor, t_train, I1_train, I2_train)
    test_loss,  test_loss_dict  = test_step(t_col_tensor, t_test, I1_test, I2_test)
    train_loss_record.append(float(train_loss))
    test_loss_record.append(float(test_loss))

    if itr % 1000 == 0:
        print(
            f"Iteration {itr}\n"
            f"Train Loss: {float(train_loss):.6f}, "
            f"Test Loss: {float(test_loss):.6f}\n"
            f"Data: {float(train_loss_dict['data_loss']):.6f}, "
            f"IC: {float(train_loss_dict['IC_loss']):.6f}, "
            f"Conservation: {float(train_loss_dict['conservation_loss']):.6f}, "
            f"ODE: {float(train_loss_dict['ODE_loss']):.6f}, "
        )

def to_numpy_flat(arr):
    if hasattr(arr, 'numpy'):
        return arr.numpy().flatten()
    else:
        return arr.flatten()

t_data_np  = to_numpy_flat(t_data)
t_train_np = to_numpy_flat(t_train)
t_test_np  = to_numpy_flat(t_test)

### Plot training loss
t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
_, _, I1_pred, _, _, _, _, I2_pred, _, _ = model(t_tensor)
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

I1_pred_np = to_numpy_flat(I1_pred)
I2_pred_np = to_numpy_flat(I2_pred)
I1_train_np = to_numpy_flat(I1_train)
I2_train_np = to_numpy_flat(I2_train)
I1_test_np  = to_numpy_flat(I1_test)
I2_test_np  = to_numpy_flat(I2_test)

plt.figure(figsize=(14, 6))
plt.plot(t_data_np, I1_pred_np, color="#ff7ee3", linewidth=2, label='I patch 1 (PINN)')
plt.plot(t_data_np, I2_pred_np, color="#ffb347", linewidth=2, label='I patch 2 (PINN)')
plt.plot(t_train_np, I1_train_np, color="#004F94", linewidth=2, label='I patch 1 (observed)')
plt.plot(t_train_np, I2_train_np, color="#228B22", linewidth=2, label='I patch 2 (observed)')
plt.plot(t_test_np, I1_test_np, color="#004F94", linewidth=2, linestyle='--', label='I patch 1 (observed – test)')
plt.plot(t_test_np, I2_test_np, color="#228B22", linewidth=2, linestyle='--', label='I patch 2 (observed – test)')
plt.axvline(x=t_train_np[-1],color='gray', linestyle='--', label='Train/Test Split')

plt.xlabel('Time')
plt.ylabel('Infected (normalized)')
plt.title('PINN network simulated data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PINN_metapopulation.png'))
plt.show()
