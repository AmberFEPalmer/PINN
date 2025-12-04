import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

### https://i-systems.github.io/tutorial/KSNVE/220525/01_PINN.html
### My PINN creation is slightly different due to having 4 coupled ODEs

### https://vitalitylearning.medium.com/solving-a-first-order-ode-with-physics-informed-neural-networks-22e385f09d35

### code from seminal paper https://github.com/maziarraissi/PINNs

### TODO test-train split data
### Time varying parameters
### evaluate model on different time frames

### Load preprocessed data (from COVID_Data.py script)
### These data are arrays
t_data = np.load("t_data.npy")       ### time points 
I_data = np.load("I_data.npy")       ### observed infections
t_col  = np.load("t_col.npy")        ### collocation points for physics loss

### Convert to TensorFlow tensors (so they can be used for model training)
### tensor = multi-dimensional list of numbers
t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
I_tensor = tf.convert_to_tensor(I_data, dtype=tf.float32)
t_col_tensor = tf.convert_to_tensor(t_col, dtype=tf.float32)

### Define PINN
def create_pinn_model():
    ### Input layer - time (shape = 1 because time is 1D)
    t_input = Input(shape=(1,), name='time_input')
    
    ### Hidden layer 1 = 32 neurons, tanh activation
    ### Tanh activation is a good choice for this model because it is non-linear
    x = Dense(32, activation='tanh')(t_input)
    
    ### Hidden layers 2 + 3 = 64 neurons, tanh activation  
    x = Dense(64, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    
    ### Hidden layer 4 = 32 neurons, tanh activation       
    x = Dense(32, activation='tanh')(x)
    
    ### Output layers for S, E, I, R
    S = Dense(1, name='S')(x)
    E = Dense(1, name='E')(x)
    I = Dense(1, name='I')(x)
    R = Dense(1, name='R')(x)

    ### Create the model - inputs = time, outputs = SEIR compartments
    model = Model(inputs=t_input, outputs=[S, E, I, R])
    return model

model = create_pinn_model()
### Print model architecture
model.summary()

### Define physics informed loss
def seir_ode_loss(t_col, t_data_loss, I_data_loss, net, beta, sigma, gamma):
    ### Convert inputs to TensorFlow tensors 
    t_col = tf.convert_to_tensor(t_col, dtype=tf.float32)

    ### if t_col is a 1D array it is reshaped to a column vector
    if len(t_col.shape) == 1:
        t_col = tf.reshape(t_col, (-1, 1))
    
    ### if t_data_loss is a 1D array it is reshaped to a column vector
    t_data_loss = tf.convert_to_tensor(t_data_loss, dtype=tf.float32)
    if len(t_data_loss.shape) == 1:
        t_data_loss = tf.reshape(t_data_loss, (-1, 1))
    
    ### if I_data_loss is a 1D array it is reshaped to a column vector
    I_data_loss = tf.convert_to_tensor(I_data_loss, dtype=tf.float32)
    if len(I_data_loss.shape) == 1:
        I_data_loss = tf.reshape(I_data_loss, (-1, 1))
    
    ### Estimate starting values for the SEIR equations based on the first infection value
    I0_val = float(I_data_loss[0])   
    E0_val = 3.0 * I0_val 
    R0_val = 0.0 ### no individuals start recovered
    S0_val = 1.0 - I0_val - E0_val - R0_val 
    S0_val = max(S0_val, 0.0)

    ### Convert to tensors 
    S0 = tf.constant([[S0_val]], dtype=tf.float32)
    E0 = tf.constant([[E0_val]], dtype=tf.float32)
    I0 = tf.constant([[I0_val]], dtype=tf.float32)
    R0 = tf.constant([[R0_val]], dtype=tf.float32)
    
    N = 1.0  ### total population in scaled units
    
    ### Physics loss at collocation points
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, E, I, R = net(t_col)
        
    ### Compute derivatives e.g. dS/dt
    dS_dt = tape.gradient(S, t_col)
    dE_dt = tape.gradient(E, t_col)
    dI_dt = tape.gradient(I, t_col)
    dR_dt = tape.gradient(R, t_col)
    del tape  
    
    ### SEIR equations
    dS_dt_true = -beta * S * I / N
    dE_dt_true = beta * S * I / N - sigma * E
    dI_dt_true = sigma * E - gamma * I
    dR_dt_true = gamma * I
    
    ### Physics-informed loss - mean squared error
    physics_loss = tf.reduce_mean(
        tf.square(dS_dt - dS_dt_true) +
        tf.square(dE_dt - dE_dt_true) +
        tf.square(dI_dt - dI_dt_true) +
        tf.square(dR_dt - dR_dt_true)
    )
    
    ### Initial condition loss (evaluate at t=0)
    t_zero = tf.constant([[0.0]], dtype=tf.float32)
    S_0, E_0, I_0, R_0 = net(t_zero)
    
    IC_loss = (tf.square(S_0 - S0) + tf.square(E_0 - E0) + 
               tf.square(I_0 - I0) + tf.square(R_0 - R0))
    IC_loss = tf.reduce_sum(IC_loss)
    
    ### Data loss 
    S_pred, E_pred, I_pred, R_pred = net(t_data_loss)  # Unpack all outputs
    data_loss = tf.reduce_mean(tf.square(I_pred - I_data_loss))
    
    ### Total loss
    total_loss = physics_loss + 1.0*IC_loss + 10.0*data_loss
    
    return total_loss

### Define parameters 
beta = tf.Variable(0.8, dtype=tf.float32, name='beta')
sigma = tf.Variable(0.2, dtype=tf.float32, name='sigma')
gamma = tf.Variable(0.1, dtype=tf.float32, name='gamma')

### Optimizer
optm = Adam(learning_rate=0.01) ### Adam = one of the most common optimisers

### Get trainable variables
if isinstance(beta, tf.Variable):
    ### If parameters are trainable, include them
    trainable_vars = model.trainable_variables + [beta, sigma, gamma]
else:
    ### If parameters are fixed, only train network weights
    trainable_vars = model.trainable_variables

### Collocation points for physics loss
### Collocation points cover the time of the model
### 100 points where the physics loss is evaluated in the model
train_t = np.linspace(0, 1, 200).reshape(-1, 1)

### Training loop
train_loss_record = []

print("Starting training...")
for itr in range(10000):
    with tf.GradientTape() as tape:
        train_loss = seir_ode_loss(train_t, t_data, I_data, model, beta, sigma, gamma)
    
    train_loss_record.append(train_loss.numpy())
    
    grad_w = tape.gradient(train_loss, trainable_vars)
    optm.apply_gradients(zip(grad_w, trainable_vars))
    
    if itr % 1000 == 0:
        print(f"Iteration {itr}, Loss: {train_loss.numpy():.6f}")
        if isinstance(beta, tf.Variable):
            print("Training complete!")

### Plot training loss
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.yscale('log')  # Log scale often helps visualize loss
plt.grid(True)
plt.show()

### Visualize SEIR predictions
t_test = np.linspace(0, 1, 200).reshape(-1, 1)  # Test points over full time range
S_pred, E_pred, I_pred, R_pred = model.predict(t_test)

plt.figure(figsize=(12, 5))

# Plot infected compartment vs data
plt.plot(t_test, I_pred, 'b-', label='I (predicted)', linewidth=2)
plt.scatter(t_data, I_data, color='red', label='I (observed)', s=50, zorder=5)
plt.xlabel('Normalized Time')
plt.ylabel('Infected (normalized)')
plt.title('Infected Compartment vs Observed Data')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()