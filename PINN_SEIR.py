import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

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
### 4 weeks of the year = 7.69% 
### 100-7.69 = 92.31 = training data 
split = int(0.9 * N_obs) 

t_train = t_data[:split] ### take all elements from 0 up to "split"
I_train = I_data[:split]

t_test  = t_data[split:] ### take all elements from "split" to the end
I_test  = I_data[split:]

### Convert to tensors
t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
I_train_tensor = tf.convert_to_tensor(I_train, dtype=tf.float32)

### Print number of testing and training samples + time range
print(f"Training samples: {len(t_train)}")
print(f"Testing samples: {len(t_test)}")
print(f"Training time range: {t_train.min():.3f} to {t_train.max():.3f}")
print(f"Testing time range: {t_test.min():.3f} to {t_test.max():.3f}")

# Create collocation points across the full range of the data
t_col_tensor = tf.convert_to_tensor(t_col, dtype=tf.float32)

### Define PINN
### L2 regularisation for hidden layers 
### https://keras.io/api/layers/regularizers/
### https://developers.google.com/machine-learning/crash-course/overfitting/regularization
def create_pinn_model():
    ### Input layer = time 
    t_input = Input(shape=(1,), name='time_input')
    
    ### 4 Hidden layers, 64 neurons each , tanh activation (tanh = non-linear + smooth)
    ### tanh outputs values in [-1,1]
    ### Milleovi et al. 2024 - tanh for hidden layers, sigmoid for output
    x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(t_input)
    x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(x)
    
    ### Output layers for S, E, I, R
    ### Sigmoid outputs variables in [0, 1]
    S = Dense(1, activation='sigmoid', name='S')(x)
    E = Dense(1, activation='sigmoid', name='E')(x)
    I = Dense(1, activation='sigmoid', name='I')(x)
    R = Dense(1, activation='sigmoid', name='R')(x)

    ### Time-varying beta 
    ### beta = softplus activation -> allows it to be greater than 1
    beta_hidden = Dense(64, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-4))(x)
    beta_hidden = Dense(64, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-4))(beta_hidden)
    beta_hidden = Dense(64, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-4))(beta_hidden)
    beta_hidden = Dense(64, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-4))(beta_hidden)
    beta = Dense(1, activation='softplus', name='beta')(beta_hidden) 

    ### Create the model -> inputs = time, outputs = SEIR compartments and beta
    model = Model(inputs=t_input, outputs=[S, E, I, R, beta])
    return model

model = create_pinn_model()
### Print model architecture
model.summary()

### Define initial conditions
### TODO - do i need this if im not doing an initial conditon loss??
S0 = tf.constant(0.9, dtype=tf.float32)
E0 = tf.constant(0.05, dtype=tf.float32)
I0 = tf.constant(0.05, dtype=tf.float32)
R0 = tf.constant(0.0, dtype=tf.float32)

### Define physics informed loss
def seir_ode_loss(t_col, t_data_loss, I_data_loss, net, sigma_raw, gamma_raw):

### Apply softplus to parameters to ensure they remain positive
    sigma = tf.nn.softplus(sigma_raw)
    gamma = tf.nn.softplus(gamma_raw)

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
        
    ### Compute derivatives e.g. dS/dt
    dS_dt = tape.gradient(S, t_col) 
    dE_dt = tape.gradient(E, t_col) 
    dI_dt = tape.gradient(I, t_col) 
    dR_dt = tape.gradient(R, t_col) 
    del tape

    ### SEIR equations
    N = 1.0
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
    total_loss = 100.0*data_loss + 0.01*physics_loss
    
    return total_loss

### Define parameters which don't vary over time
### Following what was done in Qian et al. 2025
sigma_raw = tf.constant(0.25, dtype=tf.float32, name='sigma_raw')
gamma_raw = tf.constant(0.25, dtype=tf.float32, name='gamma_raw')

S0_fixed = S0
E0_fixed = E0
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
n_collocation = 365
t_col_uniform = np.linspace(0, 1, n_collocation).reshape(-1, 1)
t_col_tensor = tf.convert_to_tensor(t_col_uniform, dtype=tf.float32)

### Training loop
train_loss_record = []
test_loss_record = []  

trainable_vars = model.trainable_variables 

print("Starting training...")
for itr in range(60000): ### 60000 iterations
    with tf.GradientTape() as tape:
        ### Use training data only
        train_loss = seir_ode_loss(t_col_tensor, t_train, I_train, model, sigma_raw, gamma_raw)
   
    train_loss_record.append(train_loss.numpy())

    grad_w = tape.gradient(train_loss, trainable_vars)
    optm.apply_gradients(zip(grad_w, trainable_vars))
   
    ### Evaluate model on the test set 
    if itr % 100 == 0:
        test_loss = seir_ode_loss(t_col_tensor, t_test, I_test, model, sigma_raw, gamma_raw)
        test_loss_record.append(test_loss.numpy())
   
    if itr % 10000 == 0:
        print(f"Iteration {itr}, Train Loss: {train_loss.numpy():.6f}, Test Loss: {test_loss.numpy():.6f}")

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
plt.title('SEIR PINN: Training vs Forecasting')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('PINN_output.png')
plt.show()

### Plot beta over time
t_plot = np.linspace(0.0, 1.0, 500)
t_plot_tensor = tf.convert_to_tensor(t_plot.reshape(-1, 1), dtype=tf.float32)
_, _, _, _, beta = model.predict(t_plot_tensor)
plt.plot(t_plot, beta.flatten(), 'g-', linewidth=2)
plt.xlabel('normalised time')
plt.ylabel('β(t)')
plt.grid(True)
plt.show()
plt.savefig('Beta_over_time.png')

### Model evaluation - mean absolute error
mae_test = tf.keras.losses.MeanAbsoluteError()(I_test, I_pred[split:]).numpy()
print("Mean Absolute Error:", mae_test)

### Model evaluation - mean sqaured error
mse_test = tf.keras.losses.MeanSquaredError()(I_test, I_pred[split:]).numpy()
print("Mean Squared Error:", mse_test)

### Model evaluation - mean absolute percentage error
mape_test = tf.keras.losses.MeanAbsolutePercentageError()(I_test, I_pred[split:]).numpy()
print("Mean Absolute Percentage Error:", mape_test)