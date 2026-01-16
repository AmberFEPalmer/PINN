import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

### Main websites used 
### https://vitalitylearning.medium.com/solving-a-first-order-ode-with-physics-informed-neural-networks-22e385f09d35
### code from seminal paper https://github.com/maziarraissi/PINNs
### https://www.tensorflow.org/tutorials/customization/basics

### Load preprocessed data as arrays (from COVID_Data.py script)
t_data = np.load("data/t_data_2020.npy")     
### Store the max time for scaling
t_max = t_data.max()
I_data = np.load("data/I_data_2020.npy")      
t_col  = np.load("data/t_col.npy")       

### Convert to arrays Tensors (multi-dimensional list of numbers)
t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
I_tensor = tf.convert_to_tensor(I_data, dtype=tf.float32)

# Create collocation points across the full range of the data
t_col_tensor = tf.convert_to_tensor(t_col, dtype=tf.float32)

### Define PINN
def create_pinn_model():
    ### Input layer = time 
    t_input = Input(shape=(1,), name='time_input')
    
    ### 3 Hidden layers, 64 50 neurons each , tanh activation (tanh = non-linear + smooth)
    ### tanh outputs values in [-1,1]
    ### Same architecture as used in Qian et al. 2025
    x = Dense(64, activation='tanh')(t_input)
    x = Dense(128, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    
    ### Output layers for S, E, I, R
    ### Sigmoid outputs variables in [0, 1]
    S = Dense(1, activation='sigmoid', name='S')(x)
    E = Dense(1, activation='sigmoid', name='E')(x)
    I = Dense(1, activation='sigmoid', name='I')(x)
    R = Dense(1, activation='sigmoid', name='R')(x)

    ### Time-varying beta 
    ### Same architecture as used in Qian et al. 2025 (3 hidden layers, 50 neurons)
    ### beta = softplus activation -> allows it to be greater than 1
    beta_hidden = Dense(50, activation = 'tanh')(x)
    beta_hidden = Dense(50, activation = 'tanh')(x)
    beta_hidden = Dense(50, activation = 'tanh')(x)
    beta = Dense(1, activation='softplus', name='beta')(beta_hidden) 

    ### Create the model -> inputs = time, outputs = SEIR compartments
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
    if len(t_col.shape) == 1:
        t_col = tf.reshape(t_col, (-1, 1))
    
    ### Convert data to tensors 
    if not isinstance(t_data_loss, tf.Tensor):
        t_data_loss = tf.convert_to_tensor(t_data_loss, dtype=tf.float32)
    if not isinstance(I_data_loss, tf.Tensor):
        I_data_loss = tf.convert_to_tensor(I_data_loss, dtype=tf.float32)

    ### if t_data_loss is a 1D array it is reshaped to a column vector
    if len(t_data_loss.shape) == 1:
        t_data_loss = tf.reshape(t_data_loss, (-1, 1))
    
    ### if I_data_loss is a 1D array it is reshaped to a column vector
    if len(I_data_loss.shape) == 1:
        I_data_loss = tf.reshape(I_data_loss, (-1, 1))
    
    ### Physics loss at collocation points
    ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, E, I, R, beta = model(t_col)
        
    ### Compute derivatives e.g. dS/dt
    ### Derivatives are scaled because time is normalised
    scale_factor = tf.constant(t_data.max(), dtype = tf.float32)
    dS_dt = tape.gradient(S, t_col) * scale_factor 
    dE_dt = tape.gradient(E, t_col) * scale_factor 
    dI_dt = tape.gradient(I, t_col) * scale_factor 
    dR_dt = tape.gradient(R, t_col) * scale_factor 
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
    total_loss = 1000.0*data_loss 
    
    return total_loss

### Define parameters 
sigma_raw = tf.Variable(0.2, dtype=tf.float32, name='sigma_raw')
gamma_raw = tf.Variable(0.2, dtype=tf.float32, name='gamma_raw')

S0_fixed = S0
E0_fixed = E0
I0_fixed = I0
R0_fixed = R0

### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
initial_lr = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True
)
optm = Adam(learning_rate=lr_schedule)

### Collocation points for physics loss
trainable_vars = model.trainable_variables + [sigma_raw, gamma_raw]
n_collocation = 500
t_col_uniform = np.linspace(0, 1, n_collocation).reshape(-1, 1)
t_col_tensor = tf.convert_to_tensor(t_col_uniform, dtype=tf.float32)

### Training loop
train_loss_record = []

print("Starting training...")
for itr in range(60000):
    with tf.GradientTape() as tape:
        train_loss = seir_ode_loss(t_col_tensor, t_data, I_data, model, 
                                   sigma_raw, gamma_raw)
    
    train_loss_record.append(train_loss.numpy())
    
    grad_w = tape.gradient(train_loss, trainable_vars)
    optm.apply_gradients(zip(grad_w, trainable_vars))
    
    if itr % 5000 == 0:
        print(f"Iteration {itr}, Loss: {train_loss.numpy():.6f}")
        sigma_current = tf.nn.softplus(sigma_raw).numpy()
        gamma_current = tf.nn.softplus(gamma_raw).numpy()
        print(f"σ={sigma_current:.4f}, γ={gamma_current:.4f}")

print("\nTraining complete!")
print(f"\nFinal learned parameters:")
sigma_final = tf.nn.softplus(sigma_raw).numpy()
gamma_final = tf.nn.softplus(gamma_raw).numpy()
print(f"σ (incubation rate) = {sigma_final:.4f}")
print(f"γ (recovery rate) = {gamma_final:.4f}")

### Plot training loss
t_test = t_data  
_, _, I_pred, _, _ = model(t_tensor)
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.yscale('log')  
plt.grid(True)
plt.show()
plt.savefig('PINN_training_loss.png')

### Plot infected compartment vs data
plt.plot(t_test, I_pred, 'b-', label='I (predicted)', linewidth=2)
plt.plot(t_data, I_data, color='red', label='I (observed)', linewidth=2)
plt.xlabel('Normalized Time')
plt.ylabel('Infected (normalized)')
plt.title('Infected Compartment vs Observed Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('Infected_compartment_vs_data.png')

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
mae = tf.keras.losses.MeanAbsoluteError()
mae_value = mae(I_data, I_pred).numpy()
print("Mean Absolute Error:", mae_value)

### Model evaluation - mean sqaured error
mse = tf.keras.losses.MeanSquaredError()
mse_value = mse(I_data, I_pred).numpy()
print("Mean Squared Error:", mse_value)

### Model evaluation - mean absolute percentage error
mape = tf.keras.losses.MeanAbsolutePercentageError()
mape_value = mape(I_data, I_pred).numpy()
print("Mean Absolute Percentage Error:", mape_value)