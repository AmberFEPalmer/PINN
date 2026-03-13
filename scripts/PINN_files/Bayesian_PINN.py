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
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

### Path to the data folder from the script location
data_folder = os.path.join("..", "..", "data")

### Load CSV 
data_path = os.path.join(data_folder, "SEIR_time_varying_results.csv")
data = pd.read_csv(data_path)

t_data = data["time"].values.reshape(-1, 1)
I_data = data["I"].values.reshape(-1, 1)    

### Train/test split
N_obs = len(I_data)
t_data = t_data[:N_obs].reshape(-1, 1)
I_data = I_data.reshape(-1, 1)
### Generate training and testing data 
split = int(0.9 * N_obs) 
t_train = t_data[:split] ### take all elements from 0 up to "split"
I_train = I_data[:split]
t_test  = t_data[split:] ### take all elements from "split" to the end
I_test  = I_data[split:]

I_scale = tf.constant(float(I_train.max()), dtype=tf.float32)

### Convert to tensors (multi dimensional arrays)
### Array = objects all of the same type
### Need to convert from an array to a tensor for neural network
t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
I_train_tensor = tf.convert_to_tensor(I_train, dtype=tf.float32)

N_data = t_train.shape[0]
N_phys = t_train_tensor.shape[0]   

### Control Kl divergence
kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)

### Define PINN
def create_bayesian_pinn_model():
    ### Import time as input
    t_input = Input(shape=(1,), name='time_input')

    ### Dense flipout = Bayesian dense layer
    ### Weights are probability distributions
    ### 50 neurons
    ### tanh activation
    ### Variational inference - approximate the posterior
    ### kl = kullback-Leibler divergence -> measures how different two probability distributions are
    ### https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout
    ### https://arxiv.org/abs/1803.04386
    seir = tfpl.DenseFlipout(
        50,
        activation='tanh',
        kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p) 
        )(t_input)
    
    seir = tfpl.DenseFlipout(
        50,
        activation='tanh',
        kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p)
        )(seir)
    
    seir = tfpl.DenseFlipout(
        50,
        activation='tanh',
        kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p) 
        )(seir)

    ### Outputs -> 4 layers -> S, E, I, R
    S = tfpl.DenseFlipout(1, activation='softplus', name='S')(seir)
    E = tfpl.DenseFlipout(1, activation='softplus', name='E')(seir)
    I = tfpl.DenseFlipout(1, activation='softplus', name='I')(seir)
    R = tfpl.DenseFlipout(1, activation='softplus', name='R')(seir)

    ### Seperate neural network for beta
    beta_hidden = tfpl.DenseFlipout(
        50,
        activation='tanh',
        kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p) 
    )(t_input)

    beta_hidden = tfpl.DenseFlipout(
        50,
        activation='tanh',
        kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p) 
    )(beta_hidden)
    
    beta_hidden = tfpl.DenseFlipout(
        50,
        activation='tanh',
        kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p) 
    )(beta_hidden)

    beta = tfpl.DenseFlipout(1, activation='softplus', name='beta')(beta_hidden)

    model = Model(inputs=t_input, outputs=[S, E, I, R, beta])
    return model

model = create_bayesian_pinn_model()
### Print model architecture
model.summary()

### Define initial conditions
S0 = tf.constant(100000/100001, dtype=tf.float32)
E0 = tf.constant(0.0, dtype=tf.float32)
I0 = tf.constant(1/100001, dtype=tf.float32)
R0 = tf.constant(0.0, dtype=tf.float32)

### Define loss function 
def loss_function(t_col, t_data_loss, I_data_loss, net, I_scale):
    
    ### if t_col is a 1D array it is reshaped to a column vector
    if len(t_col.shape) == 1:t_col = tf.reshape(t_col, (-1, 1))
    
    ### Convert data to tensors 
    if not isinstance(t_data_loss, tf.Tensor):t_data_loss = tf.convert_to_tensor(t_data_loss, dtype=tf.float32)
    if not isinstance(I_data_loss, tf.Tensor):I_data_loss = tf.convert_to_tensor(I_data_loss, dtype=tf.float32)

    ### reshape arrays to column vectors
    if len(t_data_loss.shape) == 1:t_data_loss = tf.reshape(t_data_loss, (-1, 1))
    if len(I_data_loss.shape) == 1:I_data_loss = tf.reshape(I_data_loss, (-1, 1))
    
    ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
    ### Gradient tape is used to record operations for automatic differentiation
    ### Calculate the gradients of a computation
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, E, I, R, beta = net(t_col)
        
    ### Define parameters which don't vary over time
    ### Following what was done in Qian et al. 2025
    sigma = tf.constant(0.1, dtype=tf.float32, name='sigma')
    gamma = tf.constant(0.1, dtype=tf.float32, name='gamma')
    
    ### Compute derivatives e.g. dS/dt
    dS_dt = tape.gradient(S, t_col) 
    dE_dt = tape.gradient(E, t_col) 
    dI_dt = tape.gradient(I, t_col) 
    dR_dt = tape.gradient(R, t_col) 
    
    d_beta_dt = tape.gradient(beta, t_col)
    beta_smooth_loss = tf.reduce_mean(tf.square(d_beta_dt))
    
    del tape

    ### SEIR equations 
    days = 100.0
    T = tf.constant(days, dtype=tf.float32)
    dS_dt_physics = T * (-beta * S * I)
    dE_dt_physics = T * (beta * S * I - sigma * E)
    dI_dt_physics = T * (sigma * E - gamma * I)
    dR_dt_physics = T * (gamma * I)
 
    ### Physics-informed loss - mean squared error
    loss_S = tf.reduce_mean(tf.square(dS_dt - dS_dt_physics))
    loss_E = tf.reduce_mean(tf.square(dE_dt - dE_dt_physics))
    loss_I = tf.reduce_mean(tf.square(dI_dt - dI_dt_physics))
    loss_R = tf.reduce_mean(tf.square(dR_dt - dR_dt_physics))

    ODE_loss = (
        1.0 * loss_S +
        1.0 * loss_E +
        1.0 * loss_I +
        1.0 * loss_R 
    )

    ### Initial condition loss (evaluate at t=0)
    t_zero = tf.constant([[0.0]], dtype=tf.float32) 
    S_0, E_0, I_0, R_0, _ = net(t_zero)
    
    Initial_condition_loss = tf.reduce_mean(
        tf.square(S_0 - S0_fixed) +
        tf.square(E_0 - E0_fixed) +
        tf.square(I_0 - I0_fixed) +
        tf.square(R_0 - R0_fixed) )
    
    ### constrain SEIR equations to equal 1
    conservation_loss = tf.reduce_mean(tf.square(S + E + I + R - 1.0))
    
    ### Data loss 
    t_data_normalized = t_data_loss 
    _, _, I_pred, _, _ = net(t_data_normalized)
    data_loss = tf.reduce_mean(tf.square((I_pred - I_data_loss) / I_scale))
    
    ### Total loss
    N_data = t_train.shape[0]
    Kl_loss = tf.add_n(net.losses) / N_data
    total_loss = 1.0 * data_loss + 1.0 * Initial_condition_loss + 1.0 * conservation_loss + 0.1*ODE_loss + (kl_weight_var * Kl_loss)
    
    return total_loss, {
        "data_loss": data_loss,
        "IC_loss": Initial_condition_loss,
        "conservation_loss": conservation_loss,
        "ODE_loss": ODE_loss,
        "beta_smooth_loss": beta_smooth_loss,
        "KL_loss": tf.reduce_sum(net.losses)
    }

S0_fixed = S0
E0_fixed = E0
I0_fixed = I0
R0_fixed = R0

### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
### learning rate scheduler added (not in original paper)
initial_lr = 0.001
optm = tf.keras.optimizers.legacy.Adam(learning_rate=initial_lr)

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

### Monte carlo sampling (not bootstrapping or MCMC)
### https://link.springer.com/chapter/10.1007/978-1-0716-4132-3_7
def predict_with_uncertainty(model, t, n_samples=1000):
    preds = []
    for _ in range(n_samples):
        S, E, I, R, beta = model(t)
        preds.append(I.numpy())
    preds = np.array(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std

@tf.function
def train_step(t_col, t_data, I_data, I_scale):
    with tf.GradientTape() as tape:
        total_loss, loss_dict = loss_function(t_col, t_data, I_data, model, I_scale)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optm.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_dict

@tf.function
def test_step(t_col, t_data, I_data, I_scale):
    return loss_function(t_col, t_data, I_data, model, I_scale)

print("Starting training...")

### https://github.com/hubertrybka/vae-annealing?
### https://arxiv.org/abs/1903.10145
total_iters = 50000
kl_ramp_iters = 20000 
kl_max = 0.0001
for itr in range(total_iters):
    ### Linearly increase KL weight from 0 to kl_max over kl_ramp_iters
    kl_weight_var.assign(tf.minimum(kl_max, kl_max * itr / kl_ramp_iters))
    ### Training step
    train_loss, train_loss_dict = train_step(t_col_tensor, t_train, I_train, I_scale)
    train_loss_record.append(float(train_loss))

    ### Evaluate loss every 1000 iterations
    if itr % 1000 == 0:
        test_loss, test_loss_dict = test_step(t_col_tensor, t_test, I_test, I_scale)
        test_loss_record.append(float(test_loss))

    ### Print progress every 10000 iterations
    if itr % 10000 == 0:
        KL_raw = train_loss_dict["KL_loss"]
        KL_scaled = kl_weight_var * (KL_raw / N_data)

        print(
        f"Iteration {itr}, KL weight: {kl_weight_var.numpy():.5f}, "
        f"Train Loss: {float(train_loss):.6f}, "
        f"Test Loss: {float(test_loss):.6f},"
        f"Iteration {itr}\n"
        f"Train Loss: {float(train_loss):.6f}, "
        f"Data: {float(train_loss_dict['data_loss']):.6f}, "
        f"IC: {float(train_loss_dict['IC_loss']):.6f}, "
        f"Conservation: {float(train_loss_dict['conservation_loss']):.6f}, "
        f"ODE: {float(train_loss_dict['ODE_loss']):.6f}, "
        f"Beta Smooth: {float(train_loss_dict['beta_smooth_loss']):.6f}, "
        f"KL raw: {float(KL_raw):.6f}, "
        f"KL scaled: {float(KL_scaled):.6f}, "
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

def to_numpy_flat(arr):
    if hasattr(arr, 'numpy'):
        return arr.numpy().flatten()
    else:
        return arr.flatten()

### Flatten all arrays
t_data_np  = to_numpy_flat(t_data)         
I_pred_np  = to_numpy_flat(I_pred)
t_train_np = to_numpy_flat(t_train)
I_train_np = to_numpy_flat(I_train)
t_test_np  = to_numpy_flat(t_test)
I_test_np = to_numpy_flat(I_test)

### PINN plot without UC
plt.figure(figsize=(14, 6))
plt.plot(t_data_np, I_pred_np, color="#ff7ee3", linewidth=2, label='I (PINN prediction)')
plt.plot(t_train_np, I_train_np, color="#004F94", linestyle='-',linewidth=2, label='I (observed – train)')
plt.plot(t_test_np, I_test_np, color='#004F94', linestyle='-', linewidth=2, label='I (observed – test)')
plt.axvline(x=t_train_np[-1], color='gray', linestyle='--', label='Train/Test Split')
plt.xlabel('Time')
plt.ylabel('Infected (normalized)')
plt.title('SEIR PINN on simulated data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Bayesian PINN plot
plt.figure(figsize=(14, 6))

# Monte Carlo posterior
t_tensor = tf.convert_to_tensor(t_data_np.reshape(-1, 1), dtype=tf.float32)
mean, std = predict_with_uncertainty(model, t_tensor, n_samples=200)

mean = mean.flatten()
std  = std.flatten()

plt.plot(t_data_np, mean, color = '#ff7ee3', linewidth=2, label='I (posterior mean)')

### Observed data
plt.plot(t_train_np, I_train_np, color = "#004F94", linewidth=2, label='I (observed – train)')
plt.plot(t_test_np, I_test_np, '#004F94', linewidth=2, label='I (observed – test)')
plt.axvline(x=t_train_np[-1], color='gray', linestyle='--', label='Train/Test Split')

### Credible interval (95%)
plt.fill_between(
    t_data_np,
    mean - 2 * std,
    mean + 2 * std,
    color='#ff7ee3',
    alpha=0.25,
    label='95% credible interval'
)

output_dir = "../../png_files"

plt.xlabel('Time')
plt.ylabel('Infected (normalized)')
plt.title('SEIR Bayesian PINN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Bayesian_PINN_beta_constant_0.4.png'))
plt.show()

### Plot beta over time
t_plot = np.linspace(0.0, 1.0, 500)
t_plot_tensor = tf.convert_to_tensor(t_plot.reshape(-1, 1), dtype=tf.float32)

### Monte Carlo sampling
n_samples = 1000
beta_samples = []

for _ in range(n_samples):
    _, _, _, _, beta = model(t_plot_tensor)
    beta_samples.append(beta.numpy())

beta_samples = np.array(beta_samples)

beta_mean = beta_samples.mean(axis=0).flatten()
beta_std  = beta_samples.std(axis=0).flatten()

### Plot
plt.figure(figsize=(8,5))

plt.plot(t_plot, beta_mean, color='#7397de', linewidth=2, label='β(t) mean')

plt.fill_between(
    t_plot,
    beta_mean - 2*beta_std,
    beta_mean + 2*beta_std,
    color="#7397de",
    alpha=0.25,
    label='95% credible interval'
)

plt.xlabel('Normalized time')
plt.ylabel('β(t)')
plt.ylim(0, 1)  
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig('B-PINN_param_est_Beta_0.4.png', dpi=300)
plt.show()
