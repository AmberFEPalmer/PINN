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
 
data_path = os.path.join(data_folder, "SEIR_metapopulation_5_patch.csv")
data = pd.read_csv(data_path)
 
t_data = data["time"].values.reshape(-1, 1)
t_data = t_data / t_data.max()
 
N_obs = len(t_data)
split = int(0.9 * N_obs)
 
t_train = tf.convert_to_tensor(t_data[:split], dtype=tf.float32)
t_test  = tf.convert_to_tensor(t_data[split:], dtype=tf.float32)
 
### SEIR
P = 5
 
I_train_list = []
I_test_list = []
 
for i in range(1, P + 1):
    I = data[f"I{i}"].values.reshape(-1, 1)
    I_train_list.append(tf.convert_to_tensor(I[:split], dtype=tf.float32))
    I_test_list.append(tf.convert_to_tensor(I[split:], dtype=tf.float32))
 
# scaling constant
I_scale = tf.constant(float(np.max([np.max(x.numpy()) for x in I_train_list])), dtype=tf.float32)
 
### PINN model definition
def create_pinn_model(P):
    t_input = Input(shape=(1,), dtype=tf.float32, name='time_input')
    patch_input = Input(shape=(1,), dtype=tf.int32, name='patch_input')
 
    patch_embed = tf.keras.layers.Embedding(
        input_dim=P,
        output_dim=8,
        name='patch_embedding'
    )(patch_input)
    patch_embed = tf.keras.layers.Flatten()(patch_embed)
 
    x = tf.keras.layers.Concatenate()([t_input, patch_embed])
    x = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(x)
 
    S = Dense(1, activation="softplus", name='S')(x)
    E = Dense(1, activation="softplus", name='E')(x)
    I = Dense(1, activation="softplus", name='I')(x)
    R = Dense(1, activation="softplus", name='R')(x)
 
    bx = tf.keras.layers.Concatenate()([t_input, patch_embed])
    bx = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(bx)
    bx = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(bx)
    bx = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(bx)
 
    log_beta = Dense(1, activation=None, name='log_beta')(bx)
    beta = Lambda(lambda x: tf.exp(x), name='beta')(log_beta)
 
    return Model(inputs=[t_input, patch_input], outputs=[S, E, I, R, beta])
 
model = create_pinn_model(P)
model.summary()
 
### Initial conditions
S0 = np.full(P, 49999/50000)
E0 = np.zeros(P)
I0 = np.zeros(P)
R0 = np.zeros(P)
I0[0] = 1/50000
 
S0 = [tf.constant(x, dtype=tf.float32) for x in S0]
E0 = [tf.constant(x, dtype=tf.float32) for x in E0]
I0 = [tf.constant(x, dtype=tf.float32) for x in I0]
R0 = [tf.constant(x, dtype=tf.float32) for x in R0]
 
### Loss function
def loss_function(t_col, t_data_loss, I_data_list, net, I_scale, P, smooth_beta=True):
 
    ### known parameters
    T = tf.constant(100.0, dtype=tf.float32)
    sigma = tf.constant(0.25, dtype=tf.float32)
    gamma = tf.constant(0.25, dtype=tf.float32)
    N = tf.constant(50000.0, dtype=tf.float32)
 
    ### Migration matrix
    M = np.full((P, P), 0.01 / (P - 1))
    np.fill_diagonal(M, -0.01)
    M_tensor = tf.constant(M, dtype=tf.float32)
 
    S_list, E_list, I_list, R_list, beta_list  = [], [], [], [], []
    dS_dt_list, dE_dt_list, dI_dt_list, dR_dt_list = [], [], [], []
 
    ### For each patch compute PINN outputs
    for i in range(P):
        
        patch = tf.fill([tf.shape(t_col)[0], 1], i)
        patch = tf.cast(patch, tf.int32)
 
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t_col)
            S, E, I, R, b = net([t_col, patch])
 
        dS_dt = tape.gradient(S, t_col)
        dE_dt = tape.gradient(E, t_col)
        dI_dt = tape.gradient(I, t_col)
        dR_dt = tape.gradient(R, t_col)
        del tape
 
        S_list.append(S); E_list.append(E)
        I_list.append(I); R_list.append(R)
        beta_list.append(b)
        dS_dt_list.append(dS_dt); dE_dt_list.append(dE_dt)
        dI_dt_list.append(dI_dt); dR_dt_list.append(dR_dt)
 
    S = tf.concat(S_list, axis=1)   
    E = tf.concat(E_list, axis=1)
    I = tf.concat(I_list, axis=1)
    R = tf.concat(R_list, axis=1)
    beta  = tf.concat(beta_list,  axis=1)
    dS_dt = tf.concat(dS_dt_list, axis=1)
    dE_dt = tf.concat(dE_dt_list, axis=1)
    dI_dt = tf.concat(dI_dt_list, axis=1)
    dR_dt = tf.concat(dR_dt_list, axis=1)
 
    ### Vectorised ODE with migration
    lambda_ = beta * I / N
 
    ### Compute ODE residuals
    dS = T * (-lambda_ * S + S @ tf.transpose(M_tensor))
    dE = T * (lambda_ * S - sigma * E + E @ tf.transpose(M_tensor))
    dI = T * (sigma * E - gamma * I + I @ tf.transpose(M_tensor))
    dR = T * (gamma * I + R @ tf.transpose(M_tensor))
 
    ### ODE loss
    ODE_loss = (
        tf.reduce_mean(tf.square(dS_dt - dS)) +
        tf.reduce_mean(tf.square(dE_dt - dE)) +
        tf.reduce_mean(tf.square(dI_dt - dI)) +
        tf.reduce_mean(tf.square(dR_dt - dR))
    )
 
    ### IC loss
    t_zero = tf.constant([[0.0]], dtype=tf.float32)
    IC_loss = 0.0
    IC = list(zip(S0, E0, I0, R0))
 
    for i, (s0, e0, i0, r0) in enumerate(IC):
        
        patch = tf.constant([[i]], dtype=tf.int32)
        S0p, E0p, I0p, R0p, _ = net([t_zero, patch])
        IC_loss += tf.reduce_mean(
            (S0p - s0)**2 + (E0p - e0)**2 +
            (I0p - i0)**2 + (R0p - r0)**2
        )
 
    ### Conservation loss
    conservation_loss = tf.reduce_mean(tf.square(S + E + I + R - 1.0))
 
    ### Data loss
    data_loss = 0.0
    for i in range(P):
        patch = tf.fill([tf.shape(t_data_loss)[0], 1], i)
        patch = tf.cast(patch, tf.int32)
        _, _, I_pred, _, _ = net([t_data_loss, patch])
        data_loss += tf.reduce_mean(((I_pred - I_data_list[i]) / I_scale)**2)
 
    if smooth_beta and P > 1:
        beta_smooth_loss = tf.reduce_mean(tf.square(beta[:, 1:] - beta[:, :-1]))
    else:
        beta_smooth_loss = tf.constant(0.0, dtype=tf.float32)
 
    ### Total loss
    total_loss = (
        1.0 * data_loss +
        0.1 * ODE_loss +
        1.0 * IC_loss +
        1.0 * conservation_loss +
        0.1 * beta_smooth_loss
    )
 
    return total_loss, {
        "data_loss":         data_loss,
        "IC_loss":           IC_loss,
        "conservation_loss": conservation_loss,
        "ODE_loss":          ODE_loss,
        "beta_smooth_loss":  beta_smooth_loss,
    }
 
### Optimiser and collocation points
### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
opt = Adam(learning_rate=0.001)
 
n_collocation = 1000
t_col_tensor = tf.convert_to_tensor(
    np.linspace(0, 1, n_collocation).reshape(-1, 1), dtype=tf.float32
)
 
I_train_list = [tf.convert_to_tensor(x, dtype=tf.float32) for x in I_train_list]
I_test_list = [tf.convert_to_tensor(x, dtype=tf.float32) for x in I_test_list]
 
### Training loop
train_loss_record = []
test_loss_record  = []
 
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss, loss_dict = loss_function(
            t_col_tensor, t_train, I_train_list, model, I_scale, P, smooth_beta=True
        )
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss, loss_dict
 
@tf.function
def test_step(t_col, t_data, I_data_list):
    total_loss, loss_dict = loss_function(
        t_col, t_data, I_data_list, model, I_scale, P, smooth_beta=True
    )
    return total_loss, loss_dict
 
print("Starting training...")
### 50,000 iterations - Qian et al. 2025
for itr in range(50000):
    train_loss, train_loss_dict = train_step()
    test_loss,  test_loss_dict  = test_step(t_col_tensor, t_test, I_test_list)
    train_loss_record.append(float(train_loss))
    test_loss_record.append(float(test_loss))
 
    ### Printing combined loss for every patch
    if itr % 1000 == 0:
        print(
            f"Iteration {itr}\n"
            f"Train Loss: {float(train_loss):.6f}, "
            f"Test Loss:  {float(test_loss):.6f}\n"
            f"Data: {float(train_loss_dict['data_loss']):.6f}, "
            f"IC: {float(train_loss_dict['IC_loss']):.6f}, "
            f"Conservation: {float(train_loss_dict['conservation_loss']):.6f}, "
            f"ODE: {float(train_loss_dict['ODE_loss']):.6f}, "
            f"Beta Smooth: {float(train_loss_dict['beta_smooth_loss']):.6f}"
        )
 
def to_numpy_flat(arr):
    if hasattr(arr, 'numpy'):
        return arr.numpy().flatten()
    return arr.flatten()
 
t_data_np  = to_numpy_flat(t_data)
t_train_np = to_numpy_flat(t_train)
t_test_np  = to_numpy_flat(t_test)
 
def predict_patch(net, t_col, patch_idx):
    n = tf.shape(t_col)[0]
    # FIX 4: int32 patch for prediction too
    patch_tensor = tf.fill([n, 1], patch_idx)
    patch_tensor = tf.cast(patch_tensor, tf.int32)
    S, E, I, R, beta = net([t_col, patch_tensor])
    return S, E, I, R, beta
 
t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
 
# Predict and collect numpy arrays for all P patches
I_pred_np  = []
I_train_np = []
I_test_np  = []
 
for i in range(P):
    _, _, I_pred, _, _ = predict_patch(model, t_tensor, patch_idx=i)
    I_pred_np.append(to_numpy_flat(I_pred))
    I_train_np.append(to_numpy_flat(I_train_list[i]))
    I_test_np.append(to_numpy_flat(I_test_list[i]))
 
### Plot training loss
plt.figure(figsize=(10, 8))
plt.plot(train_loss_record, label='Train loss')
plt.plot(test_loss_record,  label='Test loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PINN_training_loss.png'))
plt.show()
 
### Distinct colours for each patch
patch_colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#9b5de5"]
 
### Plot PINN predictions vs observed — all P patches
plt.figure(figsize=(14, 6))
for i in range(P):
    c = patch_colors[i]
    plt.plot(t_data_np,  I_pred_np[i],  color=c, linewidth=2,
             label=f'Patch {i+1} (PINN)')
    plt.plot(t_train_np, I_train_np[i], color=c, linewidth=2, linestyle='-',
             alpha=0.5, label=f'Patch {i+1} (observed – train)')
    plt.plot(t_test_np,  I_test_np[i],  color=c, linewidth=2, linestyle='--',
             alpha=0.5, label=f'Patch {i+1} (observed – test)')
 
plt.axvline(x=t_train_np[-1], color='gray', linestyle='--', label='Train/Test Split')
plt.xlabel('Time')
plt.ylabel('Infected (normalised)')
plt.title('PINN metapopulation — 5 patch')
plt.legend(ncol=2, fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PINN_metapopulation.png'))
plt.show()