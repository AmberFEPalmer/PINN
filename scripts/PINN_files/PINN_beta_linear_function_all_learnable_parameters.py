import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

data_folder = os.path.join("..", "..", "data")
output_dir  = "../../png_files"

### True beta(t) from the SEIR data-generation script — used for comparison plot only
### (PINN has no knowledge of these knots during training)
gamma_true = 0.25
def beta_t_true(t):
    t_knots  = np.array([0, 10, 22, 40, 57, 70, 78, 100, 140])
    Rt_knots = np.array([2.0, 2.4, 2.8, 2.0, 0.7, 0.9, 1.0, 1.0, 1.0])
    return np.interp(t, t_knots, Rt_knots * gamma_true)

### Load data
data_path = os.path.join(data_folder, "SEIR_140_days_time_varying_beta.csv")
data = pd.read_csv(data_path)

### Time is already normalised to [0, 1] in the CSV
t_data = data["time"].values.reshape(-1, 1)
I_data = data["I"].values.reshape(-1, 1)

### Train/test split
N_obs = len(I_data)
split = int(0.9 * N_obs)
t_train = t_data[:split]
I_train = I_data[:split]
t_test  = t_data[split:]
I_test  = I_data[split:]

I_scale = tf.constant(float(I_train.max()), dtype=tf.float32)

t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
I_train_tensor = tf.convert_to_tensor(I_train, dtype=tf.float32)
t_test_tensor  = tf.convert_to_tensor(t_test,  dtype=tf.float32)
I_test_tensor  = tf.convert_to_tensor(I_test,  dtype=tf.float32)

### Initial conditions (normalised, matching data-generation script)
### S0=99950, E0=0, I0=50, R0=0, N=100000
S0_fixed = tf.constant(99950 / 100000, dtype=tf.float32)
E0_fixed = tf.constant(0.0, dtype=tf.float32)
I0_fixed = tf.constant(50 / 100000, dtype=tf.float32)
R0_fixed = tf.constant(0.0, dtype=tf.float32)

### PINN architecture
def create_pinn_model():
    t_input = Input(shape=(1,), name='time_input')

    ### Shared SEIR trunk — 3 hidden layers, 50 neurons, tanh (Qian et al. 2025)
    seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir)
    seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir)

    ### SEIR outputs — softplus keeps values non-negative
    S = Dense(1, activation='softplus', name='S')(seir)
    E = Dense(1, activation='softplus', name='E')(seir)
    I = Dense(1, activation='softplus', name='I')(seir)
    R = Dense(1, activation='softplus', name='R')(seir)

    ### Separate beta(t) sub-network — own trunk so it can learn an independent
    ### time-varying curve without being constrained by the SEIR compartment branch
    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_h)
    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_h)

    ### Parameterise as exp(log_beta) so beta is always positive
    log_beta = Dense(1, activation=None, name='log_beta')(beta_h)
    beta     = Lambda(lambda x: tf.exp(x), name='beta')(log_beta)

    ### Separate gamma(t) sub-network — learnable time-varying recovery rate
    gamma_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    gamma_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(gamma_h)
    log_gamma = Dense(1, activation=None, name='log_gamma')(gamma_h)
    gamma     = Lambda(lambda x: tf.exp(x), name='gamma')(log_gamma)

    ### Separate sigma(t) sub-network — learnable time-varying incubation rate
    sigma_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    sigma_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(sigma_h)
    log_sigma = Dense(1, activation=None, name='log_sigma')(sigma_h)
    sigma     = Lambda(lambda x: tf.exp(x), name='sigma')(log_sigma)

    return Model(inputs=t_input, outputs=[S, E, I, R, beta, gamma, sigma])

tf.keras.backend.clear_session()
model = create_pinn_model()
model.summary()

### Loss
def loss_function(t_col, t_data_loss, I_data_loss, net, I_scale):
    if len(t_col.shape) == 1:
        t_col = tf.reshape(t_col, (-1, 1))
    if not isinstance(t_data_loss, tf.Tensor):
        t_data_loss = tf.convert_to_tensor(t_data_loss, dtype=tf.float32)
    if not isinstance(I_data_loss, tf.Tensor):
        I_data_loss = tf.convert_to_tensor(I_data_loss, dtype=tf.float32)
    if len(t_data_loss.shape) == 1:
        t_data_loss = tf.reshape(t_data_loss, (-1, 1))
    if len(I_data_loss.shape) == 1:
        I_data_loss = tf.reshape(I_data_loss, (-1, 1))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, E, I, R, beta, gamma, sigma = net(t_col)

    dS_dt = tape.gradient(S, t_col)
    dE_dt = tape.gradient(E, t_col)
    dI_dt = tape.gradient(I, t_col)
    dR_dt = tape.gradient(R, t_col)

    d_beta_dt  = tape.gradient(beta,  t_col)
    beta_smooth_loss = tf.reduce_mean(tf.square(d_beta_dt))

    del tape

    ### ODE residuals — time is normalised so multiply by T=140
    T = tf.constant(140.0, dtype=tf.float32)
    dS_dt_physics = T * (-beta  * S * I)
    dE_dt_physics = T * ( beta  * S * I - sigma * E)
    dI_dt_physics = T * ( sigma * E     - gamma * I)
    dR_dt_physics = T * ( gamma * I)

    loss_S = tf.reduce_mean(tf.square(dS_dt - dS_dt_physics))
    loss_E = tf.reduce_mean(tf.square(dE_dt - dE_dt_physics))
    loss_I = tf.reduce_mean(tf.square(dI_dt - dI_dt_physics))
    loss_R = tf.reduce_mean(tf.square(dR_dt - dR_dt_physics))
    ODE_loss = loss_S + loss_E + loss_I + loss_R

    ### Initial condition loss
    t_zero = tf.constant([[0.0]], dtype=tf.float32)
    S_0, E_0, I_0, R_0, _, _, _ = net(t_zero)
    IC_loss = (
        tf.square(S_0 - S0_fixed) +
        tf.square(E_0 - E0_fixed) +
        tf.square(I_0 - I0_fixed) +
        tf.square(R_0 - R0_fixed)
    )
    IC_loss = tf.reduce_mean(IC_loss)

    ### Conservation: S + E + I + R = 1 at all collocation points
    S_c, E_c, I_c, R_c, _, _, _ = net(t_col)
    conservation_loss = tf.reduce_mean(tf.square(S_c + E_c + I_c + R_c - 1.0))

    ### Data loss on I only (normalised by peak I)
    _, _, I_pred, _, _, _, _ = net(t_data_loss)
    data_loss = tf.reduce_mean(tf.square((I_pred - I_data_loss) / I_scale))

    ### Total loss
    total_loss = (
        1.0  * data_loss         +
        1.0  * IC_loss           +
        1.0  * conservation_loss +
        0.1  * ODE_loss
    )

    return total_loss, {
        "data_loss": data_loss,
        "IC_loss": IC_loss,
        "conservation_loss": conservation_loss,
        "ODE_loss": ODE_loss,
        "beta_smooth_loss": beta_smooth_loss,
    }

### Optimiser
optm = Adam(learning_rate=0.001)

n_collocation = 1000
t_col_uniform = np.linspace(0, 1, n_collocation).reshape(-1, 1)
t_col_tensor = tf.convert_to_tensor(t_col_uniform, dtype=tf.float32)

### Training
@tf.function
def train_step(t_col, t_data, I_data):
    with tf.GradientTape() as tape:
        total_loss, loss_dict = loss_function(t_col, t_data, I_data, model, I_scale)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optm.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_dict

@tf.function
def test_step(t_col, t_data, I_data):
    return loss_function(t_col, t_data, I_data, model, I_scale)

### Training loop
train_loss_record = []
test_loss_record  = []

print("Starting training...")
for itr in range(50000):
    train_loss, train_loss_dict = train_step(t_col_tensor, t_train_tensor, I_train_tensor)
    train_loss_record.append(float(train_loss))

    test_loss, test_loss_dict = test_step(t_col_tensor, t_test_tensor, I_test_tensor)
    test_loss_record.append(float(test_loss))

    if itr % 1000 == 0:
        print(
            f"Iteration {itr} | "
            f"Train: {float(train_loss):.6f}  Test: {float(test_loss):.6f} | "
            f"Data: {float(train_loss_dict['data_loss']):.6f}  "
            f"IC: {float(train_loss_dict['IC_loss']):.6f}  "
            f"Cons: {float(train_loss_dict['conservation_loss']):.6f}  "
            f"ODE: {float(train_loss_dict['ODE_loss']):.6f}  "
            f"β-smooth: {float(train_loss_dict['beta_smooth_loss']):.6f}"
        )

### Predictions for plotting
t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
_, _, I_pred, _, _, _, _ = model(t_tensor)

def to_np(arr):
    return arr.numpy().flatten() if hasattr(arr, 'numpy') else arr.flatten()

t_data_np  = to_np(t_data)
I_pred_np  = to_np(I_pred)
t_train_np = to_np(t_train)
I_train_np = to_np(I_train)
t_test_np  = to_np(t_test)
I_test_np  = to_np(I_test)

### Training loss plot
plt.figure(figsize=(10, 5))
plt.plot(train_loss_record, label='Train loss')
plt.plot(test_loss_record,  label='Test loss', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training loss — time-varying β PINN (90/10 split)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PINN_tv_beta_training_loss.png'), dpi=150)
plt.show()

### Predicted vs observed infections 
plt.figure(figsize=(14, 6))
plt.plot(t_data_np,  I_pred_np,   color='#ff7ee3', linewidth=2,  label='I — PINN prediction')
plt.plot(t_train_np, I_train_np,  color='#004F94', linewidth=2,  label='I — data (train)')
plt.plot(t_test_np,  I_test_np,   color='#004F94', linewidth=2,  linestyle='--', label='I — data (test)')
plt.axvline(x=t_train_np[-1], color='gray', linestyle='--', label='Train/Test split')
plt.xlabel('Normalised time')
plt.ylabel('Infected (normalised)')
plt.title('PINN observed vs predicted — β as a linear function (90/10 split)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PINN_forecasting_linear_function_all_learnable_parameters_I_pred.png'), dpi=150)
plt.show()

### Beta
t_plot_norm = np.linspace(0.0, 1.0, 500)
t_plot_days = t_plot_norm * 140.0 # un-normalise for true beta lookup

t_plot_tensor = tf.convert_to_tensor(t_plot_norm.reshape(-1, 1), dtype=tf.float32)
_, _, _, _, beta_pred, gamma_pred, sigma_pred = model.predict(t_plot_tensor)

beta_true_vals = beta_t_true(t_plot_days)

plt.figure(figsize=(10, 5))
plt.plot(t_plot_norm, beta_pred.flatten(), color='#ff7ee3', linewidth=2, label='β(t) — PINN estimate')
plt.plot(t_plot_norm, beta_true_vals, color='#004F94', linewidth=2, linestyle='--', label='β(t) — true (data-generating)')
plt.xlabel('Normalised time')
plt.ylabel('β(t)')
plt.title('Estimated β(t) vs true β(t) — linear function (90/10 split)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'PINN_predict_beta_linear_function_all_learnable_parameters.png'), dpi=150)
plt.show()

