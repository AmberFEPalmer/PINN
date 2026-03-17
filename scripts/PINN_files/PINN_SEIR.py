import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

### Load data from data_processing.py
t_data = np.load("../../data/t_data_study.npy").reshape(-1, 1)
I_data = np.load("../../data/I_data_study.npy").reshape(-1, 1)

### Cap to 93 weeks to match study period (July 2020 – April 2022)
t_data = t_data[:93]
I_data = I_data[:93]

N_total_points = len(t_data)
print(f"Total weekly points loaded: {N_total_points}")
assert 85 < N_total_points <= 93, \
    f"Expected ~93 weekly points, got {N_total_points} — check COVID_Data.py"

### Define PINN
### L2 regularisation for hidden layers 
### https://keras.io/api/layers/regularizers/
### https://developers.google.com/machine-learning/crash-course/overfitting/regularization
def create_pinn_model():
    t_input = Input(shape=(1,), name='time_input')

    ### SEIR —> 3 hidden layers, 100 neurons, tanh activation
    seir = Dense(100, activation='tanh')(t_input)
    seir = Dense(100, activation='tanh')(seir)
    seir = Dense(100, activation='tanh')(seir)

    ### SEIR compartment outputs (softplus keeps values non-negative)
    S = Dense(1, activation='softplus', name='S')(seir)
    E = Dense(1, activation='softplus', name='E')(seir)
    I = Dense(1, activation='softplus', name='I')(seir)
    R = Dense(1, activation='softplus', name='R')(seir)

    ### Separate beta sub-network —> 3 hidden layers, 100 neurons, tanh
    beta_h = Dense(100, activation='tanh')(t_input)
    beta_h = Dense(100, activation='tanh')(beta_h)
    beta_h = Dense(100, activation='tanh')(beta_h)
    beta   = Dense(1,   activation='softplus', name='beta')(beta_h)

    return Model(inputs=t_input, outputs=[S, E, I, R, beta])

### Define physics informed loss
def compute_loss(t_col, t_data_loss, I_data_loss, net, t_max, I0, E0, S0, R0=0.0):

     ### if t_col is a 1D array it is reshaped to a column vector
    if len(t_col.shape) == 1:
        t_col = tf.reshape(t_col, (-1, 1))
        
    ### Convert data to tensors and ensure they are column vectors
    t_data_loss = tf.cast(tf.reshape(t_data_loss, (-1, 1)), tf.float32)
    I_data_loss = tf.cast(tf.reshape(I_data_loss, (-1, 1)), tf.float32)

    ### Physics loss at collocation points
    ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, E, I, R, beta = net(t_col)

    ### Define parameters which don't vary over time
    ### Following what was done in Qian et al. 2025
    sigma = tf.constant(0.25, dtype=tf.float32)   # incubation rate  (Qian et al. 2025)
    gamma = tf.constant(0.25, dtype=tf.float32)   # recovery rate    (Qian et al. 2025)

    ### Compute derivatives e.g. dS/dt
    dS_dt = tape.gradient(S, t_col)
    dE_dt = tape.gradient(E, t_col)
    dI_dt = tape.gradient(I, t_col)
    dR_dt = tape.gradient(R, t_col)
    d_beta_dt = tape.gradient(beta, t_col)
    del tape

    beta_smooth_loss = tf.reduce_mean(tf.square(d_beta_dt))

    ### Physics-informed loss - mean squared error
    ode_loss = (
        tf.reduce_mean(tf.square(dS_dt - t_max * (-beta * S * I))) +
        tf.reduce_mean(tf.square(dE_dt - t_max * (beta * S * I - sigma * E))) +
        tf.reduce_mean(tf.square(dI_dt - t_max * (sigma * E - gamma * I))) +
        tf.reduce_mean(tf.square(dR_dt - t_max * (gamma * I)))
    )

    ### Initial condition loss
    t_zero = tf.constant([[0.0]], dtype=tf.float32)
    S_0, E_0, I_0, R_0_pred, _ = net(t_zero)
    ic_loss = tf.reduce_mean(
        tf.square(S_0 - S0) + tf.square(E_0 - E0) +
        tf.square(I_0 - I0) + tf.square(R_0_pred - R0)
    )

    ### Conservation loss (S + E + I + R = 1)
    S_c, E_c, I_c, R_c, _ = net(t_col)
    conservation_loss = tf.reduce_mean(tf.square(S_c + E_c + I_c + R_c - 1.0))

    ### Data loss 
    _, _, I_pred, _, _ = net(t_data_loss)
    data_loss = tf.reduce_mean(tf.square(I_pred - I_data_loss))

    ### Total loss 
    total = (100.0   * data_loss +
             1.0   * ode_loss +
             1.0 * ic_loss +
             5.0 * conservation_loss +
             1.0 * beta_smooth_loss)

    return total, {
        "data_loss":         data_loss,
        "IC_loss":           ic_loss,
        "conservation_loss": conservation_loss,
        "ODE_loss":          ode_loss,
        "beta_smooth_loss":  beta_smooth_loss,
    }

### Single window training
def train_window(t_train, I_train, t_max, n_iter=100_000):
    model = create_pinn_model()
    optm  = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

    I0 = float(I_train[0])
    E0 = 2.0 * I0
    S0 = 1.0 - E0 - I0
    R0 = 0.0

    ### 1000 collocation points evenly covering the training window
    t_col_np     = np.linspace(0.0, float(t_train[-1]), 1000).reshape(-1, 1)
    
    ### Convert arrays to tensors
    t_col_tensor = tf.convert_to_tensor(t_col_np, dtype=tf.float32)
    t_tr = tf.convert_to_tensor(t_train,  dtype=tf.float32)
    I_tr = tf.convert_to_tensor(I_train,  dtype=tf.float32)

    ### Training step function with @tf.function for performance
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            loss, _ = compute_loss(t_col_tensor, t_tr, I_tr, model,
                                   t_max, I0, E0, S0, R0)
        grads = tape.gradient(loss, model.trainable_variables)
        optm.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for itr in range(n_iter):
        loss = step()
        if itr % 5000 == 0:
            print(f"  iter {itr:5d}  loss {float(loss):.6f}")

    return model

### Rolling window forecasting
First_train_weeks = 17   ### initial window uses weeks 1–17
Forecast_horizon  = 4    ### forecast 1, 2, 3, 4 weeks ahead
N_ITER = 100_000

### Storage dictionaries
all_predictions  = {}   ### predicted I 
all_observations = {}   ### observed  I 
all_beta         = {}   ### predicted beta

### rolling window loop
for train_end in range(First_train_weeks, N_total_points - Forecast_horizon + 1):

    ### Slice and normalise training data to [0, 1]
    t_tr_np   = t_data[:train_end]
    I_tr_np   = I_data[:train_end]
    t_tr_max  = float(t_tr_np[-1])
    t_tr_norm = t_tr_np / t_tr_max

    print(f"Training on weeks 1–{train_end}"
          f"| forecasting weeks {train_end+1}–{train_end+Forecast_horizon}")

    ### Train the PINN
    model = train_window(t_tr_norm, I_tr_np, t_max=t_tr_max, n_iter=N_ITER)

    ### Generate 1–4 week ahead forecasts
    for h in range(1, Forecast_horizon + 1):
        forecast_idx = train_end + h   
        if forecast_idx >= N_total_points:
            continue

        ### Normalise forecast time using the same scale as training
        t_fc = float(t_data[forecast_idx]) / t_tr_max
        ### Convert to tensor
        t_fc_tensor = tf.constant([[t_fc]], dtype=tf.float32)

        _, _, I_pred, _, beta_pred = model(t_fc_tensor)

        pred_val = float(np.clip(I_pred.numpy()[0, 0],    0.0, 1.0))
        beta_val = float(np.clip(beta_pred.numpy()[0, 0], 0.0, None))   # beta >= 0

        key = (train_end, forecast_idx)
        all_predictions[key]           = pred_val
        all_beta[key]                  = beta_val
        all_observations[forecast_idx] = float(I_data[forecast_idx])


### Collect results per forecast horizon for plotting
horizon_results = {h: {"t": [], "pred": [], "obs": []} for h in range(1, 5)}
horizon_beta = {h: {"t": [], "beta": []} for h in range(1, 5)}

for train_end in range(First_train_weeks, N_total_points - Forecast_horizon + 1):
    for h in range(1, Forecast_horizon + 1):
        forecast_idx = train_end + h
        if forecast_idx >= N_total_points:
            continue
        key = (train_end, forecast_idx)
        if key in all_predictions:
            t_val = t_data[forecast_idx, 0]
            horizon_results[h]["t"].append(t_val)
            horizon_results[h]["pred"].append(all_predictions[key])
            horizon_results[h]["obs"].append(all_observations[forecast_idx])
        if key in all_beta:
            horizon_beta[h]["t"].append(t_data[forecast_idx, 0])
            horizon_beta[h]["beta"].append(all_beta[key])


### Visualising observed vs predicted infections
N_UK = 56_000_000

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for h, ax in zip(range(1, 5), axes.flatten()):
    t_vals   = np.array(horizon_results[h]["t"])
    pred_arr = np.array(horizon_results[h]["pred"]) * N_UK
    obs_arr  = np.array(horizon_results[h]["obs"])  * N_UK

    ax.plot(t_vals, obs_arr,  color="#004F94", lw=1.5, label="Observed")
    ax.plot(t_vals, pred_arr, color="#ff7ee3", lw=1.5, label=f"{h}-week forecast")
    ax.set_title(f"{h}-week-ahead forecast")
    ax.set_xlabel("Normalised time")
    ax.set_ylabel("Infected (count)")
    ax.legend()
    ax.grid(True)

plt.suptitle("SEIR-PINN rolling window forecasts (1–4 weeks ahead)", fontsize=14)
plt.tight_layout()
plt.savefig("rolling_window_forecasts.png", dpi=150)
plt.show()

### Plotting beta over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for h, ax in zip(range(1, 5), axes.flatten()):
    t_vals   = np.array(horizon_beta[h]["t"])
    beta_arr = np.array(horizon_beta[h]["beta"])

    ax.plot(t_vals, beta_arr, color="#ff7ee3", lw=1.5, label=f"β(t) — {h}-week ahead")
    ax.set_title(f"{h}-week-ahead β(t)")
    ax.set_xlabel("Normalised time")
    ax.set_ylabel("β(t)")
    ax.legend()
    ax.grid(True)

plt.suptitle("SEIR-PINN rolling window: β(t) (1–4 weeks ahead)", fontsize=14)
plt.tight_layout()
plt.savefig("rolling_window_beta.png", dpi=150)
plt.show()

### Model evaluation
print("\n── Forecast evaluation ──────────────────────────")
for h in range(1, 5):
    pred = np.array(horizon_results[h]["pred"])
    obs  = np.array(horizon_results[h]["obs"])
    mae  = np.mean(np.abs(pred - obs))
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    print(f"  {h}-week ahead  |  MAE = {mae:.6f}  |  RMSE = {rmse:.6f}")