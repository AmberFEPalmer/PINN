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
 
### Load data from data_processing.py
### t_data_study.npy is already normalised to [0, 1] by data_processing.py
### Load the raw (un-normalised) time in days separately so we have the true
### physical time scale needed to correctly weight the ODE residuals.
t_data_norm = np.load("../../data/t_data_study.npy").reshape(-1, 1)   # [0, 1]
I_data       = np.load("../../data/I_data_study.npy").reshape(-1, 1)
 
### Cap to 93 weeks to match study period (July 2020 – April 2022)
t_data_norm = t_data_norm[:93]
I_data      = I_data[:93]
 
### The normalised array already spans [0, 1] — no further normalisation needed.
N_total_points = len(t_data_norm)
t_data_weeks   = np.linspace(0.0, N_total_points - 1, N_total_points).reshape(-1, 1)

### Define PINN
### L2 regularisation for hidden layers -> helps to prevent overfitting
### Add penalty proportional to the sum of squared coefficients to the loss function
### Reduce model complexity, penalise large weights
### https://keras.io/api/layers/regularizers/
### https://developers.google.com/machine-learning/crash-course/overfitting/regularization
def create_pinn_model():
    t_input = Input(shape=(1,), name='time_input')
 
    ### SEIR —> 3 hidden layers, 50 neurons, tanh activation
    seir = Dense(50, activation='tanh')(t_input)
    seir = Dense(50, activation='tanh')(seir)
    seir = Dense(50, activation='tanh')(seir)
 
    ### SEIR compartment outputs (softplus keeps values non-negative)
    S = Dense(1, activation='softplus', name='S')(seir)
    E = Dense(1, activation='softplus', name='E')(seir)
    I = Dense(1, activation='softplus', name='I')(seir)
    R = Dense(1, activation='softplus', name='R')(seir)
 
    ### Separate beta sub-network —> 3 hidden layers, 50 neurons, tanh
    beta_h = Dense(50, activation='tanh')(t_input)
    beta_h = Dense(50, activation='tanh')(beta_h)
    beta_h = Dense(50, activation='tanh')(beta_h)
    
    beta   = Dense(1,   activation=None, name='beta')(beta_h)
 
    return Model(inputs=t_input, outputs=[S, E, I, R, beta])
 
### Define physics-informed loss
def compute_loss(t_col, t_data_loss, I_data_loss, net, t_max_weeks, I0, E0, S0, R0=0.0):
 
    ### Ensure column vectors
    if len(t_col.shape) == 1:
        t_col = tf.reshape(t_col, (-1, 1))
 
    t_data_loss = tf.cast(tf.reshape(t_data_loss, (-1, 1)), tf.float32)
    I_data_loss = tf.cast(tf.reshape(I_data_loss, (-1, 1)), tf.float32)
 
    ### Physics loss at collocation points
    ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, E, I, R, beta = net(t_col)
 
    ### Fixed epidemiological parameters (Qian et al. 2025)
    sigma = tf.constant(0.25, dtype=tf.float32)   # incubation rate  
    gamma = tf.constant(0.25, dtype=tf.float32)   # recovery rate 
 
    ### Compute derivatives w.r.t. normalised time t_norm ∈ [0, 1]
    ### d/dt_physical = (1 / t_max_weeks) * d/dt_norm
    ### So: d/dt_norm = t_max_weeks * d/dt_physical
    ### The ODE is written in physical time 
    T = tf.cast(t_max_weeks, tf.float32)
 
    dS_dt = tape.gradient(S, t_col)
    dE_dt = tape.gradient(E, t_col)
    dI_dt = tape.gradient(I, t_col)
    dR_dt = tape.gradient(R, t_col)
    d_beta_dt = tape.gradient(beta, t_col)
    del tape
 
    ### Physics-informed residuals in normalised time
    ode_loss = (
        tf.reduce_mean(tf.square(dS_dt - T * (-beta * S * I))) +
        tf.reduce_mean(tf.square(dE_dt - T * (beta * S * I - sigma * E))) +
        tf.reduce_mean(tf.square(dI_dt - T * (sigma * E - gamma * I))) +
        tf.reduce_mean(tf.square(dR_dt - T * (gamma * I)))
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
    total = (1.0  * data_loss +
             0.5  * ode_loss +
             1.0  * ic_loss +
             1.0  * conservation_loss)
 
    return total, {
        "data_loss": data_loss,
        "IC_loss": ic_loss,
        "conservation_loss": conservation_loss,
        "ODE_loss": ode_loss,
        "beta_smooth_loss": beta_smooth_loss,
    }
 
### Single window training
def train_window(t_train_norm, I_train, t_max_weeks, n_iter=50_000,
                 warm_start_model=None):
    model = create_pinn_model()
 
    ### Warm-start - copy weights from the previous window's trained model.
    if warm_start_model is not None:
        model.set_weights(warm_start_model.get_weights())
 
    ### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
    ### Optimiser and collocation points 
    optm = Adam(learning_rate=0.001)
 
    I0 = float(I_train[0])
    E0 = 2.0 * I0
    S0 = 1.0 - E0 - I0
    R0 = 0.0

    ### Collocation points for physics
    n_collocation = 1000
    t_col_tensor = tf.convert_to_tensor(
        np.linspace(0, 1, n_collocation).reshape(-1, 1), dtype=tf.float32
    )

    t_tr = tf.convert_to_tensor(t_train_norm, dtype=tf.float32)
    I_tr = tf.convert_to_tensor(I_train, dtype=tf.float32)
 
    ### Training step with @tf.function for performance
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            loss, _ = compute_loss(t_col_tensor, t_tr, I_tr, model,
                                   t_max_weeks, I0, E0, S0, R0)
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
 
### Storage dictionaries
all_predictions  = {}   ### predicted I
all_observations = {}   ### observed  I
all_beta         = {}   ### predicted beta
 
### Initialise model to None — first window trains from random initialisation,
### all subsequent windows warm-start from the previous window's model.
model = None
 
### Rolling window loop
for train_end in range(First_train_weeks, N_total_points - Forecast_horizon + 1):
 
    ### Normalised times from the saved array (already [0, 1] over full study)
    t_tr_norm_global = t_data_norm[:train_end]        # shape (train_end, 1)
    I_tr_np = I_data[:train_end]
 
    ### Re-normalise so the window spans [0, 1].
    ### The network sees a consistent [0,1] input each window.
    t_win_min = float(t_tr_norm_global[0])
    t_win_max = float(t_tr_norm_global[-1])
    t_tr_norm = (t_tr_norm_global - t_win_min) / (t_win_max - t_win_min)
 
    ### Physical duration of the training window in weeks
    t_max_weeks = float(t_data_weeks[train_end - 1]) 
 
    ### Model iterations
    n_iter = 50_000 
 
    print(f"Training on weeks 1–{train_end} "
          f"| forecasting weeks {train_end+1}–{train_end+Forecast_horizon} "
          f"| T = {t_max_weeks:.1f} weeks | iters = {n_iter}"
          f"{'  [warm start]' if model is not None else '  [cold start]'}")
 
    ### Train the PINN — pass previous model for warm-starting
    model = train_window(t_tr_norm, I_tr_np, t_max_weeks=t_max_weeks,
                         n_iter=n_iter, warm_start_model=model)
 
    ### Generate 1–4 week ahead forecasts
    for h in range(1, Forecast_horizon + 1):
        forecast_idx = train_end + h
        if forecast_idx >= N_total_points:
            continue
 
        ### Normalise the forecast time into the same window-local [0, 1] scale.
        t_fc_global = float(t_data_norm[forecast_idx])
        t_fc_norm   = (t_fc_global - t_win_min) / (t_win_max - t_win_min)
        ### Note: t_fc_norm > 1 for all forecast points — this is expected and
        ### correct; the network extrapolates beyond its training horizon.
 
        t_fc_tensor = tf.constant([[t_fc_norm]], dtype=tf.float32)
        _, _, I_pred, _, beta_pred = model(t_fc_tensor)
 
        pred_val = float(np.clip(I_pred.numpy()[0, 0],    0.0, 1.0))
        beta_val = float(np.clip(beta_pred.numpy()[0, 0], 0.0, None))
 
        key = (train_end, forecast_idx)
        all_predictions[key] = pred_val
        all_beta[key] = beta_val
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
            t_val = t_data_norm[forecast_idx, 0]
            horizon_results[h]["t"].append(t_val)
            horizon_results[h]["pred"].append(all_predictions[key])
            horizon_results[h]["obs"].append(all_observations[forecast_idx])
        if key in all_beta:
            horizon_beta[h]["t"].append(t_data_norm[forecast_idx, 0])
            horizon_beta[h]["beta"].append(all_beta[key])
 
### Visualise observed vs predicted infections — all horizons on one plot
N_UK = 56_000_000

# Four shades of pink from light to deep
pink_shades = ["#ffb3e6", "#ff69b4", "#ff1493", "#c71585"]

fig, ax = plt.subplots(figsize=(14, 6))

# Plot observed once (use horizon 1 as reference — all horizons share the same obs)
t_ref = np.array(horizon_results[1]["t"])
obs_ref = np.array(horizon_results[1]["obs"]) * N_UK
ax.plot(t_ref, obs_ref, color="#004F94", lw=2, label="Observed", zorder=5)

# Overlay all four forecast horizons in different pinks
for h, color in zip(range(1, 5), pink_shades):
    t_vals   = np.array(horizon_results[h]["t"])
    pred_arr = np.array(horizon_results[h]["pred"]) * N_UK
    ax.plot(t_vals, pred_arr, color=color, lw=1.5, label=f"{h}-week forecast")

ax.set_title("SEIR-PINN rolling window forecasts (1–4 weeks ahead)", fontsize=14)
ax.set_xlabel("Normalised time (study period)")
ax.set_ylabel("Infected (count)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("rolling_window_forecasts_combined.png", dpi=150)
plt.show()
 
### Plot R(t) = beta(t) / gamma over time
gamma = 0.25  # matches the value used in training

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for h, ax in zip(range(1, 5), axes.flatten()):
    t_vals   = np.array(horizon_beta[h]["t"])
    R_arr    = np.array(horizon_beta[h]["beta"]) / gamma  # R(t) = β(t) / γ

    ax.plot(t_vals, R_arr, color="#ff7ee3", lw=1.5, label=f"R(t) — {h}-week ahead")
    ax.axhline(y=1.0, color="gray", lw=1, linestyle="--", label="R = 1 threshold")
    ax.set_title(f"{h}-week-ahead R(t)")
    ax.set_xlabel("Normalised time (study period)")
    ax.set_ylabel("R(t) = β(t) / γ")
    ax.legend()
    ax.grid(True)

plt.suptitle("SEIR-PINN rolling window: effective reproduction number R(t) (1–4 weeks ahead)", fontsize=14)
plt.tight_layout()
plt.savefig("rolling_window_Rt.png", dpi=150)
plt.show()
 
### Model evaluation
print("Forecast evaluation (MAE and RMSE):")
for h in range(1, 5):
    pred = np.array(horizon_results[h]["pred"])
    obs  = np.array(horizon_results[h]["obs"])
    mae  = np.mean(np.abs(pred - obs))
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    print(f"  {h}-week ahead  |  MAE = {mae:.6f}  |  RMSE = {rmse:.6f}")