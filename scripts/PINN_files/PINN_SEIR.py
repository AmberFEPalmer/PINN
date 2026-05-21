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

np.random.seed(42)
tf.random.set_seed(42)

### Population size for England – used for normalisation and unnormalisation
N_val = 56_000_000                                   
N     = tf.constant(float(N_val), dtype=tf.float32)   
N_sq  = N ** 2                                        

### Load data from data_processing.py
### t_data_study.npy is already normalised to [0, 1] by data_processing.py
t_data_norm = np.load("../../data/t_data_study.npy").reshape(-1, 1)
I_data = np.load("../../data/I_data_study.npy").reshape(-1, 1)

### Cap to 93 weeks to match study period (July 2020 – April 2022)
t_data_norm = t_data_norm[:93]
I_data = I_data[:93]

N_total_points = len(t_data_norm)
t_data_weeks = np.linspace(0.0, N_total_points - 1, N_total_points).reshape(-1, 1)

### Define PINN
### L2 regularisation for hidden layers -> helps to prevent overfitting
### Add penalty proportional to the sum of squared coefficients to the loss
### Reduce model complexity, penalise large weights
### https://keras.io/api/layers/regularizers/
def create_pinn_model():
    t_input = Input(shape=(1,), name='time_input')

    ### SEIR trunk — 3 hidden layers, 50 neurons, tanh activation
    seir = Dense(50, activation='tanh')(t_input)
    seir = Dense(50, activation='tanh')(seir)
    seir = Dense(50, activation='tanh')(seir)

    ### SEIR compartment outputs (softplus keeps values non-negative)
    S = Dense(1, activation='softplus', name='S')(seir)
    E = Dense(1, activation='softplus', name='E')(seir)
    I = Dense(1, activation='softplus', name='I')(seir)
    R = Dense(1, activation='softplus', name='R')(seir)

    ### Separate beta sub-network — 3 hidden layers, 50 neurons, tanh
    beta_h = Dense(50, activation='tanh')(t_input)
    beta_h = Dense(50, activation='tanh')(beta_h)
    beta_h = Dense(50, activation='tanh')(beta_h)

    ### softplus keeps beta strictly positive
    beta = Dense(1, activation='softplus', name='beta')(beta_h)

    return Model(inputs=t_input, outputs=[S, E, I, R, beta])

### Loss function
def compute_loss(t_col, t_data_loss, I_data_loss, net,
                 TOTAL_WEEKS, I0, E0, S0, R0=0.0):

    if len(t_col.shape) == 1:
        t_col = tf.reshape(t_col, (-1, 1))

    t_data_loss = tf.cast(tf.reshape(t_data_loss, (-1, 1)), tf.float32)
    I_data_loss = tf.cast(tf.reshape(I_data_loss, (-1, 1)), tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, E, I, R, beta = net(t_col)

    ### Fixed epidemiological parameters (Qian et al. 2025)
    sigma = tf.constant(0.25 * 7, dtype=tf.float32)  # 1.75 per week
    gamma = tf.constant(0.25 * 7, dtype=tf.float32)  # 1.75 per week

    ### d/dt_physical = (1 / TOTAL_WEEKS) * d/dt_norm
    ### So: d/dt_norm = TOTAL_WEEKS * d/dt_physical
    T = tf.cast(TOTAL_WEEKS, tf.float32)

    dS_dt = tape.gradient(S, t_col)
    dE_dt = tape.gradient(E, t_col)
    dI_dt = tape.gradient(I, t_col)
    dR_dt = tape.gradient(R, t_col)
    d_beta_dt = tape.gradient(beta, t_col)

    beta_smooth_loss = tf.reduce_mean(tf.square(d_beta_dt))
    del tape

    ### Convert fractional outputs to absolute counts 
    S_abs = S * N
    E_abs = E * N
    I_abs = I * N
    R_abs = R * N

    ### SEIR equations in absolute counts
    dS_dt_physics = T * (-beta * S_abs * I_abs / N)
    dE_dt_physics = T * ( beta * S_abs * I_abs / N - sigma * E_abs)
    dI_dt_physics = T * ( sigma * E_abs - gamma * I_abs)
    dR_dt_physics = T * ( gamma * I_abs)

    ### ODE loss in absolute counts (network derivatives scaled by N to match)
    ### Divided by N_sq to normalise back to O(1) scale
    ode_loss = (
        tf.reduce_mean(tf.square((dS_dt * N) - dS_dt_physics)) +
        tf.reduce_mean(tf.square((dE_dt * N) - dE_dt_physics)) +
        tf.reduce_mean(tf.square((dI_dt * N) - dI_dt_physics)) +
        tf.reduce_mean(tf.square((dR_dt * N) - dR_dt_physics))
    ) / N_sq

    ### Initial condition loss
    t_zero = tf.constant([[0.0]], dtype=tf.float32)
    S_0, E_0, I_0, R_0_pred, _ = net(t_zero)
    ic_loss = tf.reduce_mean(
        tf.square(S_0 - S0) + tf.square(E_0 - E0) +
        tf.square(I_0 - I0) + tf.square(R_0_pred - R0)
    )

    ### Conservation loss – S+E+I+R must equal N in absolute counts
    ### Divided by N_sq to normalise back to O(1) scale
    conservation_loss = tf.reduce_mean(
        tf.square(S_abs + E_abs + I_abs + R_abs - N)
    ) / N_sq

    ### Data loss — fit sigma*E (incidence) not I (prevalence)
    ### UKHSA data = new cases per week = flow E->I = sigma*E
    _, E_pred, _, _, _ = net(t_data_loss)
    incidence_pred = sigma * E_pred
    data_loss = tf.reduce_mean(tf.square(incidence_pred - I_data_loss))

    ### Total loss
    total = (
        1.0  * data_loss +
        0.01 * ode_loss +
        1.0  * ic_loss +
        1.0 * conservation_loss
    )

    return total, {
        "data_loss": data_loss,
        "IC_loss": ic_loss,
        "conservation_loss": conservation_loss,
        "ODE_loss": ode_loss,
        "beta_smooth_loss": beta_smooth_loss,
    }


### Training function for a single window
def train_window(t_train_norm, I_train, TOTAL_WEEKS, n_iter=50_000,
                 warm_start_model=None):
    model = create_pinn_model()

    ### Warm-start — copy weights from previous window's trained model
    if warm_start_model is not None:
        model.set_weights(warm_start_model.get_weights())

    ### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
    optm = Adam(learning_rate=0.001)

    ### Initial conditions derived from incidence at last training point
    incidence_0 = float(I_train[-1])
    sigma_val = 1.75
    E0 = incidence_0 / sigma_val
    I0 = E0
    R0 = 0.0
    S0 = 1.0 - E0 - I0 - R0

    ### Collocation points for physics
    t_col_tensor = tf.convert_to_tensor(
        np.linspace(0, 1, 1000).reshape(-1, 1), dtype=tf.float32
    )

    t_tr = tf.convert_to_tensor(t_train_norm, dtype=tf.float32)
    I_tr = tf.convert_to_tensor(I_train, dtype=tf.float32)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            loss, loss_dict = compute_loss(
                t_col_tensor, t_tr, I_tr, model,
                TOTAL_WEEKS, I0, E0, S0, R0
            )
        grads = tape.gradient(loss, model.trainable_variables)
        optm.apply_gradients(zip(grads, model.trainable_variables))
        return loss, loss_dict

    for itr in range(n_iter):
        loss, loss_dict = step()
        if itr % 5000 == 0:
            print(f"  iter {itr:5d} | total {float(loss):.6f} | "
                  f"data {float(loss_dict['data_loss']):.6f} | "
                  f"ODE {float(loss_dict['ODE_loss']):.6f} | "
                  f"IC {float(loss_dict['IC_loss']):.6f} | "
                  f"cons {float(loss_dict['conservation_loss']):.6f}")

    return model

### Rolling window training and forecasting
First_train_weeks = 17 # initial window uses weeks 1–17
Forecast_horizon  = 4 # forecast 1, 2, 3, 4 weeks ahead

### Storage dictionaries
all_predictions = {}
all_observations = {}
all_beta = {}
all_naive = {}   # naive baseline: last known observation

model = None

for train_end in range(First_train_weeks,
                        N_total_points - Forecast_horizon + 1):

    t_tr_norm_global = t_data_norm[:train_end]
    I_tr_np = I_data[:train_end]

    t_tr_norm   = t_tr_norm_global
    TOTAL_WEEKS = float(N_total_points - 1)
    n_iter = 50_000

    print(f"Training on weeks 1–{train_end} "
          f"| forecasting weeks {train_end+1}–{train_end+Forecast_horizon} "
          f"| T = {TOTAL_WEEKS:.1f} weeks | iters = {n_iter}"
          f"{'[warm start]' if model is not None else '  [cold start]'}")

    model = train_window(t_tr_norm, I_tr_np, TOTAL_WEEKS=TOTAL_WEEKS,
                         n_iter=n_iter, warm_start_model=model)

    ### Last observed value for naive baseline
    last_obs = float(I_data[train_end - 1])

    for h in range(1, Forecast_horizon + 1):
        forecast_idx = train_end + h
        if forecast_idx >= N_total_points:
            continue

        t_fc_global = float(t_data_norm[forecast_idx])
        t_fc_norm = t_fc_global
        t_fc_tensor = tf.constant([[t_fc_norm]], dtype=tf.float32)

        _, E_pred, _, _, beta_pred = model(t_fc_tensor)
        sigma_val = 1.75
        pred_val = float(np.clip(sigma_val * E_pred.numpy()[0, 0], 0.0, 1.0))
        beta_val = float(np.clip(beta_pred.numpy()[0, 0], 0.0, None))

        key = (train_end, forecast_idx)
        all_predictions[key] = pred_val
        all_beta[key] = beta_val
        all_naive[key] = last_obs
        all_observations[forecast_idx] = float(I_data[forecast_idx])


### Collect results by forecast horizon for plotting and evaluation
horizon_results = {
    h: {"t": [], "pred": [], "obs": [], "naive": []}
    for h in range(1, 5)
}
horizon_beta = {h: {"t": [], "beta": []} for h in range(1, 5)}

for train_end in range(First_train_weeks,
                        N_total_points - Forecast_horizon + 1):
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
            horizon_results[h]["naive"].append(all_naive[key])
        if key in all_beta:
            horizon_beta[h]["t"].append(t_data_norm[forecast_idx, 0])
            horizon_beta[h]["beta"].append(all_beta[key])


### Plotting
### 1–4 week forecast grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for h, ax in zip(range(1, 5), axes.flatten()):
    t_vals = np.array(horizon_results[h]["t"])
    pred_arr  = np.array(horizon_results[h]["pred"])  * N_val
    obs_arr = np.array(horizon_results[h]["obs"])   * N_val
    naive_arr = np.array(horizon_results[h]["naive"]) * N_val

    ax.plot(t_vals, obs_arr, color="#004F94", lw=1.5, label="Observed")
    ax.plot(t_vals, pred_arr,color="#ff7ee3", lw=1.5,
            label=f"PINN {h}-week")
    ax.plot(t_vals, naive_arr, color="orange",  lw=1.0,
            linestyle="--", label="Naive baseline")
    ax.set_title(f"{h}-week-ahead forecast")
    ax.set_xlabel("Normalised time")
    ax.set_ylabel("New cases per week")
    ax.legend(fontsize=8)
    ax.grid(True)

plt.suptitle("SEIR-PINN rolling window forecasts (1–4 weeks ahead)",
             fontsize=14)
plt.tight_layout()
plt.savefig("rolling_window_forecasts.png", dpi=150)
plt.show()

### Standalone 1-week-ahead forecast plot
fig, ax = plt.subplots(figsize=(10, 5))
t_vals = np.array(horizon_results[1]["t"])
pred_arr = np.array(horizon_results[1]["pred"])  * N_val
obs_arr = np.array(horizon_results[1]["obs"])   * N_val
naive_arr = np.array(horizon_results[1]["naive"]) * N_val
ax.plot(t_vals, obs_arr, color="#004F94", lw=1.5, label="Observed")
ax.plot(t_vals, pred_arr, color="#ff7ee3", lw=1.5, label="PINN 1-week")
ax.plot(t_vals, naive_arr, color="orange",  lw=1.0,
        linestyle="--", label="Naive baseline")
ax.set_title("SEIR-PINN rolling window: 1-week-ahead forecast", fontsize=14)
ax.set_xlabel("Normalised time")
ax.set_ylabel("New cases per week")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("rolling_window_forecast_1week.png", dpi=150)
plt.show()

### R(t) plots
gamma_plot = 1.75  # matches value used in training

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
for h, ax in zip(range(1, 5), axes.flatten()):
    t_vals = np.array(horizon_beta[h]["t"])
    R_arr  = np.array(horizon_beta[h]["beta"]) / gamma_plot

    ax.plot(t_vals, R_arr, color="#ff7ee3", lw=1.5,
            label=f"R(t) — {h}-week ahead")
    ax.axhline(y=1.0, color="gray", lw=1, linestyle="--",
               label="R = 1 threshold")
    ax.set_title(f"{h}-week-ahead R(t)")
    ax.set_xlabel("Normalised time (study period)")
    ax.set_ylabel("R(t) = β(t) / γ")
    ax.legend()
    ax.grid(True)

plt.suptitle("SEIR-PINN rolling window: effective reproduction number R(t) "
             "(1–4 weeks ahead)", fontsize=14)
plt.tight_layout()
plt.savefig("rolling_window_Rt.png", dpi=150)
plt.show()

### Observed weekly cases
plt.figure(figsize=(12, 5))
plt.plot(t_data_norm.reshape(-1), I_data.reshape(-1) * N_val,
         color="#004F94", lw=1.5, marker='o', markersize=3)
plt.title("Weekly reported COVID-19 cases in England (Jul 2020 – Apr 2022)",
          fontsize=13)
plt.xlabel("Normalised time")
plt.ylabel("New cases per week")
plt.grid(True)
plt.tight_layout()
plt.savefig("observed_weekly_cases.png", dpi=150)
plt.show()


### Evaluation metrics
print("\nForecast evaluation (MAE, RMSE, MASE):")
rows = []

for h in range(1, 5):
    pred  = np.array(horizon_results[h]["pred"])
    obs = np.array(horizon_results[h]["obs"])
    naive = np.array(horizon_results[h]["naive"])

    mae_pinn = np.mean(np.abs(pred  - obs))
    rmse_pinn = np.sqrt(np.mean((pred  - obs) ** 2))
    mae_naive = np.mean(np.abs(naive - obs))

    ### MASE < 1 means PINN beats naive baseline
    mase = mae_pinn / (mae_naive + 1e-10)

    print(f" {h}-week | "
          f"PINN MAE={mae_pinn:.6f}  RMSE={rmse_pinn:.6f} | "
          f"Naive MAE={mae_naive:.6f} | "
          f"MASE={mase:.4f} "
          f"{'[PINN wins]' if mase < 1 else '[Naive wins]'}")

    rows.append({
        "horizon": h,
        "PINN_MAE": mae_pinn,
        "PINN_RMSE": rmse_pinn,
        "Naive_MAE": mae_naive,
        "MASE": mase,
    })

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv("forecast_metrics.csv", index=False)
print("\nMetrics saved to forecast_metrics.csv")
print(metrics_df.to_string(index=False))