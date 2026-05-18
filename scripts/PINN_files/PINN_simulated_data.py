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
 
data_folder = os.path.join("..", "..", "data")
output_dir = "../../png_files"
 
### Accumulate metrics across all scenarios
all_metrics = []
 
### Define scenarios to run
scenarios = [
    ### Constant-beta scenarios (one entry per beta value)
    {"label": "beta_0.75", "csv": "SEIR_data_beta_0.75.csv", "beta_true": 0.75, "smooth_beta": True},
    {"label": "beta_0.5",  "csv": "SEIR_data_beta_0.5.csv",  "beta_true": 0.5,  "smooth_beta": True},
    {"label": "beta_0.4",  "csv": "SEIR_data_beta_0.4.csv",  "beta_true": 0.4,  "smooth_beta": True},
    ### Time-varying beta scenarios
    {"label": "beta_piecewise", "csv": "SEIR_beta_peicewise.csv", "beta_true": None, "smooth_beta": False},
    {"label": "beta_spline",    "csv": "SEIR_beta_spline.csv", "beta_true": None, "smooth_beta": False},
    {"label": "beta_exp_decay", "csv": "SEIR_beta_exponential_decay_results.csv", "beta_true": None, "smooth_beta": False},
]
 
### Gaussian noise scenarios (1% - 20%), beta_true = 0.75
for noise_percent in range(1, 21):
    scenarios.append({
        "label": f"Gaussian_noise_{noise_percent}percent",
        "csv": f"SEIR_Gaussian_noise_{noise_percent}percent.csv",
        "beta_true": 0.75,
        "smooth_beta": True,
    })
 
### Compute and report error metrics
def compute_metrics(pred, true, name, label, compartment):
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
 
    mae  = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
 
    mean_true = np.mean(true)
    peak_true = np.max(true)
    mae_pct = 100 * mae  / (mean_true + 1e-8)
    rmse_pct = 100 * rmse / (mean_true + 1e-8)
    peak_mae_pct = 100 * mae  / (peak_true + 1e-8)
 
    print(f"{name}:")
    print(f"MAE = {mae:.4f} ({mae_pct:.3f}%)")
    print(f"RMSE = {rmse:.4f} ({rmse_pct:.3f}%)")
    print(f"Peak-normalised MAE = {peak_mae_pct:.3f}%\n")
 
    return {
        "scenario": label,
        "compartment": compartment,
        "MAE": mae,
        "RMSE": rmse,
        "MAE_pct": mae_pct,
        "RMSE_pct": rmse_pct,
        "peak_MAE_pct": peak_mae_pct,
    }
 
### Scenario loop
for scenario in scenarios:
    label = scenario["label"]
    csv_file  = scenario["csv"]
    beta_true = scenario["beta_true"]   # None for time-varying scenarios
    smooth_beta = scenario["smooth_beta"]
 
    print(f"\n{'='*50}")
    print(f"Running PINN for scenario: {label}")
    print(f"{'='*50}")
 
    data_path = os.path.join(data_folder, csv_file)
    data = pd.read_csv(data_path)
 
    t_data = data["time"].values.reshape(-1, 1)
    I_data = data["I"].values.reshape(-1, 1)
 
    ### Train/test split
    N_obs = len(I_data)
    t_data = t_data[:N_obs].reshape(-1, 1)
    I_data = I_data.reshape(-1, 1)
 
    split = int(0.9 * N_obs)
    t_train = t_data[:split]
    I_train = I_data[:split]
    t_test = t_data[split:]
    I_test = I_data[split:]
 
    I_scale = tf.constant(float(I_train.max()), dtype=tf.float32)
 
    N_total = 100001
 
    t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
    I_train_tensor = tf.convert_to_tensor(I_train, dtype=tf.float32)
 
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
        
        beta   = Dense(1, activation=None, name='beta')(beta_h)
    
        return Model(inputs=t_input, outputs=[S, E, I, R, beta])
 
    ### Fresh model for each scenario
    tf.keras.backend.clear_session()
    model = create_pinn_model()
    model.summary()
 
    ### Define intitial conditions
    S0_fixed = tf.constant(100000/100001, dtype=tf.float32)
    E0_fixed = tf.constant(0.0, dtype=tf.float32)
    I0_fixed = tf.constant(1/100001, dtype=tf.float32)
    R0_fixed = tf.constant(0.0, dtype=tf.float32)
 
    ### Loss function
    def loss_function(t_col, t_data_loss, I_data_loss, net, I_scale, smooth_beta=True):
 
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
        sigma = tf.constant(0.25, dtype=tf.float32)
        gamma = tf.constant(0.25, dtype=tf.float32)
 
        ### Compute derivatives e.g. dS/dt 
        ### Use automatic differentiation
        dS_dt = tape.gradient(S, t_col)
        dE_dt = tape.gradient(E, t_col)
        dI_dt = tape.gradient(I, t_col)
        dR_dt = tape.gradient(R, t_col)
 
        d_beta_dt = tape.gradient(beta, t_col)
        beta_smooth_loss = tf.reduce_mean(tf.square(d_beta_dt)) if smooth_beta else 0.0
 
        del tape
 
        ### SEIR equations 
        T = tf.constant(100.0, dtype=tf.float32)
        dS_dt_physics = T * (-beta * S * I)
        dE_dt_physics = T * (beta * S * I - sigma * E)
        dI_dt_physics = T * (sigma * E - gamma * I)
        dR_dt_physics = T * (gamma * I)
 
        ### Physics informed loss
        ODE_loss = (
            tf.reduce_mean(tf.square(dS_dt - dS_dt_physics)) +
            tf.reduce_mean(tf.square(dE_dt - dE_dt_physics)) +
            tf.reduce_mean(tf.square(dI_dt - dI_dt_physics)) +
            tf.reduce_mean(tf.square(dR_dt - dR_dt_physics))
        )
 
         ### Initial condition loss (evaluate at t=0)
        t_zero = tf.constant([[0.0]], dtype=tf.float32)
        S_0, E_0, I_0, R_0, _ = net(t_zero)
        Initial_condition_loss = tf.reduce_mean(
            tf.square(S_0 - S0_fixed) +
            tf.square(E_0 - E0_fixed) +
            tf.square(I_0 - I0_fixed) +
            tf.square(R_0 - R0_fixed)
        )
 
        ### constrain SEIR equations to equal 1
        S, E, I, R, beta = net(t_col)
        conservation_loss = tf.reduce_mean(tf.square(S + E + I + R - 1.0))

        ### Data loss 
        _, _, I_pred, _, _ = net(t_data_loss)
        data_loss = tf.reduce_mean(tf.square((I_pred - I_data_loss) / I_scale))
 
        ### Total loss
        total_loss = (
            1.0 * data_loss +
            0.1 * ODE_loss +
            1.0 * Initial_condition_loss +
            1.0 * conservation_loss +
            0.1 * beta_smooth_loss
        )
 
        return total_loss, {
            "data_loss": data_loss,
            "IC_loss": Initial_condition_loss,
            "conservation_loss": conservation_loss,
            "ODE_loss": ODE_loss,
        }
 
    ### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
    ### Optimiser and collocation points 
    optm = Adam(learning_rate=0.001)
 
    ### Collocation points for physics
    n_collocation = 1000
    t_col_tensor = tf.convert_to_tensor(
        np.linspace(0, 1, n_collocation).reshape(-1, 1), dtype=tf.float32
    )
 
    ### ensure all inputs are float32 for training
    t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)
    I_train = tf.convert_to_tensor(I_train, dtype=tf.float32)
    t_test = tf.convert_to_tensor(t_test,  dtype=tf.float32)
    I_test = tf.convert_to_tensor(I_test,  dtype=tf.float32)
 
    ### Training loop
    @tf.function
    def train_step(t_col, t_data, I_data):
        with tf.GradientTape() as tape:
            total_loss, loss_dict = loss_function(t_col, t_data, I_data, model, I_scale, smooth_beta)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optm.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, loss_dict
 
    @tf.function
    def test_step(t_col, t_data, I_data):
        return loss_function(t_col, t_data, I_data, model, I_scale, smooth_beta)
 
    train_loss_record = []
    test_loss_record  = []
 
    print("Starting training...")
    for itr in range(50000):
        train_loss, train_loss_dict = train_step(t_col_tensor, t_train, I_train)
        train_loss_record.append(float(train_loss))
 
        test_loss, test_loss_dict = test_step(t_col_tensor, t_test, I_test)
        test_loss_record.append(float(test_loss))
 
        if itr % 1000 == 0:
            print(
                f"Iteration {itr} | "
                f"Train: {float(train_loss):.6f}  Test: {float(test_loss):.6f} | "
                f"Data: {float(train_loss_dict['data_loss']):.6f}  "
                f"IC: {float(train_loss_dict['IC_loss']):.6f}  "
                f"Conservation: {float(train_loss_dict['conservation_loss']):.6f}  "
                f"ODE: {float(train_loss_dict['ODE_loss']):.6f}"
            )
 
    ### Predictions for plotting and metrics
    t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
    S_pred, E_pred, I_pred, R_pred, _ = model(t_tensor)
 
    def to_numpy_flat(arr):
        return arr.numpy().flatten() if hasattr(arr, 'numpy') else arr.flatten()
 
    t_data_np = to_numpy_flat(t_data)
    I_pred_np = to_numpy_flat(I_pred)
    t_train_np = to_numpy_flat(t_train)
    I_train_np = to_numpy_flat(I_train)
    t_test_np = to_numpy_flat(t_test)
    I_test_np = to_numpy_flat(I_test)
 
    ### Un-normalise time and counts for plotting and metrics
    days_total = 100
    N_total = 100001
 
    t_data_unnorm = t_data_np * days_total
    t_train_unnorm = t_train_np * days_total
    t_test_unnorm = t_test_np * days_total
 
    I_pred_unnorm = I_pred_np * N_total
    I_train_unnorm = I_train_np * N_total
    I_test_unnorm = I_test_np * N_total
 
    S_rec = 1.0 - to_numpy_flat(E_pred) - to_numpy_flat(I_pred) - to_numpy_flat(R_pred)
    S_rec_unnorm  = S_rec * N_total
    E_pred_unnorm = to_numpy_flat(E_pred) * N_total
    R_pred_unnorm = to_numpy_flat(R_pred) * N_total
 
    ### Training loss
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_record, label='Train loss')
    plt.plot(test_loss_record,  label='Test loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({label}) 90/10 split')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PINN_training_loss_{label}_90_10.png'))
    plt.close()
 
    ### PINN prediction vs observed
    plt.figure(figsize=(14, 6))
    plt.plot(t_data_unnorm,  I_pred_unnorm,  color="#ff7ee3", linewidth=2, label='Infected - PINN prediction')
    plt.plot(t_train_unnorm, I_train_unnorm, color="#004F94", linewidth=2, label='Infected - data')
    plt.plot(t_test_unnorm,  I_test_unnorm,  color="#004F94", linewidth=2)
    plt.axvline(x=t_train_unnorm[-1], color='gray', linestyle='--', label='Train/Test Split')
    plt.xlabel('Days')
    plt.ylabel('Number of infected individuals')
    plt.title(f'PINN prediction {label} - 90/10 split')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PINN_beta_{label}_90_10.png'))
    plt.close()
 
    ### Estimated beta
    t_plot = np.linspace(0.0, 1.0, 500).reshape(-1, 1)
    t_plot_tensor = tf.convert_to_tensor(t_plot, dtype=tf.float32)
    _, _, _, _, beta_pred = model.predict(t_plot_tensor)
    t_plot_unnorm = t_plot.flatten() * days_total
 
    plt.figure(figsize=(8, 5))
    plt.plot(t_plot_unnorm, beta_pred.flatten(), 'g-', linewidth=2, label='β(t) estimated')
    if beta_true is not None:
        plt.axhline(y=beta_true, color='gray', linestyle='--', linewidth=1.5,
                    label=f'β true = {beta_true}')
    plt.xlabel('Days')
    plt.ylabel('β(t)')
    plt.ylim(0, 1)
    plt.title(f'Estimated β(t) vs true β ({label}) 90/10 split')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PINN_parameter_est_beta_{label}_90_10.png'))
    plt.close()
 
    ### Susceptible
    plt.figure(figsize=(14, 6))
    plt.plot(t_data_unnorm, S_rec_unnorm, color="green", linewidth=2, label='Susceptible (PINN)')
    if "S" in data.columns:
        plt.plot(t_data_unnorm, data["S"].values.flatten() * N_total,
                 'k--', linewidth=2, label='Susceptible (ground truth)')
    plt.xlabel('Days'); plt.ylabel('Number of individuals')
    plt.title(f'Susceptible population ({label})')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PINN_S_comparison_{label}.png'))
    plt.close()
 
    ### Exposed
    plt.figure(figsize=(14, 6))
    plt.plot(t_data_unnorm, E_pred_unnorm, color="orange", linewidth=2, label='Exposed (PINN)')
    if "E" in data.columns:
        plt.plot(t_data_unnorm, data["E"].values.flatten() * N_total,
                 'k--', linewidth=2, label='Exposed (ground truth)')
    plt.xlabel('Days'); plt.ylabel('Number of individuals')
    plt.title(f'Exposed population ({label})')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PINN_E_comparison_{label}.png'))
    plt.close()
 
    ### Recovered
    plt.figure(figsize=(14, 6))
    plt.plot(t_data_unnorm, R_pred_unnorm, color="red", linewidth=2, label='Recovered (PINN)')
    if "R" in data.columns:
        plt.plot(t_data_unnorm, data["R"].values.flatten() * N_total,
                 'k--', linewidth=2, label='Recovered (ground truth)')
    plt.xlabel('Days'); plt.ylabel('Number of individuals')
    plt.title(f'Recovered population ({label})')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PINN_R_comparison_{label}.png'))
    plt.close()
 
    ### Error metrics
    print("Error metrics:")

    I_true_unnorm = data["I"].values.flatten() * N_total
 
    if "S" in data.columns:
        S_true_unnorm = data["S"].values.flatten() * N_total
        all_metrics.append(compute_metrics(S_rec_unnorm,  S_true_unnorm, "Susceptible", label, "S"))
 
    if "E" in data.columns:
        E_true_unnorm = data["E"].values.flatten() * N_total
        all_metrics.append(compute_metrics(E_pred_unnorm, E_true_unnorm, "Exposed",     label, "E"))
 
    all_metrics.append(compute_metrics(I_pred_unnorm, I_true_unnorm, "Infected", label, "I"))
 
    if "R" in data.columns:
        R_true_unnorm = data["R"].values.flatten() * N_total
        all_metrics.append(compute_metrics(R_pred_unnorm, R_true_unnorm, "Recovered",   label, "R"))
 
### Save evaluation metrics to CSV
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(output_dir, "PINN_error_metrics_90_10.csv"), index=False)
print("\nMetrics saved to PINN_error_metrics_90_10.csv")
print(metrics_df.to_string(index=False))
 