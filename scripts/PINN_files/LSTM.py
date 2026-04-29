import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
 
data_folder = os.path.join("..", "..", "data")
output_dir = "../../png_files"
 
### Creating sequences for LSTM input
def create_sequences(I, seq_length=10):
    X, y = [], []
    for i in range(len(I) - seq_length):
        X.append(I[i:i+seq_length])
        y.append(I[i+seq_length])
    return np.array(X), np.array(y)
 
### Define scenarios to run
scenarios = [
    ### Constant-beta scenarios (one entry per beta value)
    {"label": "beta_0.75", "csv": "SEIR_data_beta_0.75.csv", "beta_true": 0.75},
    {"label": "beta_0.5",  "csv": "SEIR_data_beta_0.5.csv",  "beta_true": 0.5},
    {"label": "beta_0.4",  "csv": "SEIR_data_beta_0.4.csv",  "beta_true": 0.4},
    ### Time-varying beta scenarios
    {"label": "beta_piecewise",  "csv": "SEIR_beta_peicewise.csv",                   "beta_true": None},
    {"label": "beta_spline",     "csv": "SEIR_beta_spline.csv",                       "beta_true": None},
    {"label": "beta_exp_decay",  "csv": "SEIR_beta_exponential_decay_results.csv",    "beta_true": None},
]
 
### Gaussian noise scenarios (1% - 20%), beta_true = 0.75
for noise_percent in range(1, 21):
    scenarios.append({
        "label": f"Gaussian_noise_{noise_percent}percent",
        "csv":   f"SEIR_Gaussian_noise_{noise_percent}percent.csv",
        "beta_true": 0.75,
    })
 
### Run each scenario
for scenario in scenarios:
    label     = scenario["label"]
    csv_file  = scenario["csv"]
    beta_true = scenario["beta_true"]   # None for time-varying scenarios
 
    print(f"\n{'='*50}")
    print(f"Running LSTM for scenario: {label}")
    print(f"{'='*50}")
 
    data_path = os.path.join(data_folder, csv_file)
    data = pd.read_csv(data_path)
 
    I_raw = data["I"].values.reshape(-1, 1)
    N_obs = len(I_raw)
 
    # --- Fix 1: use actual day indices so x-axis shows real days ---
    # If the CSV time column is normalised (0–1) or absent, fall back to
    # integer day indices so the plot x-axis always shows meaningful numbers.
    if "time" in data.columns:
        t_raw = data["time"].values
        t_range = t_raw[-1] - t_raw[0]
        if t_range <= 1.0:
            # Normalised → rescale to integer days
            t_raw = np.round(t_raw * (N_obs - 1)).astype(int)
    else:
        t_raw = np.arange(N_obs)
 
    t_data = t_raw.reshape(-1, 1)
    I_data = I_raw
 
    # --- Fix 2: normalise I so the LSTM trains on [0, 1] values ---
    I_max  = float(I_data.max())
    I_norm = I_data / I_max          # normalised; predictions rescaled later
 
    ### Train / test split (90 / 10)
    split   = int(0.9 * N_obs)
    t_train = t_data[:split]
    t_test  = t_data[split:]
 
    I_train_norm = I_norm[:split]
    I_test_norm  = I_norm[split:]
 
    seq_length = 10
 
    X_train, y_train = create_sequences(I_train_norm, seq_length)
    X_test,  y_test  = create_sequences(I_test_norm,  seq_length)
 
    N_total = 100001
 
    ### Convert to tensors
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test_tensor  = tf.convert_to_tensor(X_test,  dtype=tf.float32)
    y_test_tensor  = tf.convert_to_tensor(y_test,  dtype=tf.float32)
 
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
 
    ### Build LSTM model
    input_layer  = Input(shape=(seq_length, 1))
    x = LSTM(50, return_sequences=True)(input_layer)
    x = LSTM(50, return_sequences=False)(x)
    x = Dense(50, activation='tanh')(x)
    output_layer = Dense(1)(x)
 
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    history = model.fit(
        X_train_tensor, y_train_tensor,
        epochs=50, batch_size=32,
        validation_data=(X_test_tensor, y_test_tensor),
        verbose=0
    )
 
    ### Predictions — rescale back to original counts
    # Note: the first seq_length steps have no full input window, so
    # predictions only start from index seq_length onward (warm-up period).
    I_pred_train = model.predict(X_train_tensor).flatten() * I_max
    I_pred_test  = model.predict(X_test_tensor).flatten()  * I_max
 
    ### Plotting
    # True data arrays — no warm-up trim, show the full time series
    t_train_plot = t_train[seq_length:].reshape(-1)   # trimmed to match pred length
    t_test_plot  = t_test.reshape(-1)                 # full test range
 
    I_train_plot = (I_train_norm[seq_length:] * I_max).reshape(-1)
    I_test_plot  = (I_test_norm * I_max).reshape(-1)  # full test range
 
    # Prediction time arrays — offset by seq_length due to warm-up window
    t_pred_test = t_test[seq_length:].reshape(-1)
 
    plt.figure(figsize=(14, 6))
 
    ### LSTM prediction (train + test concatenated)
    plt.plot(
        np.concatenate([t_train_plot, t_pred_test]),
        np.concatenate([I_pred_train, I_pred_test]),
        color="#ff7ee3",
        linewidth=2,
        label='Infected - LSTM prediction'
    )
 
    ### True training data
    plt.plot(
        t_train_plot,
        I_train_plot,
        color="#004F94",
        linewidth=2,
        label='Infected - data'
    )
 
    ### True test data (same colour, no duplicate legend entry)
    plt.plot(
        t_test_plot,
        I_test_plot,
        color="#004F94",
        linewidth=2
    )
 
    ### Train/test split vertical line
    plt.axvline(
        x=t_test_plot[0],
        color='gray',
        linestyle='--',
        label='Train/Test Split'
    )
 
    plt.xlabel('Days')
    plt.ylabel('Number of infected individuals')
    plt.title(f'LSTM prediction {label} - 90/10 split')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
 
    plt.savefig(os.path.join(output_dir, f'LSTM_{label}_90_10.png'))
    plt.close()