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

data_folder = os.path.join("..", "..", "data")
output_dir = "../../png_files"

### Define scenarios to run
scenarios = [
    ### Constant-beta scenarios (one entry per beta value)
    {"label": "beta_0.75", "csv": "SEIR_data_beta_0.75.csv",                    "beta_true": 0.75, "smooth_beta": True},
    {"label": "beta_0.5",  "csv": "SEIR_data_beta_0.5.csv",                     "beta_true": 0.5,  "smooth_beta": True},
    {"label": "beta_0.4",  "csv": "SEIR_data_beta_0.4.csv",                     "beta_true": 0.4,  "smooth_beta": True},
    
    ### Time-varying beta scenarios
    {"label": "beta_piecewise",  "csv": "SEIR_beta_peicewise.csv",                     "beta_true": None, "smooth_beta": False},
    {"label": "beta_spline",     "csv": "SEIR_beta_spline.csv",                        "beta_true": None, "smooth_beta": False},
    {"label": "beta_exp_decay",  "csv": "SEIR_beta_exponential_decay_results.csv",     "beta_true": None, "smooth_beta": False},
]

### Gaussian noise scenarios (1% – 20%), beta_true = 0.75 (the ground truth used to generate the data)
for noise_percent in range(1, 21):
    scenarios.append({
        "label":      f"Gaussian_noise_{noise_percent}percent",
        "csv":        f"SEIR_Gaussian_noise_{noise_percent}percent.csv",
        "beta_true":  0.75,
        "smooth_beta": True,
    })

### information for creating figures
for scenario in scenarios:
    label     = scenario["label"]
    csv_file  = scenario["csv"]
    beta_true = scenario["beta_true"]   # None for time-varying scenarios
    smooth_beta = scenario["smooth_beta"]

    print(f"\n{'='*50}")
    print(f"Running Bayesian PINN for scenario: {label}")
    print(f"{'='*50}")

    data_path = os.path.join(data_folder, csv_file)
    data = pd.read_csv(data_path)

    t_data = data["time"].values.reshape(-1, 1)
    I_data = data["I"].values.reshape(-1, 1)

    ### Train/test split
    N_obs = len(I_data)
    t_data = t_data[:N_obs].reshape(-1, 1)
    I_data = I_data.reshape(-1, 1)
    ### Generate training and testing data
    split = int(0.7 * N_obs)
    t_train = t_data[:split]   # take all elements from 0 up to "split"
    I_train = I_data[:split]
    t_test  = t_data[split:]   # take all elements from "split" to the end
    I_test  = I_data[split:]

    I_scale = tf.constant(float(I_train.max()), dtype=tf.float32)

    N_data = t_train.shape[0]

    ### Convert to tensors
    t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
    I_train_tensor = tf.convert_to_tensor(I_train, dtype=tf.float32)

    ### Control KL divergence weight (annealed during training)
    kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    ### Define Bayesian PINN
    ### Dense flipout = Bayesian layer; weights are probability distributions
    ### Variational inference approximates the posterior
    ### KL = Kullback–Leibler divergence: measures difference between two distributions
    ### https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout
    ### https://arxiv.org/abs/1803.04386
    def create_bayesian_pinn_model():
        t_input = Input(shape=(1,), name='time_input')

        seir = tfpl.DenseFlipout(50, activation='tanh',
            kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p))(t_input)
        seir = tfpl.DenseFlipout(50, activation='tanh',
            kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p))(seir)
        seir = tfpl.DenseFlipout(50, activation='tanh',
            kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p))(seir)

        S = tfpl.DenseFlipout(1, activation='softplus', name='S')(seir)
        E = tfpl.DenseFlipout(1, activation='softplus', name='E')(seir)
        I = tfpl.DenseFlipout(1, activation='softplus', name='I')(seir)
        R = tfpl.DenseFlipout(1, activation='softplus', name='R')(seir)

        ### Separate sub-network for time-varying beta
        beta_hidden = tfpl.DenseFlipout(50, activation='tanh',
            kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p))(t_input)
        beta_hidden = tfpl.DenseFlipout(50, activation='tanh',
            kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p))(beta_hidden)
        beta_hidden = tfpl.DenseFlipout(50, activation='tanh',
            kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p))(beta_hidden)
        beta = tfpl.DenseFlipout(1, activation='softplus', name='beta')(beta_hidden)

        model = Model(inputs=t_input, outputs=[S, E, I, R, beta])
        return model

    ### Fresh model for each scenario
    tf.keras.backend.clear_session()
    kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    model = create_bayesian_pinn_model()
    model.summary()

    ### Define initial conditions
    S0_fixed = tf.constant(100000/100001, dtype=tf.float32)
    E0_fixed = tf.constant(0.0,           dtype=tf.float32)
    I0_fixed = tf.constant(1/100001,      dtype=tf.float32)
    R0_fixed = tf.constant(0.0,           dtype=tf.float32)

    N_total = 100001

    ### Define loss function
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
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t_col)
            S, E, I, R, beta = net(t_col)

        ### Fixed epidemiological parameters (Qian et al. 2025)
        sigma = tf.constant(0.25, dtype=tf.float32, name='sigma')
        gamma = tf.constant(0.25, dtype=tf.float32, name='gamma')

        dS_dt = tape.gradient(S, t_col)
        dE_dt = tape.gradient(E, t_col)
        dI_dt = tape.gradient(I, t_col)
        dR_dt = tape.gradient(R, t_col)

        d_beta_dt = tape.gradient(beta, t_col)
        beta_smooth_loss = tf.reduce_mean(tf.square(d_beta_dt)) if smooth_beta else 0.0

        del tape

        ### SEIR ODEs (time rescaled to [0,1])
        days = 100.0
        T = tf.constant(days, dtype=tf.float32)
        dS_dt_physics = T * (-beta * S * I)
        dE_dt_physics = T * (beta * S * I - sigma * E)
        dI_dt_physics = T * (sigma * E - gamma * I)
        dR_dt_physics = T * (gamma * I)

        loss_S = tf.reduce_mean(tf.square(dS_dt - dS_dt_physics))
        loss_E = tf.reduce_mean(tf.square(dE_dt - dE_dt_physics))
        loss_I = tf.reduce_mean(tf.square(dI_dt - dI_dt_physics))
        loss_R = tf.reduce_mean(tf.square(dR_dt - dR_dt_physics))

        ODE_loss = loss_S + loss_E + loss_I + loss_R

        ### Initial condition loss
        t_zero = tf.constant([[0.0]], dtype=tf.float32)
        S_0, E_0, I_0, R_0, _ = net(t_zero)
        Initial_condition_loss = tf.reduce_mean(
            tf.square(S_0 - S0_fixed) +
            tf.square(E_0 - E0_fixed) +
            tf.square(I_0 - I0_fixed) +
            tf.square(R_0 - R0_fixed))

        ### Conservation loss: S + E + I + R = 1
        S, E, I, R, beta = net(t_col)
        conservation_loss = tf.reduce_mean(tf.square(S + E + I + R - 1.0))

        ### Data loss
        _, _, I_pred, _, _ = net(t_data_loss)
        data_loss = tf.reduce_mean(tf.square((I_pred - I_data_loss) / I_scale))

        ### KL divergence (annealed)
        Kl_loss = tf.add_n(net.losses) / N_data
        total_loss = (
            1.0 * data_loss
            + 1.0 * Initial_condition_loss
            + 0.1 * beta_smooth_loss
            + 1.0 * conservation_loss
            + 0.1 * ODE_loss
            + kl_weight_var * Kl_loss
        )

        return total_loss, {
            "data_loss":         data_loss,
            "IC_loss":           Initial_condition_loss,
            "conservation_loss": conservation_loss,
            "ODE_loss":          ODE_loss,
            "beta_smooth_loss":  beta_smooth_loss,
            "KL_loss":           tf.reduce_sum(net.losses),
        }

    ### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
    optm = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    ### Collocation points for physics loss
    n_collocation = 1000
    t_col_uniform = np.linspace(0, 1, n_collocation).reshape(-1, 1)
    t_col_tensor  = tf.convert_to_tensor(t_col_uniform, dtype=tf.float32)

    ### Ensure float32 tensors
    t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)
    I_train = tf.convert_to_tensor(I_train, dtype=tf.float32)
    t_test  = tf.convert_to_tensor(t_test,  dtype=tf.float32)
    I_test  = tf.convert_to_tensor(I_test,  dtype=tf.float32)

    train_loss_record = []
    test_loss_record  = []

    @tf.function
    def train_step(t_col, t_data, I_data, I_scale):
        with tf.GradientTape() as tape:
            total_loss, loss_dict = loss_function(t_col, t_data, I_data, model, I_scale, smooth_beta)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optm.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, loss_dict

    @tf.function
    def test_step(t_col, t_data, I_data, I_scale):
        return loss_function(t_col, t_data, I_data, model, I_scale, smooth_beta)

    print("Starting training...")

    ### KL annealing: linearly ramp weight from 0 → kl_max over kl_ramp_iters
    ### https://arxiv.org/abs/1903.10145
    total_iters   = 50000
    kl_ramp_iters = 20000
    kl_max        = 0.0001

    for itr in range(total_iters):
        kl_weight_var.assign(tf.minimum(kl_max, kl_max * itr / kl_ramp_iters))

        train_loss, train_loss_dict = train_step(t_col_tensor, t_train, I_train, I_scale)
        train_loss_record.append(float(train_loss))

        test_loss, test_loss_dict = test_step(t_col_tensor, t_test, I_test, I_scale)

        if itr % 1000 == 0:
            KL_raw    = train_loss_dict["KL_loss"]
            KL_scaled = kl_weight_var * (KL_raw / N_data)
            print(
                f"Iteration {itr}, KL weight: {kl_weight_var.numpy():.5f}\n"
                f"Train Loss: {float(train_loss):.6f}, "
                f"Test Loss: {float(test_loss):.6f}\n"
                f"Data: {float(train_loss_dict['data_loss']):.6f}, "
                f"IC: {float(train_loss_dict['IC_loss']):.6f}, "
                f"Conservation: {float(train_loss_dict['conservation_loss']):.6f}, "
                f"ODE: {float(train_loss_dict['ODE_loss']):.6f}, "
                f"Beta Smooth: {float(train_loss_dict['beta_smooth_loss']):.6f}, "
                f"KL raw: {float(KL_raw):.6f}, "
                f"KL scaled: {float(KL_scaled):.6f}"
            )

    ### Predictions with uncertainty quantification

    def to_numpy_flat(arr):
        if hasattr(arr, 'numpy'):
            return arr.numpy().flatten()
        return arr.flatten()

    ### Monte Carlo sampling for uncertainty quantification
    ### https://link.springer.com/chapter/10.1007/978-1-0716-4132-3_7
    def predict_with_uncertainty(model, t, n_samples=200):
        preds = [model(t)[2].numpy() for _ in range(n_samples)]
        preds = np.array(preds)
        return preds.mean(axis=0), preds.std(axis=0)

    days_total = 100

    t_data_np  = to_numpy_flat(t_data)
    t_train_np = to_numpy_flat(t_train)
    I_train_np = to_numpy_flat(I_train)
    t_test_np  = to_numpy_flat(t_test)
    I_test_np  = to_numpy_flat(I_test)

    t_tensor = tf.convert_to_tensor(t_data_np.reshape(-1, 1), dtype=tf.float32)
    mean, std = predict_with_uncertainty(model, t_tensor, n_samples=200)
    mean = mean.flatten()
    std  = std.flatten()

    ### Un-normalise for plotting
    t_data_unnorm  = t_data_np  * days_total
    t_train_unnorm = t_train_np * days_total
    t_test_unnorm  = t_test_np  * days_total

    mean_unnorm = mean * N_total
    std_unnorm  = std  * N_total
    I_train_unnorm = I_train_np * N_total
    I_test_unnorm  = I_test_np  * N_total

    ### Plot training loss
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_record)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Bayesian PINN Training Loss ({label}) 70/30 split')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'Bayesian_PINN_training_loss_{label}_70_30.png'))
    plt.close()

    ### Plot prediction with uncertainty
    plt.figure(figsize=(14, 6))
    plt.plot(t_data_unnorm, mean_unnorm, color='#ff7ee3', linewidth=2, label='Infected – posterior mean')
    plt.fill_between(
        t_data_unnorm,
        mean_unnorm - 2 * std_unnorm,
        mean_unnorm + 2 * std_unnorm,
        color='#ff7ee3', alpha=0.25, label='95% credible interval')
    plt.plot(t_train_unnorm, I_train_unnorm, color='#004F94', linewidth=2, label='Infected – data')
    plt.plot(t_test_unnorm,  I_test_unnorm,  color='#004F94', linewidth=2)
    plt.axvline(x=t_train_unnorm[-1], color='gray', linestyle='--', label='Train/Test Split')
    plt.xlabel('Days')
    plt.ylabel('Number of infected individuals')
    plt.title(f'Bayesian PINN prediction {label} – 70/30 split')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Bayesian_PINN_{label}_70_30.png'))
    plt.close()

    ### Plot estimated beta(t) over time with uncertainty
    t_plot = np.linspace(0.0, 1.0, 500)
    t_plot_tensor = tf.convert_to_tensor(t_plot.reshape(-1, 1), dtype=tf.float32)

    beta_samples = np.array([model(t_plot_tensor)[4].numpy() for _ in range(1000)])
    beta_mean = beta_samples.mean(axis=0).flatten()
    beta_std  = beta_samples.std(axis=0).flatten()
    t_plot_unnorm = t_plot * days_total

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot_unnorm, beta_mean, color='#7397de', linewidth=2, label='β(t) mean')
    plt.fill_between(
        t_plot_unnorm,
        beta_mean - 2 * beta_std,
        beta_mean + 2 * beta_std,
        color='#7397de', alpha=0.25, label='95% credible interval')
    if beta_true is not None:
        plt.axhline(y=beta_true, color='gray', linestyle='--', linewidth=1.5,
                    label=f'β true = {beta_true}')
    plt.xlabel('Days')
    plt.ylabel('β(t)')
    plt.ylim(0, 1)
    plt.title(f'Bayesian PINN – estimated β(t) ({label}) 70/30 split')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Bayesian_PINN_parameter_est_beta_{label}_70_30.png'))
    plt.close()