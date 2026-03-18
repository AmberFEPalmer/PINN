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

beta_values = [0.75, 0.5, 0.4]

for beta_true in beta_values:

    print(f"\n{'='*50}")
    print(f"Running PINN for beta = {beta_true}")
    print(f"{'='*50}")

    ### Load the CSV file for this beta
    data_path = os.path.join(data_folder, f"SEIR_data_beta_{beta_true}.csv")
    data = pd.read_csv(data_path)

    t_data = data["time"].values.reshape(-1, 1)
    I_data = data["I"].values.reshape(-1, 1)

    ### Train/test split
    N_obs = len(I_data)
    t_data = t_data[:N_obs].reshape(-1, 1)
    I_data = I_data.reshape(-1, 1)

    split = int(0.8 * N_obs)
    t_train = t_data[:split]
    I_train = I_data[:split]
    t_test  = t_data[split:]
    I_test  = I_data[split:]

    I_scale = tf.constant(float(I_train.max()), dtype=tf.float32)

    t_train_tensor = tf.convert_to_tensor(t_train, dtype=tf.float32)
    I_train_tensor = tf.convert_to_tensor(I_train, dtype=tf.float32)

    ### Define PINN model
    def create_pinn_model():
        t_input = Input(shape=(1,), name='time_input')

        seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
        seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir)
        seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir)

        S = Dense(1, activation=None, name='S')(seir)
        E = Dense(1, activation=None, name='E')(seir)
        I = Dense(1, activation=None, name='I')(seir)
        R = Dense(1, activation=None, name='R')(seir)

        beta_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
        beta_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_hidden)
        beta_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_hidden)
        log_beta = Dense(1, activation=None, name='log_beta')(beta_hidden)
        beta = Lambda(lambda x: tf.exp(x), name='beta')(log_beta)

        model = Model(inputs=t_input, outputs=[S, E, I, R, beta])
        return model

    ### Fresh model for each beta
    tf.keras.backend.clear_session()
    model = create_pinn_model()
    model.summary()

    ### Initial conditions
    S0 = tf.constant(100000/100001, dtype=tf.float32)
    E0 = tf.constant(0.0, dtype=tf.float32)
    I0 = tf.constant(1/100001, dtype=tf.float32)
    R0 = tf.constant(0.0, dtype=tf.float32)

    S0_fixed = S0
    E0_fixed = E0
    I0_fixed = I0
    R0_fixed = R0

    ### Loss function
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
            S, E, I, R, beta = net(t_col)

        sigma = tf.constant(0.25, dtype=tf.float32)
        gamma = tf.constant(0.25, dtype=tf.float32)

        dS_dt = tape.gradient(S, t_col)
        dE_dt = tape.gradient(E, t_col)
        dI_dt = tape.gradient(I, t_col)
        dR_dt = tape.gradient(R, t_col)
        del tape

        T = tf.constant(100.0, dtype=tf.float32)
        dS_dt_physics = T * (-beta * S * I)
        dE_dt_physics = T * (beta * S * I - sigma * E)
        dI_dt_physics = T * (sigma * E - gamma * I)
        dR_dt_physics = T * (gamma * I)

        loss_S = tf.reduce_mean(tf.square(dS_dt - dS_dt_physics))
        loss_E = tf.reduce_mean(tf.square(dE_dt - dE_dt_physics))
        loss_I = tf.reduce_mean(tf.square(dI_dt - dI_dt_physics))
        loss_R = tf.reduce_mean(tf.square(dR_dt - dR_dt_physics))
        ODE_loss = loss_S + loss_E + loss_I + loss_R

        t_zero = tf.constant([[0.0]], dtype=tf.float32)
        S_0, E_0, I_0, R_0, _ = net(t_zero)
        Initial_condition_loss = tf.reduce_mean(
            tf.square(S_0 - S0_fixed) +
            tf.square(E_0 - E0_fixed) +
            tf.square(I_0 - I0_fixed) +
            tf.square(R_0 - R0_fixed)
        )

        S, E, I, R, beta = net(t_col)
        conservation_loss = tf.reduce_mean(tf.square(S + E + I + R - 1.0))

        _, _, I_pred, _, _ = net(t_data_loss)
        data_loss = tf.reduce_mean(tf.square((I_pred - I_data_loss) / I_scale))

        total_loss = data_loss + ODE_loss + Initial_condition_loss + conservation_loss

        return total_loss, {
            "data_loss": data_loss,
            "IC_loss": Initial_condition_loss,
            "conservation_loss": conservation_loss,
            "ODE_loss": ODE_loss,
        }

    ### Optimiser — fresh instance each loop
    optm = Adam(learning_rate=0.001)

    ### Collocation points
    n_collocation = 1000
    t_col_uniform = np.linspace(0, 1, n_collocation).reshape(-1, 1)
    t_col_tensor = tf.convert_to_tensor(t_col_uniform, dtype=tf.float32)

    t_train = tf.convert_to_tensor(t_train, dtype=tf.float32)
    I_train = tf.convert_to_tensor(I_train, dtype=tf.float32)
    t_test  = tf.convert_to_tensor(t_test,  dtype=tf.float32)
    I_test  = tf.convert_to_tensor(I_test,  dtype=tf.float32)

    @tf.function
    def train_step(t_col, t_data, I_data):
        with tf.GradientTape() as tape:
            total_loss, loss_dict = loss_function(t_col, t_data, I_data, model, I_scale)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optm.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, loss_dict

    @tf.function
    def test_step(t_col, t_data, I_data):
        total_loss, loss_dict = loss_function(t_col, t_data, I_data, model, I_scale)
        return total_loss, loss_dict

    ### Training loop
    train_loss_record = []
    test_loss_record = []

    print("Starting training...")
    for itr in range(50000):
        train_loss, train_loss_dict = train_step(t_col_tensor, t_train, I_train)
        train_loss_record.append(float(train_loss))
        test_loss, test_loss_dict = test_step(t_col_tensor, t_test, I_test)

        if itr % 1000 == 0:
            print(
                f"Iteration {itr} | "
                f"Train: {float(train_loss):.6f}, Test: {float(test_loss):.6f} | "
                f"Data: {float(train_loss_dict['data_loss']):.6f}, "
                f"IC: {float(train_loss_dict['IC_loss']):.6f}, "
                f"Conservation: {float(train_loss_dict['conservation_loss']):.6f}, "
                f"ODE: {float(train_loss_dict['ODE_loss']):.6f}"
            )

    ### Predictions
    t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
    _, _, I_pred, _, _ = model(t_tensor)

    def to_numpy_flat(arr):
        if hasattr(arr, 'numpy'):
            return arr.numpy().flatten()
        else:
            return arr.flatten()

    t_data_np  = to_numpy_flat(t_data)
    I_pred_np  = to_numpy_flat(I_pred)
    t_train_np = to_numpy_flat(t_train)
    I_train_np = to_numpy_flat(I_train)
    t_test_np  = to_numpy_flat(t_test)
    I_test_np  = to_numpy_flat(I_test)

    ### Plot training loss
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss_record)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (β={beta_true})')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'PINN_training_loss_beta_{beta_true}.png'))
    plt.show()

    ### Plot PINN prediction vs observed
    plt.figure(figsize=(14, 6))
    plt.plot(t_data_np, I_pred_np, color="#ff7ee3", linewidth=2, label='I (PINN prediction)')
    plt.plot(t_train_np, I_train_np, color="#004F94", linewidth=2, label='I (observed – train)')
    plt.plot(t_test_np, I_test_np, color="#004F94", linewidth=2, linestyle='--', label='I (observed – test)')
    plt.axvline(x=t_train_np[-1], color='gray', linestyle='--', label='Train/Test Split')
    plt.xlabel('Time')
    plt.ylabel('Infected (normalized)')
    plt.title(f'PINN prediction (β={beta_true})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PINN_beta_piecewise_{beta_true}.png'))
    plt.show()

    ### Plot estimated beta over time
    t_plot = np.linspace(0.0, 1.0, 500)
    t_plot_tensor = tf.convert_to_tensor(t_plot.reshape(-1, 1), dtype=tf.float32)
    _, _, _, _, beta_pred = model.predict(t_plot_tensor)

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot, beta_pred.flatten(), 'g-', linewidth=2, label='β(t) estimated')
    plt.axhline(y=beta_true, color='r', linestyle='--', linewidth=1.5, label=f'β true = {beta_true}')
    plt.xlabel('Normalised time')
    plt.ylabel('β(t)')
    plt.ylim(0, 1)
    plt.title(f'Estimated β(t) vs true β (β={beta_true})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'PINN_parameter_est_beta_{beta_true}.png'))
    plt.show()

    print(f"Finished beta = {beta_true}")