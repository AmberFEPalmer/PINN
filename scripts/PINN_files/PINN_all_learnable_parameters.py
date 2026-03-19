import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import pandas as pd
import os

data_folder = os.path.join("..", "..", "data")
output_dir = "../../png_files"

### Define scenarios to run
scenarios = [
    ### Constant-beta scenarios (one entry per beta value)
    {"label": "beta_0.75", "csv": "SEIR_data_beta_0.75.csv","beta_true": 0.75, "smooth_beta": True},
    {"label": "beta_0.5", "csv": "SEIR_data_beta_0.5.csv", "beta_true": 0.5, "smooth_beta": True},
    {"label": "beta_0.4","csv": "SEIR_data_beta_0.4.csv","beta_true": 0.4, "smooth_beta": True},
    ### Time-varying beta scenarios
    {"label": "beta_piecewise","csv": "SEIR_beta_peicewise.csv","beta_true": None, "smooth_beta": False},
    {"label": "beta_spline","csv": "SEIR_beta_spline.csv", "beta_true": None, "smooth_beta": False},
    {"label": "beta_exp_decay","csv": "SEIR_beta_exponential_decay_results.csv", "beta_true": None, "smooth_beta": False},
]

### Gaussian noise scenarios (1% – 20%), beta_true = 0.75 (the ground truth used to generate the data)
for noise_percent in range(1, 21):
    scenarios.append({
        "label":      f"Gaussian_noise_{noise_percent}percent",
        "csv":        f"SEIR_Gaussian_noise_{noise_percent}percent.csv",
        "beta_true":  0.75,
        "smooth_beta": False,
    })

for scenario in scenarios:
    label     = scenario["label"]
    csv_file  = scenario["csv"]
    beta_true = scenario["beta_true"]   # None for time-varying scenarios

    print(f"\n{'='*50}")
    print(f"Running PINN for scenario: {label}")
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

    N_total = 100001 

    ### Define PINN
    ### L2 regularisation for hidden layers -> helps to prevent overfitting
    ### Add penalty proportional to the sum of squared coefficients to the loss function
    ### Reduce model complexity, penalise large weights
    ### https://keras.io/api/layers/regularizers/
    ### https://developers.google.com/machine-learning/crash-course/overfitting/regularization
    def create_pinn_model():
        ### Input layer = time 
        t_input = Input(shape=(1,), name='time_input')

        ### 3 Hidden layers, 50 neurons each , tanh activation (tanh = non-linear + smooth)
        ### 3 hidden layers with 50 neurons to match Qian et al. 2025
        seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
        seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir)
        seir = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(seir)

        ### SEIR outputs 
        ### 4 output layers
        ### No activation function
        S = Dense(1, activation="softplus", name='S')(seir)
        E = Dense(1, activation="softplus", name='E')(seir)
        I = Dense(1, activation="softplus", name='I')(seir)
        R = Dense(1, activation="softplus", name='R')(seir)

        ### Time-varying beta 
        ### beta = softplus activation -> allows it to be greater than 1
        ### 3 hidden layers, 50 neurons, tanh activation
        beta_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
        beta_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_hidden)   
        beta_hidden = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_hidden)
        log_beta = Dense(1, activation=None, name='log_beta')(beta_hidden) 

        ### Beta outputs
        ### One output layer
        ### No activation function
        beta = Lambda(lambda x: tf.exp(x), name='beta')(log_beta) 

        ### Time-varying gamma
        gamma_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
        gamma_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(gamma_hidden)

        log_gamma = Dense(1)(gamma_hidden)
        gamma = Lambda(lambda x: tf.exp(x))(log_gamma)
        
        ### Time-varying sigma
        sigma_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
        sigma_hidden = Dense(50, activation = 'tanh', kernel_regularizer=regularizers.l2(1e-5))(sigma_hidden)

        log_sigma = Dense(1)(sigma_hidden)
        sigma = Lambda(lambda x: tf.exp(x))(log_sigma)

        model = Model(inputs=t_input, outputs=[S, E, I, R, beta, gamma, sigma])
        
        return model

    ### Fresh model for each beta scenario
    tf.keras.backend.clear_session()
    model = create_pinn_model()
    ### Print model achitecture
    model.summary()

    ### Define initial conditions
    S0 = tf.constant(100000/100001, dtype=tf.float32)
    E0 = tf.constant(0.0, dtype=tf.float32)
    I0 = tf.constant(1/100001, dtype=tf.float32)
    R0 = tf.constant(0.0, dtype=tf.float32)

    ### Define loss function for PINN
    def loss_function(t_col, t_data_loss, I_data_loss, net, I_scale, smooth_beta=True):
        
        ### if t_col is a 1D array it is reshaped to a column vector
        if len(t_col.shape) == 1:t_col = tf.reshape(t_col, (-1, 1))
        
        ### Convert data to tensors 
        if not isinstance(t_data_loss, tf.Tensor):t_data_loss = tf.convert_to_tensor(t_data_loss, dtype=tf.float32)
        if not isinstance(I_data_loss, tf.Tensor):I_data_loss = tf.convert_to_tensor(I_data_loss, dtype=tf.float32)

        ### reshape arrays to column vectors
        if len(t_data_loss.shape) == 1:t_data_loss = tf.reshape(t_data_loss, (-1, 1))
        if len(I_data_loss.shape) == 1:I_data_loss = tf.reshape(I_data_loss, (-1, 1))
        
        ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
        ### Gradient tape is used to record operations for automatic differentiation
        ### Calculate the gradients of a computation
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t_col)
            S, E, I, R, beta, gamma, sigma = net(t_col)
            
        ### Compute derivatives e.g. dS/dt
        dS_dt = tape.gradient(S, t_col) 
        dE_dt = tape.gradient(E, t_col) 
        dI_dt = tape.gradient(I, t_col) 
        dR_dt = tape.gradient(R, t_col) 
        
        d_beta_dt = tape.gradient(beta, t_col)
        beta_smooth_loss = tf.reduce_mean(tf.square(d_beta_dt)) if smooth_beta else 0.0
        
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
        S_0, E_0, I_0, R_0, _, _, _ = net(t_zero)
        
        Initial_condition_loss = tf.reduce_mean(
            tf.square(S_0 - S0_fixed) +
            tf.square(E_0 - E0_fixed) +
            tf.square(I_0 - I0_fixed) +
            tf.square(R_0 - R0_fixed) )
        
        ### constrain SEIR equations to equal 1
        S, E, I, R, beta, gamma, sigma = net(t_col)
        conservation_loss = tf.reduce_mean(tf.square(S + E + I + R - 1.0))
        
        ### Data loss 
        t_data_normalized = t_data_loss 
        _, _, I_pred, _, _, _, _ = net(t_data_normalized)
        data_loss = tf.reduce_mean(tf.square((I_pred - I_data_loss) / I_scale))
        
        ### Total loss
        total_loss = 1.0 * data_loss+ 0.1*ODE_loss + 1.0 * Initial_condition_loss + 1.0 * conservation_loss + 0.1*beta_smooth_loss
        
        return total_loss, {
            "data_loss": data_loss,
            "IC_loss": Initial_condition_loss,
            "conservation_loss": conservation_loss,
            "ODE_loss": ODE_loss,
        }

    S0_fixed = S0
    E0_fixed = E0
    I0_fixed = I0
    R0_fixed = R0

    ### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
    ### learning rate scheduler added (not in original paper)
    optm = Adam(learning_rate=0.001)

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

    smooth_beta = scenario["smooth_beta"]
    
    @tf.function
    def train_step(t_col, t_data, I_data):
        with tf.GradientTape() as tape:
            loss, loss_dict = loss_function(t_col, t_data, I_data, model, I_scale, smooth_beta)
        grads = tape.gradient(loss, model.trainable_variables)
        optm.apply_gradients(zip(grads, model.trainable_variables))
        return loss, loss_dict
    
    @tf.function
    def test_step(t_col, t_data, I_data):
        return loss_function(t_col, t_data, I_data, model, I_scale, smooth_beta)
    
    print("Starting training...")
    
    ### 50,000 iterations (Qian et al. 2025)
    for itr in range(50000):
        train_loss, train_loss_dict = train_step(t_col_tensor, t_train, I_train)
        train_loss_record.append(float(train_loss))
    
        test_loss, test_loss_dict = test_step(t_col_tensor, t_test, I_test)

        test_loss_record.append(float(test_loss))
    
        if itr % 1000 == 0:
            print(
                f"Iteration {itr}\n"
                f"Train Loss: {float(train_loss):.6f}, "
                f"Test Loss: {float(test_loss):.6f}\n"
                f"Data: {float(train_loss_dict['data_loss']):.6f}, "
                f"IC: {float(train_loss_dict['IC_loss']):.6f}, "
                f"Conservation: {float(train_loss_dict['conservation_loss']):.6f}, "
                f"ODE: {float(train_loss_dict['ODE_loss']):.6f}"
            )
### Predictions
    t_tensor = tf.convert_to_tensor(t_data, dtype=tf.float32)
    _, _, I_pred, _, _, _, _ = model(t_tensor)

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
    plt.title(f'Training Loss (β={beta_true})all learnable parameters')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'PINN_training_loss_beta_{beta_true}all_learnable_parameters90_10.png'))
    plt.show()

    ### Plot PINN prediction vs observed
    plt.figure(figsize=(14, 6))
    plt.plot(t_data_np, I_pred_np, color="#ff7ee3", linewidth=2, label='Infected - PINN prediction')
    plt.plot(t_train_np, I_train_np, color="#004F94", linewidth=2, label='Infected - data')
    plt.plot(t_test_np, I_test_np, color="#004F94", linewidth=2)
    plt.axvline(x=t_train_np[-1], color='gray', linestyle='--', label='Train/Test Split')
    plt.xlabel('Normalised time')
    plt.ylabel('Infected (normalised)')
    plt.title(f'PINN prediction (β={beta_true} - all learnable parameters 90/10 split)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'PINN_beta_{beta_true}all_learnable_parameters90_10.png'))
    plt.show()

    ### Plot estimated beta over time
    t_plot = np.linspace(0.0, 1.0, 500)
    t_plot_tensor = tf.convert_to_tensor(t_plot.reshape(-1, 1), dtype=tf.float32)
    _, _, _, _, beta_pred, _, _ = model.predict(t_plot_tensor)

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot, beta_pred.flatten(), 'g-', linewidth=2, label='β(t) estimated')
    plt.axhline(y=beta_true, color='r', linestyle='--', linewidth=1.5, label=f'β true = {beta_true}')
    plt.xlabel('Normalised time')
    plt.ylabel('β(t)')
    plt.ylim(0, 1)
    plt.title(f'Estimated β(t) vs true β (β={beta_true}) all learnable parameters 90/10 split')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'PINN_parameter_est_beta_{beta_true}all_learnable_parameters90_10.png'))
    plt.show()

    print(f"Finished beta = {beta_true}")