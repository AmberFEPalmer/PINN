import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

### Synthetic data generation for PINN

### Initial conditions
E0, I0, R0, S0 = 0, 1, 0, 100000
N = 100001
days = 100

### Parameters
beta_values = [0.75, 0.5, 0.4]
sigma, gamma = 0.25, 0.25

### SEIR model
def ode_model(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

### ODE solver
def ode_solver(t, initial_conditions, parameters, N):
    beta, sigma, gamma = parameters
    return solve_ivp(
        ode_model,
        (t[0], t[-1]),
        initial_conditions,
        args=(beta, sigma, gamma, N),
        t_eval=t
    )

### Run model
def run_seir(days, S0, E0, I0, R0, beta, sigma, gamma, N):
    t = np.linspace(0, days, days + 1)
    sol = ode_solver(t, [S0, E0, I0, R0], [beta, sigma, gamma], N)
    S, E, I, R = sol.y
    return t, S, E, I, R

### Output directories
output_dir = "../../png_files"
data_folder = os.path.join("..", "..", "data")

### Loop over beta values
for beta in beta_values:

    t, S, E, I, R = run_seir(days, S0, E0, I0, R0, beta, sigma, gamma, N)

    ### Normalize
    N_norm = S0 + E0 + I0 + R0
    S_norm = S / N_norm
    E_norm = E / N_norm
    I_norm = I / N_norm
    R_norm = R / N_norm

    ### Print values for t and I
    print(f"\n--- beta = {beta} ---")
    print(t)
    print(I)

    ### Plot results from SEIR model
    plt.figure(figsize=(10, 6))
    plt.plot(t, S)
    plt.plot(t, E)
    plt.plot(t, I)
    plt.plot(t, R)
    plt.legend(["S", "E", "I", "R"])
    plt.title(f"SEIR model (β={beta})")
    plt.xlabel("Days")
    plt.ylabel("Number of people in each compartment")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'SEIR_constant_beta_{beta}.png'))
    plt.show()

    ### Normalise time for PINN
    t_norm = t / t.max()

    ### Export SEIR results to a csv file
    SEIR_data = pd.DataFrame({
    "time": t_norm,
    "S": S_norm,
    "E": E_norm,
    "I": I_norm,
    "R": R_norm
    })
    print(SEIR_data)
    csv_path = os.path.join(data_folder, f"SEIR_data_beta_{beta}.csv")
    SEIR_data.to_csv(csv_path, index=False)
    
    
    from scipy.optimize import minimize

def loss(params, t, I_data):
    beta = params[0]
    _, _, _, I_pred, _ = run_seir(days, S0, E0, I0, R0, beta, sigma, gamma, N)
    I_pred_norm = I_pred / N
    return np.mean((I_pred_norm - I_data)**2)

result = minimize(loss, x0=[0.3], args=(t_norm, I_noisy))
beta_est = result.x[0]