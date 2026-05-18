import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

### Initial conditions
E0, I0, R0, S0 = 0, 1, 0, 100000
N = 100001
days = 100
t = np.linspace(0, days, days + 1)

### Parameters
sigma, gamma = 0.25, 0.25

### Initial beta and new beta after day 30
beta0 = 0.75
beta1 = 0.55
change_day = 92
beta_t = np.where(t < change_day, beta0, beta1)

def beta_func(t):
    if t < change_day:
        return beta0
    else:
        return beta1

### SEIR model
def ode_model(t, y, sigma, gamma, N):
    S, E, I, R = y
    beta = beta_func(t)  # beta now changes at day 30
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

output_dir = "../../png_files"

### Visualise beta over time
plt.plot(t, beta_t)
plt.title("Time-varying transmission rate β(t)")
plt.xlabel("Days")
plt.ylabel("β(t)")
plt.ylim(0, 1)  
plt.savefig(os.path.join(output_dir, 'Beta_peicewise.png'))
plt.show()

### ODE solver
def ode_solver(t, initial_conditions, sigma, gamma, N):
    return solve_ivp(
        ode_model,
        (t[0], t[-1]),
        initial_conditions,
        args=(sigma, gamma, N),
        t_eval=t
    )

### Run model
def run_seir(days, S0, E0, I0, R0, sigma, gamma, N):
    t = np.linspace(0, days, days + 1)
    sol = ode_solver(t, [S0, E0, I0, R0], sigma, gamma, N)
    S, E, I, R = sol.y
    return t, S, E, I, R

t, S, E, I, R = run_seir(days, S0, E0, I0, R0, sigma, gamma, N)

### Normalize
N = S0 + E0 + I0 + R0
S_norm = S / N
E_norm = E / N
I_norm = I / N
R_norm = R / N

### Print values for t and I 
print(t)
print(I)
print(t, I)
type(t)

output_dir = "../../png_files"

### Plot results from SEIR model
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.plot(t, E)
plt.plot(t, I)
plt.plot(t, R)
plt.legend(["S", "E", "I", "R"])
plt.title("SEIR model")
plt.xlabel("Days")
plt.ylabel("Number of people in each compartment")
plt.savefig(os.path.join(output_dir, 'SEIR_beta_peicewise.png'))
plt.show()

### Normalise time for PINN 
t_norm = t / t.max()

### Export SEIR results to a csv file
data_folder = os.path.join("..", "..", "data")

SEIR__beta_peicewise_data = pd.DataFrame({"time": t_norm,"I": I_norm,})

print(SEIR__beta_peicewise_data)

csv_path = os.path.join(data_folder, "SEIR_beta_peicewise.csv")
SEIR__beta_peicewise_data.to_csv(csv_path, index=False)