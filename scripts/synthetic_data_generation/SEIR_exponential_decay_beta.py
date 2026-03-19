import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

### Initial conditions
E0, I0, R0, S0 = 0, 1, 0, 100000
N = 100001
days = 100

### Parameters
sigma, gamma = 0.25, 0.25

beta_c = 1.0
beta_1 = 0.8
Lambda = 0.1

def beta_t(t, beta_c, beta_1, Lambda):
    return beta_1 + abs(beta_c - beta_1) * np.exp(-Lambda * t)

### Define SEIR model with exponential decay for beta
def ode_model(t, y, beta_c, beta_1, Lambda, sigma, gamma, N):
    S, E, I, R = y
    
    beta = beta_t(t, beta_c, beta_1, Lambda)
    
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dEdt, dIdt, dRdt]

### ODE solver
def ode_solver(t, initial_conditions, parameters, N):
    beta_c, beta_1, Lambda, sigma, gamma = parameters
    return solve_ivp(
        ode_model,
        (t[0], t[-1]),
        initial_conditions,
        args=(beta_c, beta_1, Lambda, sigma, gamma, N),
        t_eval=t
    )

### Run model
def run_seir(days, S0, E0, I0, R0, beta_c, beta_1, Lambda, sigma, gamma, N):
    t = np.linspace(0, days, days + 1)
    sol = ode_solver(t,[S0, E0, I0, R0], [beta_c, beta_1, Lambda, sigma, gamma],N)
    S, E, I, R = sol.y
    return t, S, E, I, R

t, S, E, I, R = run_seir(days, S0, E0, I0, R0, beta_c, beta_1, Lambda, sigma, gamma, N)

### Visualise beta over time
beta_values = beta_t(t, beta_c, beta_1, Lambda)
plt.plot(t, beta_values)
plt.title("Time-varying transmission rate β(t)")
plt.xlabel("Days")
plt.ylabel("β(t)")
plt.grid()
plt.show()

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
plt.show()

### Normalise time for PINN 
t_norm = t / t.max()

### Export SEIR results to a csv file
data_folder = os.path.join("..", "..", "data")

SEIR_beta_exponential_decay_data = pd.DataFrame({"time": t_norm,"I": I_norm,})
print(SEIR_beta_exponential_decay_data)

csv_path = os.path.join(data_folder, "SEIR_beta_exponential_decay_data.csv")
SEIR_beta_exponential_decay_data.to_csv(csv_path, index=False)