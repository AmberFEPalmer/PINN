import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

### Initial conditions
E0, I0, R0, S0 = 0, 1, 0, 100000
N = 100001
days = 100

### Parameters
beta, sigma, gamma = 0.8, 0.1, 0.1

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

t, S, E, I, R = run_seir(days, S0, E0, I0, R0, beta, sigma, gamma, N)

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
SEIR_data = pd.DataFrame({"time": t_norm,"I": I_norm,})
print(SEIR_data)
SEIR_data.to_csv("SEIR_results.csv", index=False)

