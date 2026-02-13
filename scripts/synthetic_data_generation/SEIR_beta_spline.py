import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from patsy import dmatrix

### https://pubmed.ncbi.nlm.nih.gov/34799850/
### beta varies as a function of time

### Initial conditions
E0, I0, R0, S0 = 0, 1, 0, 100000
N = 100001
days = 100

### Parameters
beta, sigma, gamma = 0.3, 0.1, 0.1

t = np.linspace(0, days, days + 1)
K = 5

### Build spline for beta
spline_basis = dmatrix(
    f"cr(t, df={K}) - 1",
    {"t": t}
)
psi = np.array([-1.2, -0.5, 0.3, -0.8, -1.0])
beta_t = np.exp(spline_basis @ psi)

### Visualise beta over time
plt.plot(t, beta_t)
plt.title("Time-varying transmission rate β(t)")
plt.xlabel("Days")
plt.ylabel("β(t)")
plt.show()

### SEIR model with time varying beta
def ode_model(t, y, beta_t, sigma, gamma, N, t_grid):
    S, E, I, R = y

    ### Find nearest time index
    idx = min(int(np.floor(t)), len(t_grid) - 1)
    beta = beta_t[idx]

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    return [dSdt, dEdt, dIdt, dRdt]

def ode_solver(t, initial_conditions, beta_t, sigma, gamma, N):
    return solve_ivp(
        ode_model,
        (t[0], t[-1]),
        initial_conditions,
        args=(beta_t, sigma, gamma, N, t),
        t_eval=t
    )

### ODE solver
def run_seir(days, S0, E0, I0, R0, beta_t, sigma, gamma, N):
    t = np.linspace(0, days, days + 1)
    sol = ode_solver(
        t,
        [S0, E0, I0, R0],
        beta_t,
        sigma,
        gamma,
        N
    )
    S, E, I, R = sol.y
    return t, S, E, I, R

### Run model
t, S, E, I, R = run_seir(
    days, S0, E0, I0, R0,
    beta_t, sigma, gamma, N
)

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

### Visualisation
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.plot(t, E)
plt.plot(t, I)
plt.plot(t, R)
plt.legend(["S", "E", "I", "R"])
plt.title("SEIR model with time-varying β(t)")
plt.xlabel("Days")
plt.ylabel("Population")
plt.show()

### Normalise time for PINN 
t_norm = t / t.max()

### Export SEIR results to a csv file
SEIR_time_varying_data = pd.DataFrame({"time": t_norm,"I": I_norm,})
print(SEIR_time_varying_data)
SEIR_time_varying_data.to_csv("SEIR_time_varying_results.csv", index=False)