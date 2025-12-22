import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

### Population of England
N = 56_000_000

### Initial conditions
E0 = 0 
I0 = 100
R0 = 0
S0 = N - I0 - R0 - E0 

### Parameters
sigma = 1/5.2
gamma = 1/2.9
beta = 0.4

### SEIR equations
def ode_model(t, y, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def ode_solver(t, initial_conditions, parameters):
    S0, E0, I0, R0 = initial_conditions
    beta, sigma, gamma = parameters

    sol = solve_ivp(
        ode_model,
        (t[0], t[-1]),
        [S0, E0, I0, R0],
        args=(N, beta, sigma, gamma),
        t_eval=t
    )
    return sol

def run_seir(days, S0, E0, I0, R0, beta, sigma, gamma):
    tspan = np.arange(0, days, 1)
    sol = ode_solver(tspan, [S0, E0, I0, R0], [beta, sigma, gamma])
    S, E, I, R = sol.y
    return tspan, I  # return I(t)

### Load real data
data = pd.read_csv("covid_england_2020.csv")

### See when the data starts and ends
print("Start date:", data['date'].min())
print("End date:", data['date'].max())

### Plot
plt.figure(figsize=(12,6))
plt.plot(t_model, I_model, label="Model: Active Infections I(t)", linewidth=2)
plt.scatter(t_data, I_data_new_cases, s=10, color='red',
            label="Observed: Daily New Cases")
plt.xlabel("Days since first observation")
plt.ylabel("Count")
plt.title("OPTION A: SEIR I(t) vs Observed Daily New Cases (England)")
plt.legend()
plt.grid(True)
plt.show()
