import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

### https://adventuresinpython.blogspot.com/2012/08/fitting-differential-equation-system-to.html

### Initial conditions
E0 = 0 
I0 = 1
R0 = 0
S0 = 10000
days = 20

### Parameters
sigma = 0.3
gamma = 0.3
beta = 0.1

### SEIR equations
def ode_model(t, y, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I 
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def ode_solver(t, initial_conditions, parameters):
    beta, sigma, gamma = parameters

    sol = solve_ivp(
        ode_model,
        (t[0], t[-1]),
        [S0, E0, I0, R0],
        args=(beta, sigma, gamma),
        t_eval=t
    )
    return sol

### Run the model
def run_seir(days, S0, E0, I0, R0, beta, sigma, gamma):
    tspan = np.arange(0, days)
    sol = ode_solver(tspan, [S0, E0, I0, R0], [beta, sigma, gamma])
    
    S, E, I, R = sol.y
    return tspan, S, E, I, R  

### Run simulation
t, S, E, I, R = run_seir(days, S0, E0, I0, R0, beta, sigma, gamma)

N = S0 + E0 + I0 + R0
S_norm = S/N
E_norm = E/N
I_norm = I/N
R_norm = R/N

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
