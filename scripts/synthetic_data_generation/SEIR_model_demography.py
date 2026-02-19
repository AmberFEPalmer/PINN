import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

### Initial conditions
E0, I0, R0, S0 = 0, 1, 0, 67000099
N = 67000000
days = 150
births = 701000
deaths = 669000
b = births / (N * 365)
d = deaths / (N * 365)

### https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/bulletins/annualmidyearpopulationestimates/mid2020#:~:text=The%20population%20of%20the%20UK,population%20growth%20in%20this%20period.
### 701,000 births in UK 2020
### 669,000 deaths in UK 2020

### Parameters
beta, sigma, gamma = 0.8, 0.1, 0.1

### SEIR model
def ode_model(t, y, beta, sigma, gamma, b, d, N):
    S, E, I, R = y
    dSdt = b * N - (beta * S * I / N) - d * S
    dEdt = (beta * S * I / N) - sigma * E - d * E
    dIdt = sigma * E - (gamma + d) * I
    dRdt = gamma * I - d * R
    return [dSdt, dEdt, dIdt, dRdt]

### ODE solver
def ode_solver(t, initial_conditions, parameters, N):
    beta, sigma, gamma, b, d = parameters
    return solve_ivp(
        ode_model,
        (t[0], t[-1]),
        initial_conditions,
        args=(beta, sigma, gamma, b, d, N),
        t_eval=t
    )

### Run model
def run_seir(days, S0, E0, I0, R0, beta, sigma, gamma, b, d, N):
    t = np.linspace(0, days, days + 1)
    sol = ode_solver(t, [S0, E0, I0, R0], [beta, sigma, gamma, b, d], N)
    S, E, I, R = sol.y
    return t, S, E, I, R

t, S, E, I, R = run_seir(days, S0, E0, I0, R0, beta, sigma, gamma, b, d, N)

### Normalize
N_t = S + E + I + R
S_norm = S / N_t
E_norm = E / N_t
I_norm = I / N_t
R_norm = R / N_t

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
plt.title("SEIR model with demography")
plt.xlabel("Days")
plt.ylabel("Number of people in each compartment")
plt.show()

### Normalise time for PINN 
t_norm = t / t.max()

### Export SEIR results to a csv file
data_folder = os.path.join("..", "..", "data")

SEIR_demography_data = pd.DataFrame({"time": t_norm,"I": I_norm,})
print(SEIR_demography_data)

csv_path = os.path.join(data_folder, "SEIR_demography_data.csv")
SEIR_demography_data.to_csv(csv_path, index=False)