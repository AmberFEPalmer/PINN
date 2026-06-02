import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os   

### Number of patches
P = 5

### Initial conditions
S0 = np.full(P, 50000.0)
E0 = np.zeros(P)
I0 = np.zeros(P)
R0 = np.zeros(P)

### Seed infection in patch 1
I0[0] = 1.0
S0[0] -= 1.0

N = S0 + E0 + I0 + R0

### Parameters
beta = np.linspace(0.6, 0.8, P)   # heterogeneous transmission rates
sigma, gamma = 0.25, 0.25

### Migration matrix (10x10)
m = 0.01
M = np.full((P, P), m / (P - 1))   # distribute outward flow evenly
np.fill_diagonal(M, -m)

### Time
t = 100
t_eval = np.linspace(0, t, t + 1)

def ode_model(t, y):
    ### Unpack state vector
    S = y[0:P]
    E = y[P:2*P]
    I = y[2*P:3*P]
    R = y[3*P:4*P]

    ### Force of infection
    lambda_ = beta * I / N

    ### SEIR dynamics
    dS = -lambda_ * S
    dE = lambda_ * S - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    ### Migration
    dS += M @ S
    dE += M @ E
    dI += M @ I
    dR += M @ R

    return np.concatenate([dS, dE, dI, dR])

### Initial state vector
y0 = np.concatenate([S0, E0, I0, R0])

sol = solve_ivp(
    ode_model,
    (t_eval[0], t_eval[-1]),
    y0,
    t_eval=t_eval
)

### Extract results
S = sol.y[0:P]
E = sol.y[P:2*P]
I = sol.y[2*P:3*P]
R = sol.y[3*P:4*P]

### Plot infected in all patches
plt.figure(figsize=(12, 6))

for i in range(P):
    plt.plot(t_eval, I[i], label=f"Patch {i+1}")

plt.xlabel("Days")
plt.ylabel("Infected")
plt.legend(ncol=2)
plt.grid(True)
plt.show()

### Normalise time for PINN
t_norm = t_eval / t_eval.max()

S_norm = S / N[:, None]
E_norm = E / N[:, None]
I_norm = I / N[:, None]
R_norm = R / N[:, None]

### Export to CSV
data = {"time": t_norm}

for i in range(P):
    data[f"S{i+1}"] = S_norm[i]
    data[f"E{i+1}"] = E_norm[i]
    data[f"I{i+1}"] = I_norm[i]
    data[f"R{i+1}"] = R_norm[i]

SEIR_metapopulation_data = pd.DataFrame(data)

print(SEIR_metapopulation_data.head())

data_folder = "."
csv_path = os.path.join(data_folder, "SEIR_metapopulation_5_patch.csv")
SEIR_metapopulation_data.to_csv(csv_path, index=False)

print("Saved to:", csv_path)