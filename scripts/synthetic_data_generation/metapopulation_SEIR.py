import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os   

### Number of patches
P = 2

### Initial conditions
### 1 infected individual in patch one
### No one infected in patch two
### 50,000 people in each patch
S0 = np.array([49999.0, 50000.0])
E0 = np.array([0.0, 0.0])
I0 = np.array([1.0, 0.0])   
R0 = np.array([0.0, 0.0])

N = S0 + E0 + I0 + R0

### Parameters
beta = np.array([0.75, 0.5])   # different transmission per patch
sigma, gamma = 0.25, 0.25

### Migration matrix (2x2)
### Movement between patches
m = 0.01
M = np.array([
    [-m,  m],
    [ m, -m]
])

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

    ### Migration (matrix form)
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

# Extract results
S = sol.y[0:P]
E = sol.y[P:2*P]
I = sol.y[2*P:3*P]
R = sol.y[3*P:4*P]

# Plot infected in both patches
plt.figure(figsize=(10, 6))
plt.plot(t_eval, I[0], label="Patch 1")
plt.plot(t_eval, I[1], label="Patch 2")
plt.xlabel("Days")
plt.ylabel("Infected")
plt.title("Two-patch SEIR (vectorized form)")
plt.legend()
plt.grid(True)
plt.show()

### Normalise time for PINN
t_norm = t_eval / t_eval.max()
 
S_norm = S / N[:, None]
E_norm = E / N[:, None]
I_norm = I / N[:, None]
R_norm = R / N[:, None]
 
### Export SEIR results to a csv file
SEIR_metapopulation_data = pd.DataFrame({
    "time": t_norm,
    "S1": S_norm[0],
    "S2": S_norm[1],
    "E1": E_norm[0],
    "E2": E_norm[1],
    "I1": I_norm[0],
    "I2": I_norm[1],
    "R1": R_norm[0],
    "R2": R_norm[1],
})
print(SEIR_metapopulation_data)

data_folder = "."  # current directory
csv_path = os.path.join(data_folder, "SEIR_metapopulation_two_patch.csv")
SEIR_metapopulation_data.to_csv(csv_path, index=False)
 