import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os   

### Number of patches
P = 3

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
beta = np.full(P, 0.75)   # same transmission rate in each patch
sigma, gamma = 0.25, 0.25

### Migration matrix — one-way chain: 1 -> 2 ->3
m = 0.01
M = np.zeros((P, P))

### Flow: patch 1 -> patch 2
M[1, 0] = m    # patch 2 receives from patch 1
M[0, 0] = -m   # patch 1 loses outflow

### Flow: patch 2 -> patch 3
M[2, 1] = m    # patch 3 receives from patch 2
M[1, 1] = -m   # patch 2 loses outflow (only sends to patch 3)

### Time
t = 100
t_eval = np.linspace(0, t, t + 1)

def ode_model(t, y):
    S = y[0:P]
    E = y[P:2*P]
    I = y[2*P:3*P]
    R = y[3*P:4*P]

    lambda_ = beta * I / N

    dS = -lambda_ * S
    dE = lambda_ * S - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    dS += M @ S
    dE += M @ E
    dI += M @ I
    dR += M @ R

    return np.concatenate([dS, dE, dI, dR])

### State vector
y0 = np.concatenate([S0, E0, I0, R0])
sol = solve_ivp(ode_model, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval)

### Extract results
S = sol.y[0:P]
E = sol.y[P:2*P]
I = sol.y[2*P:3*P]
R = sol.y[3*P:4*P]

colors = ['purple', 'red', 'green']

### Plot
plt.figure(figsize=(12, 6))
for i in range(P):
    plt.plot(t_eval, I[i], label=f"Patch {i+1}", color=colors[i])
plt.xlabel("Days")
plt.ylabel("Infected")
plt.title("Simple3-patch SEIR model (1->2->3)")
plt.legend()
plt.grid(True)
plt.show()

### Normalise and export
t_norm = t_eval / t_eval.max()
S_norm = S / N[:, None]
E_norm = E / N[:, None]
I_norm = I / N[:, None]
R_norm = R / N[:, None]

data = {"time": t_norm}
for i in range(P):
    data[f"S{i+1}"] = S_norm[i]
    data[f"E{i+1}"] = E_norm[i]
    data[f"I{i+1}"] = I_norm[i]
    data[f"R{i+1}"] = R_norm[i]

df = pd.DataFrame(data)
print(df.head())
csv_path = os.path.join(".", "SimpleSEIR_metapopulation_3_patch.csv")
df.to_csv(csv_path, index=False)
print("Saved to:", csv_path)