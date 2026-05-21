import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

### Synthetic data generation for PINN
### Initial conditions
E0, I0, R0, S0 = 0, 10, 0, 99990
N = 100000
days = 100

### Parameters
sigma, gamma = 0.25, 0.25

### Time-varying beta(t) defined directly.
def beta_t(t):
    t_knots    = np.array([0, 10, 22, 40, 57, 70, 78, 100, 140])
    beta_knots = np.array([0.500, 0.600, 0.700, 0.500,  0.175,  0.225, 0.250, 0.250, 0.250])
    return np.interp(t, t_knots, beta_knots)

### SEIR model with time-varying beta
def ode_model_tv(t, y, sigma, gamma, N):
    S, E, I, R = y
    beta = beta_t(t)
    dSdt = -beta * S * I / N
    dEdt =  beta * S * I / N - sigma * E
    dIdt =  sigma * E - gamma * I
    dRdt =  gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

### ODE solver
def run_seir_tv(days, S0, E0, I0, R0, sigma, gamma, N):
    t = np.linspace(0, days, days + 1)
    sol = solve_ivp(
        ode_model_tv,
        (t[0], t[-1]),
        [S0, E0, I0, R0],
        args=(sigma, gamma, N),
        t_eval=t,
        max_step=0.5
    )
    S, E, I, R = sol.y
    return t, S, E, I, R

### Output directories
output_dir = "../../png_files"
data_folder = os.path.join("..", "..", "data")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

### Run model
t, S, E, I, R = run_seir_tv(days, S0, E0, I0, R0, sigma, gamma, N)

### Compute beta and Rt over time
beta_vals = beta_t(t)
Rt = beta_vals / gamma

### Plot beta over time
plt.figure(figsize=(7, 4))
plt.plot(t, beta_vals, color='black', linewidth=1.5)
plt.xlabel("Days")
plt.ylabel("β(t)")
plt.xlim(0, 100)
plt.ylim(0.1, 0.8)
plt.grid(True, color='lightgrey', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'beta_time_varying.png'), dpi=150)
plt.show()

### Plot SEIR compartments
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='S')
plt.plot(t, E, label='E')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.legend()
plt.title("SEIR model (time-varying β)")
plt.xlabel("Days")
plt.ylabel("Number of people in each compartment")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'SEIR_140_days_time_varying_beta.png'), dpi=150)
plt.show()

### Normalise
N_norm = S0 + E0 + I0 + R0
t_norm = t / t.max()
S_norm = S / N_norm
E_norm = E / N_norm
I_norm = I / N_norm
R_norm = R / N_norm

### Print summary
print("--- Time-varying beta SEIR ---")
print(f"Peak I: {I.max():.1f} at day {t[np.argmax(I)]:.0f}")
print(f"beta range: {beta_vals.min():.3f} – {beta_vals.max():.3f}")
print(f"Rt range:   {Rt.min():.3f} – {Rt.max():.3f}")

### Export to CSV
SEIR_data = pd.DataFrame({
    "time": t_norm,
    "S": S_norm,
    "E": E_norm,
    "I": I_norm,
    "R": R_norm,
    "beta": beta_vals,
    "Rt": Rt
})
print(SEIR_data.head(10))
csv_path = os.path.join(data_folder, "SEIR_140_days_time_varying_beta.csv")
SEIR_data.to_csv(csv_path, index=False)
print(f"\nCSV saved to {csv_path}")