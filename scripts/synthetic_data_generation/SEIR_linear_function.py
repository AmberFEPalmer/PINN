import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

### Synthetic data generation for PINN
### Initial conditions
# I0=50 represents realistic community seeding at outbreak start
E0, I0, R0, S0 = 0, 10, 0, 99990
N = 100000
days = 100

### Parameters
sigma, gamma = 0.25, 0.25

### Time-varying beta(t) with COVID-realistic Rt values.
### Rt = beta(t) / gamma, so beta(t) = Rt(t) * gamma.
### Target Rt profile:
###   t=0:   Rt ~ 2.0  -> consistent with early COVID (no immunity, some awareness)
###   t=22:  Rt ~ 2.8  -> peak transmission (Alpha/Delta range)
###   t=57:  Rt ~ 0.7  -> suppression via lockdown/interventions
###   t=78:  Rt ~ 1.0  -> gradual relaxation to endemic plateau
###   t>78:  Rt ~ 1.0  -> sustained low-level transmission
def beta_t(t):
    t_knots  = np.array([0,    10,    22,    40,    57,    70,    78,    100,   140])
    Rt_knots = np.array([2.0,  2.4,   2.8,   2.0,   0.7,   0.9,   1.0,   1.0,   1.0])
    return np.interp(t, t_knots, Rt_knots * gamma)

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
        max_step=0.5          # small step for accuracy with time-varying beta
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

### Compute Rt over time
Rt = beta_t(t) / gamma

### --- Plot 1: Rt over time (matches the image) ---
plt.figure(figsize=(7, 4))
plt.plot(t, Rt, color='black', linewidth=1.5)
plt.axhline(1.0, color='black', linestyle='--', linewidth=1)
plt.xlabel("x")
plt.ylabel("reproduction number")
plt.xlim(0, 100)
plt.ylim(0.7, 3.0)
plt.grid(True, color='lightgrey', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Rt_time_varying.png'), dpi=150)
plt.show()

### --- Plot 2: SEIR compartments ---
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
print(f"Rt range: {Rt.min():.3f} – {Rt.max():.3f}")

### Export to CSV
SEIR_data = pd.DataFrame({
    "time": t_norm,
    "S": S_norm,
    "E": E_norm,
    "I": I_norm,
    "R": R_norm,
    "Rt": Rt
})
print(SEIR_data.head(10))

csv_path = os.path.join(data_folder, "SEIR_140_days_time_varying_beta.csv")
SEIR_data.to_csv(csv_path, index=False)
print(f"\nCSV saved to {csv_path}")