import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from patsy import dmatrix
import os

### https://pubmed.ncbi.nlm.nih.gov/34799850/
### beta varies as a function of time

### Simulation parameters — fixed across all scripts 
DAYS = 100
N_VAL = 100001
SIGMA = 0.25
GAMMA = 0.25
I0 = 1
E0 = 0
R0 = 0
S0 = N_VAL - I0

t = np.linspace(0, DAYS, DAYS + 1)

### Plot style — fixed across all scripts
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLOURS = {
    "S": "#2ca02c",
    "E": "#ff7f0e",
    "I": "#d62728",
    "R": "#1f33b4",
    "beta":"#008094",
}

### Scenario definitions
### psi values kept negative or slightly positive so beta = exp(spline @ psi)
### stays in biologically realistic range (roughly 0.1 - 1.0)
### R0 = beta/gamma so beta = 0.75 -> R0 = 3, beta = 0.25 -> R0 = 1
scenarios = [
    ### Original spline scenario
    {
        "label": "beta_spline",
        "psi": [-1.2, -0.5, 0.3, -0.8, -1.0],
        "K": 5,
        "title": "Original spline",
    },
    ### Easy: slow gentle rise then fall
    {
        "label": "beta_spline_slow_rise",
        "psi": [-1.5, -1.0, -0.5, -1.0, -1.5],
        "K": 5,
        "title": "Slow rise and fall",
    },
    ### Easy: gradual decline (intervention effect)
    {
        "label": "beta_spline_gradual_decline",
        "psi": [-0.5, -0.8, -1.1, -1.4, -1.7],
        "K": 5,
        "title": "Gradual decline",
    },
    ### Easy: single small pulse
    {
        "label": "beta_spline_single_pulse",
        "psi": [-1.5, -1.0, -0.3, -1.0, -1.5],
        "K": 5,
        "title": "Single pulse",
    },
    ### Medium: two modest waves
    {
        "label": "beta_spline_two_waves",
        "psi": [-1.2, -0.4, -1.2, -0.4, -1.2, -0.4, -1.2],
        "K": 7,
        "title": "Two waves",
    },
    ### Medium: three waves
    {
        "label": "beta_spline_three_waves",
        "psi": [-1.2, -0.3, -1.2, -0.3, -1.2, -0.3, -1.2],
        "K": 7,
        "title": "Three waves",
    },
    ### Hard: rapid oscillations (4 peaks)
    {
        "label": "beta_spline_rapid",
        "psi": [-1.2, -0.3, -1.2, -0.3, -1.2, -0.3, -1.2, -0.3, -1.2],
        "K": 9,
        "title": "Rapid oscillations (4 peaks)",
    },
    ### Hard: very rapid oscillations (6 peaks)
    {
        "label": "beta_spline_very_rapid",
        "psi": [-1.2, -0.3, -1.2, -0.3, -1.2, -0.3, -1.2, -0.3, -1.2, -0.3, -1.2, -0.3, -1.2],
        "K": 13,
        "title": "Very rapid oscillations (6 peaks)",
    },
    ### Hard: escalating amplitude
    {
        "label": "beta_spline_escalating",
        "psi": [-1.5, -1.2, -1.0, -0.7, -0.5, -0.3, -0.5, -0.3, -0.5],
        "K": 9,
        "title": "Escalating amplitude",
    },
    ### Hard: damped oscillations
    {
        "label": "beta_spline_damped",
        "psi": [-0.3, -1.2, -0.3, -0.8, -0.5, -1.0, -0.7, -1.2, -1.4],
        "K": 9,
        "title": "Damped oscillations",
    },
]

output_dir  = "../../png_files"
data_folder = os.path.join("..", "..", "data")

### SEIR model with time-varying beta 
def ode_model(t, y, beta_t, sigma, gamma, N, t_grid):
    S, E, I, R = y
    idx = min(int(np.floor(t)), len(t_grid) - 1)
    beta = beta_t[idx]
    dSdt = -beta * S * I / N
    dEdt =  beta * S * I / N - sigma * E
    dIdt =  sigma * E - gamma * I
    dRdt =  gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def ode_solver(t, initial_conditions, beta_t, sigma, gamma, N):
    return solve_ivp(
        ode_model,
        (t[0], t[-1]),
        initial_conditions,
        args=(beta_t, sigma, gamma, N, t),
        t_eval=t
    )

def run_seir(days, S0, E0, I0, R0, beta_t, sigma, gamma, N):
    t = np.linspace(0, days, days + 1)
    sol = ode_solver(t, [S0, E0, I0, R0], beta_t, sigma, gamma, N)
    S, E, I, R = sol.y
    return t, S, E, I, R

### Scenario loop
for sc in scenarios:
    label = sc["label"]
    psi = sc["psi"]
    K = sc["K"]

    ### Build spline for beta
    spline_basis = dmatrix(f"cr(t, df={K}) - 1", {"t": t})
    beta_t = np.exp(np.array(spline_basis) @ np.array(psi))

    print(f"\n{label}: beta range [{beta_t.min():.3f}, {beta_t.max():.3f}]")

    ### Visualise beta over time
    plt.figure(figsize=(8, 4))
    plt.plot(t, beta_t, color=COLOURS["beta"])
    plt.xlabel("Time (days)")
    plt.ylabel(r"$\beta(t)$")
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"beta_{label}.pdf"), bbox_inches='tight', dpi=300)
    plt.show()

    ### Run model
    t_run, S, E, I, R = run_seir(DAYS, S0, E0, I0, R0, beta_t, SIGMA, GAMMA, N_VAL)

    ### Normalise by population
    S_norm = S / N_VAL
    E_norm = E / N_VAL
    I_norm = I / N_VAL
    R_norm = R / N_VAL

    ### Visualise SEIR compartments
    plt.figure(figsize=(10, 6))
    plt.plot(t_run, S, color=COLOURS["S"], label=r"$S(t)$")
    plt.plot(t_run, E, color=COLOURS["E"], label=r"$E(t)$")
    plt.plot(t_run, I, color=COLOURS["I"], label=r"$I(t)$")
    plt.plot(t_run, R, color=COLOURS["R"], label=r"$R(t)$")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of individuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"SEIR_{label}.pdf"), bbox_inches='tight', dpi=300)
    plt.show()

    ### Normalise time for PINN
    t_norm = t_run / t_run.max()

    ### Export SEIR results to csv
    SEIR_time_varying_data = pd.DataFrame({
        "time": t_norm,
        "S": S_norm,
        "E": E_norm,
        "I": I_norm,
        "R": R_norm,
        "beta": beta_t,
    })
    print(SEIR_time_varying_data)
    csv_path = os.path.join(data_folder, f"{label}.csv")
    SEIR_time_varying_data.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")