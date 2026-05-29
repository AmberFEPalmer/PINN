import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os

### Synthetic data generation for PINN
### Simulation parameters — fixed across all scripts 
DAYS  = 100
N_VAL = 100001
SIGMA = 0.25
GAMMA = 0.25
I0 = 1
E0 = 0
R0 = 0
S0 = N_VAL - I0

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

### Beta values to simulate
beta_values = [0.75, 0.5, 0.4]

### Output directories 
output_dir  = "../../png_files"
data_folder = os.path.join("..", "..", "data")

### SEIR model
def ode_model(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt =  beta * S * I / N - sigma * E
    dIdt =  sigma * E - gamma * I
    dRdt =  gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def ode_solver(t, initial_conditions, parameters, N):
    beta, sigma, gamma = parameters
    return solve_ivp(
        ode_model,
        (t[0], t[-1]),
        initial_conditions,
        args=(beta, sigma, gamma, N),
        t_eval=t
    )

def run_seir(days, S0, E0, I0, R0, beta, sigma, gamma, N):
    t = np.linspace(0, days, days + 1)
    sol = ode_solver(t, [S0, E0, I0, R0], [beta, sigma, gamma], N)
    S, E, I, R = sol.y
    return t, S, E, I, R

### Store results for panel plot
results = {}

### Loop over beta values and simulate SEIR dynamics
for beta in beta_values:
    t, S, E, I, R = run_seir(DAYS, S0, E0, I0, R0, beta, SIGMA, GAMMA, N_VAL)

    ### Normalise by population
    S_norm = S / N_VAL
    E_norm = E / N_VAL
    I_norm = I / N_VAL
    R_norm = R / N_VAL

    print(f"\n--- beta = {beta} ---")
    print(t)
    print(I)

    ### Compute R0
    R0_val = beta / GAMMA

    ### Store for panel
    results[beta] = dict(t=t, S=S, E=E, I=I, R=R, R0_val=R0_val)

    ### Individual plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, color=COLOURS["S"], label=r"$S(t)$")
    plt.plot(t, E, color=COLOURS["E"], label=r"$E(t)$")
    plt.plot(t, I, color=COLOURS["I"], label=r"$I(t)$")
    plt.plot(t, R, color=COLOURS["R"], label=r"$R(t)$")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of individuals")
    plt.legend()
    plt.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=plt.gca().transAxes,
        fontsize=13,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'SEIR_constant_beta_{beta}.pdf'), bbox_inches='tight', dpi=300)
    plt.show()

    ### Normalise time for PINN
    t_norm = t / t.max()

    ### Export SEIR results to csv
    SEIR_data = pd.DataFrame({
        "time": t_norm,
        "S": S_norm,
        "E": E_norm,
        "I": I_norm,
        "R": R_norm,
    })
    print(SEIR_data)
    csv_path = os.path.join(data_folder, f"SEIR_data_beta_{beta}.csv")
    SEIR_data.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

### 1x3 panelled figure for all beta values
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

for i, (ax, beta) in enumerate(zip(axes, beta_values)):
    res = results[beta]
    t, S, E, I, R, R0_val = res['t'], res['S'], res['E'], res['I'], res['R'], res['R0_val']

    ax.plot(t, S, color=COLOURS["S"], label=r"$S(t)$")
    ax.plot(t, E, color=COLOURS["E"], label=r"$E(t)$")
    ax.plot(t, I, color=COLOURS["I"], label=r"$I(t)$")
    ax.plot(t, R, color=COLOURS["R"], label=r"$R(t)$")
    ax.set_title(rf"$\beta = {beta}$")

    # R0 annotation — top right
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )

    # Panel label — top left
    ax.text(
        -0.08, 1.02,
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment='bottom',
        horizontalalignment='left',
    )

    ax.legend(fontsize=9)

fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel("Number of individuals", fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "SEIR_beta_panel.pdf"), bbox_inches='tight', dpi=300)
plt.show()
plt.close()