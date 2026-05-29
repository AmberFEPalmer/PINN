import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

### Simulation parameters — fixed across all scripts 
DAYS  = 100
N_VAL = 100001
SIGMA = 0.25
GAMMA = 0.25
BETA  = 0.75
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

### Output directories 
output_dir  = "../../png_files"
data_folder = os.path.join("..", "..", "data")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(output_dir,  exist_ok=True)

### Compute R0
R0_val = BETA / GAMMA

### SEIR simulation 
dt    = 1.0
t_arr = np.arange(0, DAYS + dt, dt)
S, E, I, R = S0, E0, I0, R0
S_list, E_list, I_list, R_list = [S], [E], [I], [R]

for _ in t_arr[1:]:
    dS = -BETA * S * I / N_VAL * dt
    dE = (BETA * S * I / N_VAL - SIGMA * E) * dt
    dI = (SIGMA * E - GAMMA * I) * dt
    dR = GAMMA * I * dt
    S  += dS;  E += dE;  I += dI;  R += dR
    S_list.append(S);  E_list.append(E)
    I_list.append(I);  R_list.append(R)

S_array = np.array(S_list)
E_array = np.array(E_list)
I_array = np.array(I_list)
R_array = np.array(R_list)

### Pre-generate noisy I for all levels (for reproducibility across plots)
np.random.seed(42)
noisy_I = {}
for noise_percent in range(1, 21):
    noise_level = noise_percent / 100
    noisy_I[noise_percent] = I_array + noise_level * I_array * np.random.normal(0, 1, len(I_array))

### Individual plots for all 20 noise levels
for noise_percent in range(1, 21):
    I_noisy = noisy_I[noise_percent]

    ### Export as normalised fractions
    df = pd.DataFrame({
        "time": t_arr  / DAYS,
        "S": S_array / N_VAL,
        "E": E_array / N_VAL,
        "I": I_noisy / N_VAL,
        "R": R_array / N_VAL,
    })
    csv_path = os.path.join(data_folder, f"SEIR_Gaussian_noise_{noise_percent}percent.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, S_array, color=COLOURS["S"], label=r"$S(t)$")
    plt.plot(t_arr, E_array, color=COLOURS["E"], label=r"$E(t)$")
    plt.plot(t_arr, R_array, color=COLOURS["R"], label=r"$R(t)$")
    plt.plot(t_arr, I_array, color=COLOURS["I"], label=r"$I(t)$ clean")
    plt.scatter(t_arr, I_noisy, color=COLOURS["I"], s=10, alpha=0.5, zorder=3,
                label=fr"$I(t)$ {noise_percent}% noise")
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
    plt.savefig(os.path.join(output_dir, f"SEIR_Gaussian_noise_{noise_percent}percent.pdf"),
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

### 2x2 panelled figure for 5%, 10%, 15%, 20%
panel_levels = [5, 10, 15, 20]
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for i, (ax, noise_percent) in enumerate(zip(axes.flat, panel_levels)):
    I_noisy = noisy_I[noise_percent]

    ax.plot(t_arr, S_array, color=COLOURS["S"], label=r"$S(t)$")
    ax.plot(t_arr, E_array, color=COLOURS["E"], label=r"$E(t)$")
    ax.plot(t_arr, R_array, color=COLOURS["R"], label=r"$R(t)$")
    ax.plot(t_arr, I_array, color=COLOURS["I"], label=r"$I(t)$ clean")
    ax.scatter(t_arr, I_noisy, color=COLOURS["I"], s=8, alpha=0.4, zorder=3,
               label=fr"$I(t)$ {noise_percent}% noise")

    ax.set_title(fr"{noise_percent}% Gaussian Noise")
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )

    # Panel label — to the side
    ax.text(
        -0.08, 1.02,
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment='bottom',
        horizontalalignment='left',
    )

    ax.legend(fontsize=9)

### Shared axis labels
fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel("Number of individuals", fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "SEIR_Gaussian_noise_panel.pdf"), bbox_inches='tight', dpi=300)
plt.show()
plt.close()