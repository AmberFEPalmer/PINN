import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

### Plot style — fixed across all scripts
plt.rcParams.update({
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
}

data_folder = os.path.join("..", "..", "data")
output_dir  = "../../png_files"
os.makedirs(output_dir, exist_ok=True)

N_val = 100001
days_total = 100
GAMMA = 0.25

beta_scenarios = [
    {"label": "beta_0.75", "csv": "SEIR_data_beta_0.75.csv", "beta_true": 0.75},
    {"label": "beta_0.5",  "csv": "SEIR_data_beta_0.5.csv",  "beta_true": 0.5},
    {"label": "beta_0.4",  "csv": "SEIR_data_beta_0.4.csv",  "beta_true": 0.4},
]

### 1x3 panel
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

for i, (ax, scenario) in enumerate(zip(axes, beta_scenarios)):
    label     = scenario["label"]
    beta_true = scenario["beta_true"]
    R0_val    = beta_true / GAMMA

    ### Load ground truth
    data     = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
    t_unnorm = data["time"].values * days_total
    I_true   = data["I"].values   * N_val

    ### Load PINN predictions
    pred     = pd.read_csv(os.path.join(output_dir, f"PINN_predictions_{label}_80_20.csv"))
    I_pred   = pred["I_pred"].values * N_val

    ### Train/test split
    split          = int(0.8 * len(t_unnorm))
    t_train_unnorm = t_unnorm[:split]

    ### Ground truth — dotted coloured scatter
    ax.scatter(t_unnorm, I_true, color=COLOURS["I"],
               s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")

    ### PINN prediction — solid black
    ax.plot(t_unnorm, I_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$I(t)$ PINN")

    ### Train/test split line
    ax.axvline(x=t_train_unnorm[-1], color='gray', linestyle='--',
               linewidth=1.5, label='Train/Test Split')

    ax.set_title(rf"$\beta = {beta_true}$")

    ### R0 annotation — top right
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )

    ### Panel label — top left
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
fig.supylabel("Number of infected individuals", fontsize=13, x=0.01)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_beta_panel.pdf"), bbox_inches='tight', dpi=300)
plt.show()
plt.close()

data_folder = os.path.join("..", "..", "data")
output_dir  = "../../png_files"
os.makedirs(output_dir, exist_ok=True)

N_val = 100001
days_total = 100
BETA = 0.75
GAMMA = 0.25
R0_val = BETA / GAMMA

panel_levels = [5, 10, 15, 20]

### 2x2 panel
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for i, (ax, noise_percent) in enumerate(zip(axes.flat, panel_levels)):
    label = f"Gaussian_noise_{noise_percent}percent"

    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, f"SEIR_Gaussian_noise_{noise_percent}percent.csv"))
    t_unnorm = data["time"].values * days_total
    I_true = data["I"].values   * N_val

    ### Load PINN predictions
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_predictions_{label}_80_20.csv"))
    I_pred = pred["I_pred"].values * N_val

    ### Train/test split
    split = int(0.8 * len(t_unnorm))
    t_train_unnorm = t_unnorm[:split]

    ### Ground truth — dotted coloured scatter
    ax.scatter(t_unnorm, I_true, color=COLOURS["I"],
               s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")

    ### PINN prediction — solid black
    ax.plot(t_unnorm, I_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$I(t)$ PINN")

    ### Train/test split line
    ax.axvline(x=t_train_unnorm[-1], color='gray', linestyle='--',
               linewidth=1.5, label='Train/Test Split')

    ax.set_title(f"{noise_percent}% Gaussian Noise")

    ### R0 annotation — top right
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )

    ### Panel label — top left
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
fig.supylabel("Number of infected individuals", fontsize=13, x=0.01)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_Gaussian_noise_panel.pdf"), bbox_inches='tight', dpi=300)
plt.show()
plt.close()