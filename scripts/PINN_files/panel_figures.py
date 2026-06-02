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
    "beta":"#008094",
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
    label = scenario["label"]
    beta_true = scenario["beta_true"]
    R0_val = beta_true / GAMMA

    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
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

data_folder = os.path.join("..", "..", "data")
output_dir  = "../../png_files"
os.makedirs(output_dir, exist_ok=True)
 
N_val = 100001
days_total = 100
GAMMA = 0.25
 
beta_scenarios = [
    {"label": "beta_0.75", "csv": "SEIR_data_beta_0.75.csv", "beta_true": 0.75},
    {"label": "beta_0.5", "csv": "SEIR_data_beta_0.5.csv", "beta_true": 0.5},
    {"label": "beta_0.4", "csv": "SEIR_data_beta_0.4.csv", "beta_true": 0.4},
]
 
### 1x3 panel
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
 
for i, (ax, scenario) in enumerate(zip(axes, beta_scenarios)):
    label = scenario["label"]
    beta_true = scenario["beta_true"]
    R0_val = beta_true / GAMMA
 
    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
    t_unnorm = data["time"].values * days_total
    I_true = data["I"].values   * N_val
 
    ### Load PINN predictions
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_all_learnable_predictions_{label}_80_20.csv"))
    I_pred = pred["I_pred"].values * N_val
 
    ### Train/test split
    split = int(0.8 * len(t_unnorm))
    t_train_unnorm = t_unnorm[:split]
 
    ### Ground truth — coloured scatter
    ax.scatter(t_unnorm, I_true, color=COLOURS["I"],
               s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")
 
    ### PINN prediction — solid dark grey
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
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_beta_panel_80_20.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_beta_panel_80_20.png"), bbox_inches='tight', dpi=300)
plt.show()
 
print("Panel saved.")

beta_scenarios = [
    {"label": "beta_0.75", "csv": "SEIR_data_beta_0.75.csv", "beta_true": 0.75},
    {"label": "beta_0.5", "csv": "SEIR_data_beta_0.5.csv", "beta_true": 0.5},
    {"label": "beta_0.4", "csv": "SEIR_data_beta_0.4.csv", "beta_true": 0.4},
]
 
### Beta parameter reconstruction - all parameters learnt 
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
 
for i, (ax, scenario) in enumerate(zip(axes, beta_scenarios)):
    label = scenario["label"]
    beta_true = scenario["beta_true"]
    R0_val = beta_true / GAMMA
 
    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
    t_unnorm = data["time"].values * days_total
 
    ### Load all-learnable predictions CSV
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_all_learnable_predictions_{label}_100_0.csv"))
    beta_pred = pred["beta_pred"].values
 
    ### Estimated beta — solid dark grey line
    ax.plot(t_unnorm, beta_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$\hat{\beta}(t)$ estimated")
 
    ### True beta — scatter
    ax.scatter(t_unnorm, np.full_like(t_unnorm, beta_true),
               color="#1f33b4", s=30, alpha=0.3, zorder=3,
               label=rf"$\beta$ true = {beta_true}")
 
    ax.set_title(rf"$\beta = {beta_true}$")
    ax.set_ylim(0, 3)
 
    ### R0 annotation
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )
 
    ### Panel label
    ax.text(
        -0.08, 1.02,
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment='bottom',
        horizontalalignment='left',
    )
 
    ax.legend(fontsize=9)
 
axes[0].set_ylabel("β(t)", fontsize=13)
fig.supxlabel("Time (days)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_beta_est_all_learnable_panel_100_0.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_beta_est_all_learnable_panel_100_0.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel 1 saved: all parameters learned beta estimation (100/0).")
 
### beta parameter reconstruction - sigma/gamma fixed
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
 
for i, (ax, scenario) in enumerate(zip(axes, beta_scenarios)):
    label     = scenario["label"]
    beta_true = scenario["beta_true"]
    R0_val    = beta_true / GAMMA
 
    ### Load ground truth
    data     = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
    t_unnorm = data["time"].values * days_total
 
    ### Load fixed-params predictions CSV
    pred      = pd.read_csv(os.path.join(output_dir, f"PINN_predictions_{label}_100_10.csv"))
    beta_pred = pred["beta_pred"].values
 
    ### Estimated beta — solid dark grey line
    ax.plot(t_unnorm, beta_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$\hat{\beta}(t)$ estimated")
 
    ### True beta — scatter
    ax.scatter(t_unnorm, np.full_like(t_unnorm, beta_true),
               color="#1f33b4", s=30, alpha=0.3, zorder=3,
               label=rf"$\beta$ true = {beta_true}")
 
    ax.set_title(rf"$\beta = {beta_true}$")
    ax.set_ylim(0, 3)
 
    ### R0 annotation
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )
 
    ### Panel label
    ax.text(
        -0.08, 1.02,
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment='bottom',
        horizontalalignment='left',
    )
 
    ax.legend(fontsize=9)
 
axes[0].set_ylabel("β(t)", fontsize=13)
fig.supxlabel("Time (days)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_beta_est_fixed_params_panel_100_10.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_beta_est_fixed_params_panel_100_10.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel 2 saved: sigma/gamma fixed beta estimation (100/10).")

scenarios = [
    ### Easy
    {"label": "beta_spline_slow_rise",       "csv": "beta_spline_slow_rise.csv"},
    {"label": "beta_spline_gradual_decline",  "csv": "beta_spline_gradual_decline.csv"},
    ### Medium
    {"label": "beta_spline_three_waves",        "csv": "beta_spline_two_waves.csv"},
    ### Hard
    {"label": "beta_spline_escalating",       "csv": "beta_spline_escalating.csv"},
    {"label": "beta_spline_rapid",            "csv": "beta_spline_rapid.csv"},
]
 
### 2 rows x 3 cols, last cell hidden
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
 
### Flatten and hide the last unused axis
axes_flat = axes.flatten()
axes_flat[-1].set_visible(False)
 
for i, (ax, scenario) in enumerate(zip(axes_flat[:5], scenarios)):
    label = scenario["label"]
 
    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
    t_unnorm = data["time"].values * days_total
    I_true = data["I"].values * N_val
 
    ### Load PINN predictions
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_predictions_{label}_80_20.csv"))
    I_pred = pred["I_pred"].values * N_val
 
    ### Train/test split
    split = int(0.8 * len(t_unnorm))
    t_train_unnorm = t_unnorm[:split]
 
    ### Ground truth — coloured scatter
    ax.scatter(t_unnorm, I_true, color=COLOURS["I"],
               s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")
 
    ### PINN prediction — solid dark grey
    ax.plot(t_unnorm, I_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$I(t)$ PINN")
 
    ### Train/test split line
    ax.axvline(x=t_train_unnorm[-1], color='gray', linestyle='--',
               linewidth=1.5, label='Train/Test Split')
 
    ### Title with difficulty tag
    readable = label.replace("beta_spline_", "").replace("_", " ").title()
    ax.set_title(f"{readable}")
 
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
plt.savefig(os.path.join(output_dir, "PINN_time_varying_beta_panel_80_20.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_time_varying_beta_panel_80_20.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel saved: time-varying beta PINN forecasting (80/20).")
 

scenarios = [
    ### Easy
    {"label": "beta_spline_slow_rise",       "csv": "beta_spline_slow_rise.csv"},
    {"label": "beta_spline_gradual_decline",  "csv": "beta_spline_gradual_decline.csv"},
    ### Medium
    {"label": "beta_spline_three_waves",      "csv": "beta_spline_two_waves.csv"},
    ### Hard
    {"label": "beta_spline_escalating",       "csv": "beta_spline_escalating.csv"},
    {"label": "beta_spline_rapid",            "csv": "beta_spline_rapid.csv"},
]
 
### 2 rows x 3 cols, last cell hidden
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
 
### Flatten and hide the last unused axis
axes_flat = axes.flatten()
axes_flat[-1].set_visible(False)
 
for i, (ax, scenario) in enumerate(zip(axes_flat[:5], scenarios)):
    label = scenario["label"]
 
    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
    t_unnorm = data["time"].values * days_total
    I_true = data["I"].values   * N_val
 
    ### Load all-learnable PINN predictions
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_all_learnable_predictions_{label}_80_20.csv"))
    I_pred = pred["I_pred"].values * N_val
 
    ### Train/test split
    split = int(0.8 * len(t_unnorm))
    t_train_unnorm = t_unnorm[:split]
 
    ### Ground truth — coloured scatter
    ax.scatter(t_unnorm, I_true, color=COLOURS["I"],
               s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")
 
    ### PINN prediction — solid dark grey
    ax.plot(t_unnorm, I_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$I(t)$ PINN")
 
    ### Train/test split line
    ax.axvline(x=t_train_unnorm[-1], color='gray', linestyle='--',
               linewidth=1.5, label='Train/Test Split')
 
    ### Title
    readable = label.replace("beta_spline_", "").replace("_", " ").title()
    ax.set_title(f"{readable}")
 
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
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_time_varying_beta_panel_80_20.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_time_varying_beta_panel_80_20.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel saved: all-learnable PINN time-varying beta forecasting (80/20).")

scenarios = [
    {"label": "beta_spline_slow_rise", "csv": "beta_spline_slow_rise.csv"},
    {"label": "beta_spline_gradual_decline",  "csv": "beta_spline_gradual_decline.csv"},
    {"label": "beta_spline_three_waves", "csv": "beta_spline_two_waves.csv"},
    {"label": "beta_spline_escalating", "csv": "beta_spline_escalating.csv"},
    {"label": "beta_spline_rapid", "csv": "beta_spline_rapid.csv"},
]
 
### Beta reconstruction - all parameters learnt (100/0 split)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()
axes_flat[-1].set_visible(False)
 
for i, (ax, scenario) in enumerate(zip(axes_flat[:5], scenarios)):
    label = scenario["label"]
 
    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
    t_unnorm = data["time"].values * days_total
    beta_true = data["beta"].values
 
    ### Load all-learnable predictions CSV
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_all_learnable_predictions_{label}_100_0.csv"))
    beta_pred = pred["beta_pred"].values
 
    ### Estimated beta — solid dark grey line
    ax.plot(t_unnorm, beta_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$\hat{\beta}(t)$ estimated")
 
    ### True beta — coloured scatter
    ax.scatter(t_unnorm, beta_true, color="#1f33b4",
               s=15, alpha=0.7, zorder=3, label=r"$\beta(t)$ true")
 
    readable = label.replace("beta_spline_", "").replace("_", " ").title()
    ax.set_title(readable)
    ax.set_ylim(0, max(beta_true.max() * 1.3, 0.5))
 
    ax.text(
        -0.08, 1.02, f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13, verticalalignment='bottom', horizontalalignment='left',
    )
    ax.legend(fontsize=9)
 
fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel(r"$\beta(t)$", fontsize=13, x=0.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_beta_recon_time_varying_100_0.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_beta_recon_time_varying_100_0.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel 1 saved: all-learnable beta reconstruction (100/0).")
 
### Beta reconstruction - sigma & gamma fixed (100/0 split)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()
axes_flat[-1].set_visible(False)
 
for i, (ax, scenario) in enumerate(zip(axes_flat[:5], scenarios)):
    label = scenario["label"]
 
    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, scenario["csv"]))
    t_unnorm = data["time"].values * days_total
    beta_true = data["beta"].values
 
    ### Load fixed-params predictions CSV
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_predictions_{label}_100_10.csv"))
    beta_pred = pred["beta_pred"].values
 
    ### Estimated beta — solid dark grey line
    ax.plot(t_unnorm, beta_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$\hat{\beta}(t)$ estimated")
 
    ### True beta — coloured scatter
    ax.scatter(t_unnorm, beta_true, color="#1f33b4",
               s=15, alpha=0.7, zorder=3, label=r"$\beta(t)$ true")
 
    readable = label.replace("beta_spline_", "").replace("_", " ").title()
    ax.set_title(readable)
    ax.set_ylim(0, max(beta_true.max() * 1.3, 0.5))
 
    ax.text(
        -0.08, 1.02, f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13, verticalalignment='bottom', horizontalalignment='left',
    )
    ax.legend(fontsize=9)
 
fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel(r"$\beta(t)$", fontsize=13, x=0.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_fixed_params_beta_recon_time_varying_100_10.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_fixed_params_beta_recon_time_varying_100_10.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel 2 saved: fixed-params beta reconstruction (100/10).")

noise_levels = [5, 10, 15, 20]

### Fixed parameters, gaussian noise forecasting 80/20 split
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
 
for i, (ax, noise_percent) in enumerate(zip(axes.flat, noise_levels)):
    label = f"Gaussian_noise_{noise_percent}percent"
 
    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, f"SEIR_Gaussian_noise_{noise_percent}percent.csv"))
    t_unnorm = data["time"].values * days_total
    I_true = data["I"].values   * N_val
 
    ### Load fixed-params predictions
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_predictions_{label}_80_20.csv"))
    I_pred = pred["I_pred"].values * N_val
 
    ### Train/test split
    split = int(0.8 * len(t_unnorm))
    t_train_unnorm = t_unnorm[:split]
 
    ### Ground truth — coloured scatter
    ax.scatter(t_unnorm, I_true, color=COLOURS["I"],
               s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")
 
    ### PINN prediction — solid dark grey
    ax.plot(t_unnorm, I_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$I(t)$ PINN")
 
    ### Train/test split line
    ax.axvline(x=t_train_unnorm[-1], color='gray', linestyle='--',
               linewidth=1.5, label='Train/Test Split')
 
    ax.set_title(f"{noise_percent}% Gaussian Noise")
 
    ### R0 annotation
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )
 
    ### Panel label
    ax.text(
        -0.08, 1.02, f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13, verticalalignment='bottom', horizontalalignment='left',
    )
 
    ax.legend(fontsize=9)
 
fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel("Number of infected individuals", fontsize=13, x=0.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_fixed_params_Gaussian_noise_panel_80_20.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_fixed_params_Gaussian_noise_panel_80_20.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel 1 saved: fixed params Gaussian noise forecasting (80/20).")
 
### All parameters learnt, gaussian noise 80/20 split
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
 
for i, (ax, noise_percent) in enumerate(zip(axes.flat, noise_levels)):
    label = f"Gaussian_noise_{noise_percent}percent"
 
    ### Load ground truth
    data = pd.read_csv(os.path.join(data_folder, f"SEIR_Gaussian_noise_{noise_percent}percent.csv"))
    t_unnorm = data["time"].values * days_total
    I_true = data["I"].values   * N_val
 
    ### Load all-learnable predictions
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_all_learnable_predictions_{label}_80_20.csv"))
    I_pred = pred["I_pred"].values * N_val
 
    ### Train/test split
    split = int(0.8 * len(t_unnorm))
    t_train_unnorm = t_unnorm[:split]
 
    ### Ground truth — coloured scatter
    ax.scatter(t_unnorm, I_true, color=COLOURS["I"],
               s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")
 
    ### PINN prediction — solid dark grey
    ax.plot(t_unnorm, I_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$I(t)$ PINN")
 
    ### Train/test split line
    ax.axvline(x=t_train_unnorm[-1], color='gray', linestyle='--',
               linewidth=1.5, label='Train/Test Split')
 
    ax.set_title(f"{noise_percent}% Gaussian Noise")
 
    ### R0 annotation
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )
 
    ### Panel label
    ax.text(
        -0.08, 1.02, f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13, verticalalignment='bottom', horizontalalignment='left',
    )
 
    ax.legend(fontsize=9)
 
fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel("Number of infected individuals", fontsize=13, x=0.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_Gaussian_noise_panel_80_20.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_Gaussian_noise_panel_80_20.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel 2 saved: all-learnable Gaussian noise forecasting (80/20).")
 
BETA_TRUE  = 0.75
GAMMA = 0.25
R0_val = BETA_TRUE / GAMMA
noise_levels = [5, 10, 15, 20]
 
### Gaussian noise beta reconstruction - sigma and gamma provided to PINN (100/10 split)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
 
for i, (ax, noise_percent) in enumerate(zip(axes.flat, noise_levels)):
    label = f"Gaussian_noise_{noise_percent}percent"
 
    ### Load ground truth data for time axis
    data = pd.read_csv(os.path.join(data_folder, f"SEIR_Gaussian_noise_{noise_percent}percent.csv"))
    t_unnorm = data["time"].values * days_total
 
    ### Load fixed-params predictions
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_predictions_{label}_100_10.csv"))
    beta_pred = pred["beta_pred"].values
 
    ### Estimated beta — solid dark grey line
    ax.plot(t_unnorm, beta_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$\hat{\beta}(t)$ estimated")
 
    ### True beta — constant scatter
    ax.scatter(t_unnorm, np.full_like(t_unnorm, BETA_TRUE),
               color="#1f33b4", s=15, alpha=0.5, zorder=3,
               label=rf"$\beta$ true = {BETA_TRUE}")
 
    ax.set_title(f"{noise_percent}% Gaussian Noise")
    ax.set_ylim(0, 3)
 
    ### R0 annotation
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )
 
    ### Panel label
    ax.text(
        -0.08, 1.02, f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13, verticalalignment='bottom', horizontalalignment='left',
    )
 
    ax.legend(fontsize=9)
 
fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel(r"$\beta(t)$", fontsize=13, x=0.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_fixed_params_beta_recon_Gaussian_noise_100_10.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_fixed_params_beta_recon_Gaussian_noise_100_10.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel 1 saved: fixed params beta reconstruction Gaussian noise (100/10).")
 
### Gaussian noise - beta reconstruction - all parameters learnt (100/0 split)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
 
for i, (ax, noise_percent) in enumerate(zip(axes.flat, noise_levels)):
    label = f"Gaussian_noise_{noise_percent}percent"
 
    ### Load ground truth data for time axis
    data = pd.read_csv(os.path.join(data_folder, f"SEIR_Gaussian_noise_{noise_percent}percent.csv"))
    t_unnorm = data["time"].values * days_total
 
    ### Load all-learnable predictions
    pred = pd.read_csv(os.path.join(output_dir, f"PINN_all_learnable_predictions_{label}_100_0.csv"))
    beta_pred = pred["beta_pred"].values
 
    ### Estimated beta — solid dark grey line
    ax.plot(t_unnorm, beta_pred, color="#444444",
            linewidth=2, linestyle='-', label=r"$\hat{\beta}(t)$ estimated")
 
    ### True beta — constant scatter
    ax.scatter(t_unnorm, np.full_like(t_unnorm, BETA_TRUE),
               color="#1f33b4", s=15, alpha=0.5, zorder=3,
               label=rf"$\beta$ true = {BETA_TRUE}")
 
    ax.set_title(f"{noise_percent}% Gaussian Noise")
    ax.set_ylim(0, 3)
 
    ### R0 annotation
    ax.text(
        0.98, 0.97,
        rf"$\mathcal{{R}}_0 = {R0_val:.2f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
    )
 
    ### Panel label
    ax.text(
        -0.08, 1.02, f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13, verticalalignment='bottom', horizontalalignment='left',
    )
 
    ax.legend(fontsize=9)
 
fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel(r"$\beta(t)$", fontsize=13, x=0.01)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_beta_recon_Gaussian_noise_100_0.pdf"), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(output_dir, "PINN_all_learnable_beta_recon_Gaussian_noise_100_0.png"), bbox_inches='tight', dpi=300)
plt.show()
print("Panel 2 saved: all-learnable beta reconstruction Gaussian noise (100/0).")
 