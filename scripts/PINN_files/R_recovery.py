import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import synthetic_scenarios as sc

### Is R(t) = beta(t)/gamma recovered better than beta(t) when sigma and gamma are learned from I(t) only?
### A low beta paired with a low gamma gives the right R, so if the two trade off, the R error is smaller than
### the beta error. Uses the existing full-window fits (all 100 days, 5 seeds) behind the report's beta
### reconstruction figures: learned setup (suffix 100_0) and, for reference, the provided setup (100_10, gamma
### fixed at the true 0.25, so its R error equals its beta error).
### Errors are MAPE over all 101 days against the true beta/0.25.

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

RES = "../../results/PINN_simulated_data"
data_folder = "../../data"
fig_dir = "../../report_figures/R_recovery"
os.makedirs(os.path.join(fig_dir, "tables"), exist_ok=True)
GAMMA_TRUE = 0.25
days_total = 100
TRUE_COLOUR = "#7397de"
R_COLOUR = "#1f3d99"
BETA_COLOUR = "#e07b39"
GROUP_COLOURS = {"Constant β": "#1f3d99", "Time-varying β": "#e07b39", "Gaussian noise": "#2ca02c"}
GROUP_MARKERS = {"Constant β": "o", "Time-varying β": "s", "Gaussian noise": "^"}

TIME_VARYING = ["beta_spline_slow_rise", "beta_spline_gradual_decline", "beta_spline_three_waves",
                "beta_spline_escalating", "beta_spline_rapid"]
### group -> (run dir, learned pred prefix, learned metrics prefix, scenarios as (label, file tag))
GROUPS = {
    "Constant β": (f"{RES}/const_beta_seeds", "PINN_all_learnable_beta_const",
                   "PINN_all_learnable_parameters_beta_const", [(s, s) for s in sc.CONSTANT]),
    "Time-varying β": (f"{RES}/origins/time_varying", "PINN_all_learnable", "PINN_all_learnable_parameters",
                       [(s, s) for s in TIME_VARYING]),
    "Gaussian noise": (f"{RES}/const_beta_noise_seeds", "PINN_all_learnable_beta_const",
                       "PINN_all_learnable_parameters_beta_const",
                       [(s, f"Gaussian_noise_{s.split('_')[1]}percent") for s in sc.NOISE]),
}


def readable(label):
    if label.startswith("noise_"):
        return f"{label.split('_')[1]}% Gaussian noise"
    if label.startswith("beta_spline_"):
        return label.replace("beta_spline_", "").replace("_", " ").title()
    return rf"$\beta = {label.split('_')[1]}$"


def panel_label(ax, i):
    ax.text(-0.08, 1.02, f"({chr(97 + i)})", transform=ax.transAxes,
            fontsize=13, verticalalignment='bottom', horizontalalignment='left')


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved {name}")


def mape(est, true):
    return 100 * np.mean(np.abs(est - true) / true)


### every seed: beta(t), learned gamma, R(t) and both errors
rows, curves = [], {}
for group, (run_dir, pred_prefix, metrics_prefix, scenarios) in GROUPS.items():
    m = pd.read_csv(os.path.join(run_dir, f"{metrics_prefix}_error_metrics_100_0.csv"))
    gammas = {r.scenario: (r.gamma, r.sigma) for r in m.itertuples()}
    for label, tag in scenarios:
        task = sc.load_task(label, data_folder)
        files = sorted(glob.glob(os.path.join(run_dir, f"{pred_prefix}_predictions_{tag}_seed*_100_0.csv")))
        for f in files:
            seed = int(re.search(r"_seed(\d+)_", f).group(1))
            gamma, sigma = gammas[f"{tag}_seed{seed}"]
            beta = pd.read_csv(f)["beta_pred"].values
            R = beta / gamma
            curves.setdefault(label, []).append((beta, R))
            rows.append({"group": group, "scenario": label, "seed": seed, "gamma_hat": gamma, "sigma_hat": sigma,
                         "beta_MAPE_pct": mape(beta, task["beta"]),
                         "R_MAPE_pct": mape(R, task["beta"] / GAMMA_TRUE),
                         "gamma_err_pct": 100 * (gamma - GAMMA_TRUE) / GAMMA_TRUE})
per_seed = pd.DataFrame(rows)
per_seed.to_csv(os.path.join(fig_dir, "tables", "R_vs_beta_recovery_per_seed.csv"), index=False)
summary = per_seed.groupby(["group", "scenario"], sort=False).agg(
    beta_MAPE_median=("beta_MAPE_pct", "median"), R_MAPE_median=("R_MAPE_pct", "median"),
    gamma_hat_median=("gamma_hat", "median"),
    R_better_seeds=("R_MAPE_pct", lambda s: int((s < per_seed.loc[s.index, "beta_MAPE_pct"]).sum())),
    n_seeds=("seed", "size")).round(2)
summary.to_csv(os.path.join(fig_dir, "tables", "R_vs_beta_recovery_summary.csv"))
print(summary.to_string())
print(f"\nR error < beta error in {(per_seed.R_MAPE_pct < per_seed.beta_MAPE_pct).sum()} of {len(per_seed)} fits")

### figure 1: R(t) for the time-varying scenarios - truth, R from beta/gamma_hat, and beta alone on the R scale
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()
axes_flat[-1].set_visible(False)
for i, (ax, label) in enumerate(zip(axes_flat, TIME_VARYING)):
    task = sc.load_task(label, data_folder)
    t = task["t"] * days_total
    beta = np.median(np.stack([b for b, _ in curves[label]]), 0)
    R = np.median(np.stack([r for _, r in curves[label]]), 0)
    ax.scatter(t, task["beta"] / GAMMA_TRUE, color=TRUE_COLOUR, s=12, label=r"True $R(t) = \beta(t)/\gamma$")
    ax.plot(t, R, color=R_COLOUR, label=r"$\hat\beta(t)/\hat\gamma$ (learned $\gamma$)")
    ax.plot(t, beta / GAMMA_TRUE, color=BETA_COLOUR, linestyle='--',
            label=r"$\hat\beta(t)/0.25$ ($\beta$ error alone)")
    ax.axhline(1.0, color='#444444', linestyle=':', linewidth=1)
    ax.set_title(readable(label))
    ax.set_ylim(0, 1.15 * task["beta"].max() / GAMMA_TRUE)
    panel_label(ax, i)
axes_flat[0].legend(fontsize=9, loc='lower center')
fig.supxlabel("Time (days)", fontsize=13)
fig.supylabel(r"$R(t)$ (median of 5 seeds)", fontsize=13, x=0.01)
save(fig, "R_recon_time_varying_infection_data_only")

### figure 2: every fit, beta error against R error; below the diagonal = R recovered better
fig, ax = plt.subplots(figsize=(7, 6.5))
for group, g in per_seed.groupby("group", sort=False):
    ax.scatter(g.beta_MAPE_pct, g.R_MAPE_pct, color=GROUP_COLOURS[group], marker=GROUP_MARKERS[group], s=60,
               edgecolor='white', linewidth=1, label=group, zorder=3)
lim = 1.08 * max(per_seed.beta_MAPE_pct.max(), per_seed.R_MAPE_pct.max())
ax.plot([0, lim], [0, lim], color='#444444', linestyle='--', linewidth=1, label="Equal error")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect('equal')
ax.set_xlabel(r"$\beta(t)$ error (MAPE, %)")
ax.set_ylabel(r"$R(t) = \beta/\gamma$ error (MAPE, %)")
ax.legend(fontsize=10, loc='upper left')
save(fig, "R_vs_beta_error_infection_data_only")
