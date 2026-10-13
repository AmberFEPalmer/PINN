import glob
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import synthetic_scenarios as sc

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
    "beta": "#008094",
}
PINN_COLOUR = "#444444"

data_folder = os.path.join("..", "..", "data")
output_dir = "../../png_files"
report_dir = "../../report_figures"
os.makedirs(output_dir, exist_ok=True)

### Where each set of fits lives (5 seeds each). Constant-beta scenarios: beta fitted as one learnable
### constant. Time-varying scenarios: beta(t) network.
CONST_DIR = "../../results/PINN_simulated_data/const_beta_seeds"
NOISE_DIR = "../../results/PINN_simulated_data/const_beta_noise_seeds"
TV_DIR = "../../results/PINN_simulated_data/origins/time_varying"

N_val = 100001
days_total = 100
GAMMA = 0.25
SPLIT = 0.8

CONSTANT = [("beta_0.75", 0.75), ("beta_0.5", 0.5), ("beta_0.4", 0.4)]
NOISE = [5, 10, 15, 20]
TIME_VARYING = ["beta_spline_slow_rise", "beta_spline_gradual_decline", "beta_spline_three_waves",
                "beta_spline_escalating", "beta_spline_rapid"]


### loading
def load_runs(run_dir, pred_prefix, tag, suffix):
    """beta(t) and I(t) predictions for every seed of one scenario, each stacked to (n_seeds, n_days).
    Falls back to the single-seed file name used before the multi-seed runs."""
    files = sorted(glob.glob(os.path.join(run_dir, f"{pred_prefix}_predictions_{tag}_seed*_{suffix}.csv")))
    if not files:
        files = [os.path.join(run_dir, f"{pred_prefix}_predictions_{tag}_{suffix}.csv")]
    preds = [pd.read_csv(f) for f in files]
    return {k: np.stack([p[k].values for p in preds]) for k in ("I_pred", "beta_pred")}


def load_metrics(run_dir, prefix, suffix):
    """Per-seed error metrics, with the scenario name (seed tag removed) and the seed in its own column."""
    m = pd.read_csv(os.path.join(run_dir, f"{prefix}_error_metrics_{suffix}.csv"))
    m["seed"] = m.scenario.str.extract(r"_seed(\d+)$")[0].fillna(0).astype(int)
    m["scenario"] = m.scenario.str.replace(r"_seed\d+$", "", regex=True)
    return m


def noise_tag(p):
    return f"Gaussian_noise_{p}percent"


### panel pieces
def panel_label(ax, i):
    ax.text(-0.08, 1.02, f"({chr(97 + i)})", transform=ax.transAxes,
            fontsize=13, verticalalignment='bottom', horizontalalignment='left')


def R0_note(ax, R0):
    """R0 annotation, top right."""
    if R0 is not None:
        ax.text(0.98, 0.97, rf"$\mathcal{{R}}_0 = {R0:.2f}$", transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))


def plot_median(ax, t, Y, label):
    """Median over seeds as a line (single seed: that seed). The spread over seeds is in the tables."""
    n = len(Y)
    ax.plot(t, np.median(Y, 0), color=PINN_COLOUR, linewidth=2, linestyle='-',
            label=f"{label} PINN" + (f" (median of {n} seeds)" if n > 1 else ""))


def forecast_panel(ax, task, runs, i, title, R0=None):
    t = task["t"] * days_total
    ax.scatter(t, task["I"] * N_val, color=COLOURS["I"], s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")
    if "I_clean" in task:
        ax.plot(t, task["I_clean"] * N_val, color=COLOURS["I"], linewidth=1, linestyle=':',
                label=r"$I(t)$ noise-free")
    plot_median(ax, t, runs["I_pred"] * N_val, r"$I(t)$")
    n_train = int(round(SPLIT * len(t)))                 ### same rule as the fits: days 0 .. n_train-1
    ax.axvline(x=t[n_train - 1], color='gray', linestyle='--', linewidth=1.5, label='Train/Test Split')
    ax.set_title(title)
    R0_note(ax, R0)
    panel_label(ax, i)
    ax.legend(fontsize=9, loc='upper left')


def beta_panel(ax, task, runs, i, title, R0=None, ylim=None):
    t = task["t"] * days_total
    beta_true = task["beta"]
    constant = np.allclose(beta_true, beta_true[0])
    ax.scatter(t, beta_true, color="#1f33b4", s=30 if constant else 15, alpha=0.3 if constant else 0.7,
               zorder=3, label=rf"$\beta$ true = {beta_true[0]:g}" if constant else r"$\beta(t)$ true")
    plot_median(ax, t, runs["beta_pred"], r"$\hat{\beta}(t)$")
    ax.set_title(title)
    ax.set_ylim(*(ylim or (0, max(beta_true.max() * 1.3, 0.5))))
    R0_note(ax, R0)
    panel_label(ax, i)
    ax.legend(fontsize=9, loc='upper left')


def clip_forecast_axes(axes, tasks):
    """Same y-range across panels: up to 1.3 x the largest observation."""
    ymax = 1.3 * max(task["I"].max() for task in tasks) * N_val
    for ax in axes:
        ax.set_ylim(-0.02 * ymax, ymax)


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved {name}")


def readable(label):
    return label.replace("beta_spline_", "").replace("_", " ").title()


### constant beta: forecasting (80/20)
for run_prefix, name in [("PINN_beta_const", "PINN_beta_panel"),
                         ("PINN_all_learnable_beta_const", "PINN_all_learnable_beta_panel_80_20")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    tasks = [sc.load_task(tag, data_folder) for tag, _ in CONSTANT]
    for i, (ax, (tag, b), task) in enumerate(zip(axes, CONSTANT, tasks)):
        forecast_panel(ax, task, load_runs(CONST_DIR, run_prefix, tag, "80_20"), i, rf"$\beta = {b}$",
                       R0=b / GAMMA)
    clip_forecast_axes(axes, tasks)
    fig.supxlabel("Time (days)", fontsize=13)
    fig.supylabel("Number of infected individuals", fontsize=13, x=0.01)
    save(fig, name)

### constant beta: beta reconstruction (all 100 days)
for run_prefix, suffix, name in [("PINN_beta_const", "100_10", "PINN_beta_est_fixed_params_panel_100_10"),
                                 ("PINN_all_learnable_beta_const", "100_0", "PINN_beta_est_all_learnable_panel_100_0")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, (ax, (tag, b)) in enumerate(zip(axes, CONSTANT)):
        beta_panel(ax, sc.load_task(tag, data_folder), load_runs(CONST_DIR, run_prefix, tag, suffix), i,
                   rf"$\beta = {b}$", R0=b / GAMMA, ylim=(0, 3))
    axes[0].set_ylabel("β(t)", fontsize=13)
    fig.supxlabel("Time (days)", fontsize=13)
    save(fig, name)

### time-varying beta: forecasting (80/20)
for run_prefix, name in [("PINN", "PINN_time_varying_beta_panel_80_20"),
                         ("PINN_all_learnable", "PINN_all_learnable_time_varying_beta_panel_80_20")]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    axes_flat[-1].set_visible(False)
    for i, (ax, tag) in enumerate(zip(axes_flat[:5], TIME_VARYING)):
        forecast_panel(ax, sc.load_task(tag, data_folder), load_runs(TV_DIR, run_prefix, tag, "80_20"), i,
                       readable(tag))
    fig.supxlabel("Time (days)", fontsize=13)
    fig.supylabel("Number of infected individuals", fontsize=13, x=0.01)
    save(fig, name)

### time-varying beta: beta reconstruction (all 100 days)
for run_prefix, suffix, name in [("PINN_all_learnable", "100_0", "PINN_all_learnable_beta_recon_time_varying_100_0"),
                                 ("PINN", "100_10", "PINN_fixed_params_beta_recon_time_varying_100_10")]:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    axes_flat[-1].set_visible(False)
    for i, (ax, tag) in enumerate(zip(axes_flat[:5], TIME_VARYING)):
        beta_panel(ax, sc.load_task(tag, data_folder), load_runs(TV_DIR, run_prefix, tag, suffix), i,
                   readable(tag))
    fig.supxlabel("Time (days)", fontsize=13)
    fig.supylabel(r"$\beta(t)$", fontsize=13, x=0.01)
    save(fig, name)

### Gaussian noise (constant beta = 0.75): forecasting (80/20)
R0_noise = 0.75 / GAMMA
for run_prefix, name in [("PINN_beta_const", "PINN_fixed_params_Gaussian_noise_panel_80_20"),
                         ("PINN_all_learnable_beta_const", "PINN_all_learnable_Gaussian_noise_panel_80_20")]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    tasks = [sc.load_task(f"noise_{p}") for p in NOISE]
    for i, (ax, p, task) in enumerate(zip(axes.flat, NOISE, tasks)):
        forecast_panel(ax, task, load_runs(NOISE_DIR, run_prefix, noise_tag(p), "80_20"), i,
                       f"{p}% Gaussian Noise", R0=R0_noise)
    clip_forecast_axes(axes.flat, tasks)
    fig.supxlabel("Time (days)", fontsize=13)
    fig.supylabel("Number of infected individuals", fontsize=13, x=0.01)
    save(fig, name)

### Gaussian noise (constant beta = 0.75): beta reconstruction (all 100 days)
for run_prefix, suffix, name in [
        ("PINN_beta_const", "100_10", "PINN_fixed_params_beta_recon_Gaussian_noise_100_10"),
        ("PINN_all_learnable_beta_const", "100_0", "PINN_all_learnable_beta_recon_Gaussian_noise_100_0")]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for i, (ax, p) in enumerate(zip(axes.flat, NOISE)):
        beta_panel(ax, sc.load_task(f"noise_{p}"), load_runs(NOISE_DIR, run_prefix, noise_tag(p), suffix), i,
                   f"{p}% Gaussian Noise", R0=R0_noise, ylim=(0, 3))
    fig.supxlabel("Time (days)", fontsize=13)
    fig.supylabel(r"$\beta(t)$", fontsize=13, x=0.01)
    save(fig, name)


### summary tables: median [min, max] over seeds, plus every seed in a long table
def summary_table(run_dir, run_prefix, metrics_prefix, suffix, scenarios):
    m = load_metrics(run_dir, metrics_prefix, suffix)
    m = m[m.scenario.isin([tag for tag, _ in scenarios])].copy()
    if suffix == "80_20":
        ### noise scenarios: forecast error against the noise-free curve as well as the noisy observations
        clean = []
        for _, row in m.iterrows():
            task = dict(scenarios)[row.scenario]
            if "I_clean" not in task:
                clean.append(np.nan); continue
            f = os.path.join(run_dir, f"{run_prefix}_predictions_{row.scenario}_seed{row.seed}_{suffix}.csv")
            I_pred = pd.read_csv(f)["I_pred"].values
            n_train = int(round(SPLIT * len(task["t"])))
            c = task["I_clean"][n_train:]
            clean.append(100 * np.mean(np.abs(I_pred[n_train:] - c)) / np.mean(c))
        m["I_forecast_MAE_vs_clean_pct"] = clean
    cols = [c for c in ("beta_MAPE_pct", "sigma", "gamma", "I_forecast_MAE_pct", "I_forecast_MAE_vs_clean_pct")
            if c in m and m[c].notna().any()]
    rows = []
    for tag, _ in scenarios:
        g = m[m.scenario == tag]
        row = {"scenario": tag, "beta_fit": "constant", "n_seeds": len(g)}
        for c in cols:
            row.update({f"{c}_median": g[c].median(), f"{c}_min": g[c].min(), f"{c}_max": g[c].max()})
        rows.append(row)
    return pd.DataFrame(rows), m


table_scenarios = ([(tag, sc.load_task(tag, data_folder)) for tag, _ in CONSTANT] +
                   [(noise_tag(p), sc.load_task(f"noise_{p}")) for p in NOISE])
tables = {}
for run_prefix, metrics_prefix, suffix, name in [
        ("PINN_beta_const", "PINN_beta_const", "80_20", "table5_forecast_params_provided"),
        ("PINN_all_learnable_beta_const", "PINN_all_learnable_parameters_beta_const", "80_20",
         "table6_forecast_infection_data_only"),
        ("PINN_beta_const", "PINN_beta_const", "100_10", "beta_recon_params_provided"),
        ("PINN_all_learnable_beta_const", "PINN_all_learnable_parameters_beta_const", "100_0",
         "beta_recon_infection_data_only")]:
    parts = [summary_table(d, run_prefix, metrics_prefix, suffix, table_scenarios) for d in (CONST_DIR, NOISE_DIR)]
    summary = pd.concat([p[0][p[0].n_seeds > 0] for p in parts], ignore_index=True)
    per_seed = pd.concat([p[1] for p in parts], ignore_index=True)
    tables[name] = (summary, per_seed)
    summary.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    per_seed.to_csv(os.path.join(output_dir, f"{name}_per_seed.csv"), index=False)
    print(f"saved {name}")


### copy into report_figures/ under the report's figure numbers; 2, 7, 12, 6b and 23 come from other scripts
REPORT_FIGURES = {
    "PINN_beta_panel": "fig03_forecast_constant_beta_params_provided",
    "PINN_all_learnable_beta_panel_80_20": "fig04_forecast_constant_beta_infection_data_only",
    "PINN_beta_est_fixed_params_panel_100_10": "fig05_beta_recon_constant_beta_params_provided",
    "PINN_beta_est_all_learnable_panel_100_0": "fig06_beta_recon_constant_beta_infection_data_only",
    "PINN_time_varying_beta_panel_80_20": "fig08_forecast_time_varying_beta_params_provided",
    "PINN_all_learnable_time_varying_beta_panel_80_20": "fig09_forecast_time_varying_beta_infection_data_only",
    "PINN_fixed_params_beta_recon_time_varying_100_10": "fig10_beta_recon_time_varying_beta_params_provided",
    "PINN_all_learnable_beta_recon_time_varying_100_0": "fig11_beta_recon_time_varying_beta_infection_data_only",
    "PINN_fixed_params_Gaussian_noise_panel_80_20": "fig13_forecast_noise_params_provided",
    "PINN_all_learnable_Gaussian_noise_panel_80_20": "fig14_forecast_noise_infection_data_only",
    "PINN_fixed_params_beta_recon_Gaussian_noise_100_10": "fig15_beta_recon_noise_params_provided",
    "PINN_all_learnable_beta_recon_Gaussian_noise_100_0": "fig16_beta_recon_noise_infection_data_only",
}
os.makedirs(os.path.join(report_dir, "tables"), exist_ok=True)
for src, dst in REPORT_FIGURES.items():
    for ext in ("pdf", "png"):
        shutil.copy(os.path.join(output_dir, f"{src}.{ext}"), os.path.join(report_dir, f"{dst}.{ext}"))
for name in tables:
    for f in (f"{name}.csv", f"{name}_per_seed.csv"):
        shutil.copy(os.path.join(output_dir, f), os.path.join(report_dir, "tables", f))
print(f"copied {len(REPORT_FIGURES)} figures and {len(tables)} tables to {report_dir}")
