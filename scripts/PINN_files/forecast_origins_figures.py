import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import forecast_origins as fo

### Figures for the multi-origin forecasting runs (forecast_origins.py must have been run first).
### Same style as panel_figures.py; PINN curves are the median over 5 seeds.

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

COLOURS = {"I": "#d62728"}
PINN_COLOUR = "#444444"
### One colour per scenario within a panel: the first five slots of a CVD-validated categorical palette,
### always in this order, with a distinct marker as well so identity never rests on colour alone
SCENARIO_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SCENARIO_MARKERS = ["o", "s", "^", "D", "v"]

N_val = 100001
days_total = 100
ORIGINS = [60, 70, 80, 90]
GROUP_TITLES = {"constant": "Constant β", "noise": "Gaussian noise (β = 0.75)",
                "time_varying": "Time-varying β", "forecastable": "Time-varying β, forecastable"}

fig_dir = "../../report_figures/forecast_origins"
os.makedirs(fig_dir, exist_ok=True)


def readable(label):
    if label.startswith("noise_"):
        return f"{label.split('_')[1]}% noise"
    if label.startswith("beta_spline_"):
        label = label.replace("beta_spline_", "")
    elif label.startswith("beta_"):
        return rf"$\beta = {label.split('_')[1]}$"
    return label.replace("_", " ").capitalize()


def panel_label(ax, i):
    ax.text(-0.08, 1.02, f"({chr(97 + i)})", transform=ax.transAxes,
            fontsize=13, verticalalignment='bottom', horizontalalignment='left')


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved {name}")


def scenario_lines(ax, df, x, y):
    """One line per scenario (median over seeds, and over origins where x is horizon)."""
    for k, (label, g) in enumerate(df.groupby("scenario", sort=False)):
        m = g.groupby(x)[y].median()
        ax.plot(m.index, m.values, color=SCENARIO_COLOURS[k], marker=SCENARIO_MARKERS[k],
                markersize=7, linewidth=2, label=readable(label))
    ax.legend(fontsize=9, loc='upper left')


per_seed = pd.read_csv(os.path.join(fo.OUT, "forecast_by_origin_per_seed.csv"))
horizon = pd.read_csv(os.path.join(fo.OUT, "forecast_by_horizon_per_seed.csv"))

def summary_figure(df, sg, x, y, xlabel, ylabel, name):
    """2x2 panels (one per scenario group), one line per scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for i, (ax, group) in enumerate(zip(axes.flat, GROUP_TITLES)):
        scenario_lines(ax, df[(df.group == group) & (df.sigma_gamma == sg)], x, y)
        ax.set_title(GROUP_TITLES[group])
        ax.set_xticks(range(1, 11) if x == "horizon" else ORIGINS)
        ax.set_ylim(0, 1.35 * ax.get_ylim()[1])           ### headroom so the legend clears the lines
        panel_label(ax, i)
    fig.supxlabel(xlabel, fontsize=13)
    fig.supylabel(ylabel, fontsize=13, x=0.01)
    save(fig, f"{name}_{sg}")


for sg in ("provided", "learned"):
    ### I(t): error against days ahead (median over seeds and the four origins) and 10-day MAE against origin
    summary_figure(horizon, sg, "horizon", "AE_pct", "Days ahead of the forecast origin",
                   "Absolute error in I(t) (%)", "forecast_error_by_horizon")
    summary_figure(per_seed, sg, "origin", "MAE10_pct", "Forecast origin (day)",
                   "MAE over the first 10 forecast days (%)", "forecast_mae10_by_origin")
    ### beta(t): error of the extrapolated beta against days ahead, and reconstruction error over the
    ### training window (days 0..origin) against origin
    summary_figure(horizon, sg, "horizon", "AE_beta_pct", "Days ahead of the forecast origin",
                   r"Absolute error in $\beta(t)$ (%)", "beta_error_by_horizon")
    summary_figure(per_seed, sg, "origin", "beta_MAPE_train_pct", "Forecast origin (day)",
                   r"$\beta(t)$ MAPE over the training window (%)", "beta_mape_train_by_origin")

### forecasts from every origin: one row per scenario, one column per origin
preds = {}
for group, label, task, beta_fit, learn, origin, seed, pred in fo.runs():
    preds.setdefault((group, label, learn, origin), []).append(pred)

for group, (scenarios, _, _) in fo.GROUPS.items():
    for learn in (False, True):
        fig, axes = plt.subplots(len(scenarios), len(ORIGINS), figsize=(18, 3.6 * len(scenarios)),
                                 sharex=True, squeeze=False)
        for r, label in enumerate(scenarios):
            task = fo.sc.load_task(label, fo.data_folder)
            t = task["t"] * days_total
            ymax = 1.3 * task["I"].max() * N_val
            for c, origin in enumerate(ORIGINS):
                ax = axes[r, c]
                ax.scatter(t, task["I"] * N_val, color=COLOURS["I"], s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")
                if "I_clean" in task:
                    ax.plot(t, task["I_clean"] * N_val, color=COLOURS["I"], linewidth=1, linestyle=':',
                            label=r"$I(t)$ noise-free")
                Y = np.stack([p["I_pred"].values for p in preds[(group, label, learn, origin)]]) * N_val
                ax.plot(t, np.median(Y, 0), color=PINN_COLOUR, linewidth=2,
                        label=rf"$I(t)$ PINN (median of {len(Y)} seeds)")
                ax.axvline(x=origin, color='gray', linestyle='--', linewidth=1.5, label='Forecast origin')
                ax.set_ylim(-0.02 * ymax, ymax)
                if r == 0:
                    ax.set_title(f"Origin: day {origin}")
                if c == 0:
                    ax.set_ylabel(readable(label), fontsize=13)
                panel_label(ax, r * len(ORIGINS) + c)
        axes[0, 0].legend(fontsize=9, loc='upper left')
        fig.supxlabel("Time (days)", fontsize=13)
        fig.supylabel("Number of infected individuals", fontsize=13, x=0.0)
        save(fig, f"forecast_origins_{group}_{'learned' if learn else 'provided'}")

### beta(t) from every origin: one row per scenario, one column per origin
for group, (scenarios, _, _) in fo.GROUPS.items():
    for learn in (False, True):
        fig, axes = plt.subplots(len(scenarios), len(ORIGINS), figsize=(18, 3.6 * len(scenarios)),
                                 sharex=True, squeeze=False)
        for r, label in enumerate(scenarios):
            task = fo.sc.load_task(label, fo.data_folder)
            t = task["t"] * days_total
            beta_true = task["beta"]
            constant = np.allclose(beta_true, beta_true[0])
            ymax = max(2 * beta_true.max(), 1.0)          ### extrapolated beta beyond this is clipped
            for c, origin in enumerate(ORIGINS):
                ax = axes[r, c]
                ax.scatter(t, beta_true, color="#1f33b4", s=15, alpha=0.5, zorder=3,
                           label=rf"$\beta$ true = {beta_true[0]:g}" if constant else r"$\beta(t)$ true")
                B = np.stack([p["beta_pred"].values for p in preds[(group, label, learn, origin)]])
                ax.plot(t, np.median(B, 0), color=PINN_COLOUR, linewidth=2,
                        label=rf"$\hat{{\beta}}(t)$ PINN (median of {len(B)} seeds)")
                ax.axvline(x=origin, color='gray', linestyle='--', linewidth=1.5, label='Forecast origin')
                ax.set_ylim(0, ymax)
                if r == 0:
                    ax.set_title(f"Origin: day {origin}")
                if c == 0:
                    ax.set_ylabel(readable(label), fontsize=13)
                panel_label(ax, r * len(ORIGINS) + c)
        axes[0, 0].legend(fontsize=9, loc='upper left')
        fig.supxlabel("Time (days)", fontsize=13)
        fig.supylabel(r"$\beta(t)$", fontsize=13, x=0.0)
        save(fig, f"beta_origins_{group}_{'learned' if learn else 'provided'}")
