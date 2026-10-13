import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import forecast_origins as fo
import conformal_rolling as cr

### Figures for the causal rolling-origin conformal intervals (conformal_rolling.py must have been run first).
### Same style as panel_figures.py; interval colours as in conformal_PINN.py. ACI is the primary variant,
### as for the UKHSA data. Point forecast = the single-seed fit at that origin.
### Usage: python conformal_rolling_figures.py [provided|learned]

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
PI_COLOUR = "#7397de"
PI_BANDS = [(0.95, 0.025, 0.975, 0.20), (0.9, 0.1, 0.9, 0.30), (0.5, 0.25, 0.75, 0.55)]
LEVEL_COLOURS = {50: "#86b6ef", 90: "#3987e5", 95: "#1c5cab"}
LEVEL_MARKERS = {50: "o", 90: "s", 95: "^"}
N_val = 100001
days_total = 100
GRID_ORIGINS = [70, 80, 90]
GROUP_TITLES = {"constant": "Constant β", "noise": "Gaussian noise (β = 0.75)",
                "time_varying": "Time-varying β", "forecastable": "Time-varying β, forecastable"}

fig_dir = "../../report_figures/conformal_rolling" + ("_learned" if cr.LEARN else "")
os.makedirs(fig_dir, exist_ok=True)
iv = pd.read_csv(os.path.join(cr.OUT, "rolling_conformal_intervals.csv"))
forecasts = cr.load_forecasts()


def readable(label):
    if label.startswith("noise_"):
        return f"{label.split('_')[1]}% Gaussian Noise"
    if label.startswith("beta_spline_"):
        return label.replace("beta_spline_", "").replace("_", " ").title()
    if label.startswith("beta_"):
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


def fan(ax, group, label, origin, variant="aci"):
    task = fo.sc.load_task(label, fo.data_folder)
    t = task["t"] * days_total
    ax.scatter(t, task["I"] * N_val, color=COLOURS["I"], s=10, alpha=0.5, zorder=3, label=r"$I(t)$ data")
    if "I_clean" in task:
        ax.plot(t, task["I_clean"] * N_val, color=COLOURS["I"], linewidth=1, linestyle=':',
                label=r"$I(t)$ noise-free")
    g = iv[(iv.variant == variant) & (iv.scenario == label) & (iv.origin == origin)].sort_values("day")
    for lvl, lo, hi, a in PI_BANDS:
        ax.fill_between(g.day, g[f"q{lo}"] * N_val, g[f"q{hi}"] * N_val, color=PI_COLOUR, alpha=a,
                        linewidth=0, label=f"{int(100 * lvl)}% PI")
    ax.plot(t, forecasts[(group, label)][origin] * N_val, color=PINN_COLOUR, linewidth=2, label=r"$I(t)$ PINN")
    ax.axvline(x=origin, color='gray', linestyle='--', linewidth=1.5, label='Forecast origin')
    ymax = 1.3 * task["I"].max() * N_val
    ax.set_ylim(-0.02 * ymax, ymax)


### day-80 fan charts, laid out as the report's forecast figures
for group, ncols, nrows, size in [("constant", 3, 1, (18, 5)), ("time_varying", 3, 2, (18, 10)),
                                  ("forecastable", 3, 2, (18, 10)), ("noise", 2, 2, (14, 10))]:
    scenarios = fo.GROUPS[group][0]
    fig, axes = plt.subplots(nrows, ncols, figsize=size, sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat[len(scenarios):]:
        ax.set_visible(False)
    for i, (ax, label) in enumerate(zip(axes_flat, scenarios)):
        fan(ax, group, label, 80)
        ax.set_title(readable(label))
        panel_label(ax, i)
        ax.legend(fontsize=8, loc='upper left')
    fig.supxlabel("Time (days)", fontsize=13)
    fig.supylabel("Number of infected individuals", fontsize=13, x=0.01)
    save(fig, f"rolling_conformal_forecast_{group}_80")

### fans from several origins: one row per scenario, one column per origin
for group, (scenarios, _, _) in fo.GROUPS.items():
    fig, axes = plt.subplots(len(scenarios), len(GRID_ORIGINS), figsize=(14, 3.6 * len(scenarios)),
                             sharex=True, squeeze=False)
    for r, label in enumerate(scenarios):
        for c, origin in enumerate(GRID_ORIGINS):
            ax = axes[r, c]
            fan(ax, group, label, origin)
            if r == 0:
                ax.set_title(f"Origin: day {origin}")
            if c == 0:
                ax.set_ylabel(readable(label), fontsize=13)
            panel_label(ax, r * len(GRID_ORIGINS) + c)
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.supxlabel("Time (days)", fontsize=13)
    fig.supylabel("Number of infected individuals", fontsize=13, x=0.0)
    save(fig, f"rolling_conformal_forecast_origins_{group}")

### empirical coverage against days ahead, by scenario group (all scenarios and origins pooled)
for variant in ("aci", "split"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    s = iv[iv.variant == variant]
    for i, (ax, group) in enumerate(zip(axes.flat, GROUP_TITLES)):
        g = s[s.group == group]
        for lvl in (50, 90, 95):
            cov = g.groupby("horizon")[f"covered_{lvl}"].mean() * 100
            ax.plot(cov.index, cov.values, color=LEVEL_COLOURS[lvl], marker=LEVEL_MARKERS[lvl], markersize=5,
                    linewidth=2, label=f"{lvl}% PI")
            ax.axhline(lvl, color=LEVEL_COLOURS[lvl], linestyle='--', linewidth=1)
        ax.set_title(GROUP_TITLES[group])
        ax.set_ylim(0, 105)
        panel_label(ax, i)
        ax.legend(fontsize=9, loc='lower left')
    fig.supxlabel("Days ahead of the forecast origin", fontsize=13)
    fig.supylabel("Empirical coverage (%)  [dashed: nominal]", fontsize=13, x=0.01)
    save(fig, f"rolling_conformal_coverage_by_horizon_{variant}")
