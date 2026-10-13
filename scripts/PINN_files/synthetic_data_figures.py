import os
import matplotlib.pyplot as plt
import forecast_origins as fo

### Synthetic SEIR data for the time-varying beta scenarios used in the forecast figures (forecast_origins.GROUPS):
### one row per scenario, S/E/I/R on the left and the true beta(t) on the right, as in the report's data figures.
### The original report figure (fig07_SEIR_time_varying_beta) shows Two Waves where the forecasts use Three Waves.
### Also the SEIR curves for the constant beta = 0.75 scenario on its own.

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

COLOURS = {"S": "#2ca02c", "E": "#ff7f0e", "I": "#d62728", "R": "#1f3fbf"}
BETA_COLOUR = "#7a85d6"
N_val = 100001
days_total = 100
fig_dir = "../../report_figures"


def readable(label):
    return label.replace("beta_spline_", "").replace("_", " ").title()


def panel_label(ax, i):
    ax.text(-0.08, 1.02, f"({chr(97 + i)})", transform=ax.transAxes,
            fontsize=13, verticalalignment='bottom', horizontalalignment='left')


def main():
    scenarios = fo.GROUPS["time_varying"][0]
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(14, 3.8 * len(scenarios)), sharex=True)
    for r, label in enumerate(scenarios):
        task = fo.sc.load_task(label, fo.data_folder)
        t = task["t"] * days_total
        ax = axes[r, 0]
        for k in ("S", "E", "I", "R"):
            ax.plot(t, task[k] * N_val, color=COLOURS[k], label=rf"${k}(t)$")
        ax.set_title(readable(label))
        ax.set_ylabel("Individuals")
        ax.legend(fontsize=9, loc='center left')
        panel_label(ax, 2 * r)
        ax = axes[r, 1]
        ax.scatter(t, task["beta"], color=BETA_COLOUR, s=14, alpha=0.8)
        ax.set_title(rf"$\beta(t)$ — {readable(label)}")
        ax.set_ylabel(r"$\beta(t)$")
        ax.set_ylim(0, 1.1 * task["beta"].max())
        panel_label(ax, 2 * r + 1)
    for ax in axes[-1]:
        ax.set_xlabel("Time (days)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"fig07_SEIR_time_varying_beta_forecast_scenarios.{ext}"),
                    bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("saved fig07_SEIR_time_varying_beta_forecast_scenarios")

    ### constant beta = 0.75 (R0 = beta/gamma = 3): the scenario used throughout the report
    task = fo.sc.load_task("beta_0.75", fo.data_folder)
    t = task["t"] * days_total
    ### Okabe-Ito colours: saturated and colour-blind safe, same S/E/I/R hue convention as the other figures
    strong = {"S": "#009E73", "E": "#E69F00", "I": "#D55E00", "R": "#0072B2"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in ("S", "E", "I", "R"):
        ax.plot(t, task[k] * N_val, color=strong[k], linewidth=2.8, label=rf"${k}(t)$")
    ax.grid(False)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Individuals")
    ax.legend(fontsize=11, loc='center right')
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"fig02_SEIR_constant_beta_0.75.{ext}"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("saved fig02_SEIR_constant_beta_0.75")


if __name__ == "__main__":
    main()
