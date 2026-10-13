import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

### England (UKHSA) data: the SEIR-PINN against the LSTM, Transformer and Prophet baselines.
### Every model is scored on the same 72 forecast origins x 4 horizons (weekly, rolling origin).
### MASE = mean |forecast - observed| / mean |naive - observed|, naive = last observed week (persistence);
### below 1 beats persistence. Run after PINN_hospitalisations_deaths.py and the baseline scripts in
### ../baseline_models; real_data_figures.py collects the figures.

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

### data / dates
n_val = 56_000_000
i_data = np.load("../../data/I_data_study.npy").reshape(-1).astype("float64")
h_data = np.load("../../data/H_data_study.npy").reshape(-1).astype("float64")
d_data = np.load("../../data/D_data_study.npy").reshape(-1).astype("float64")
study_dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")
n_total_points = len(i_data)
study_dates = study_dates[:n_total_points]

first_train_weeks = 17                      ### forecasts begin here - used to crop the x-axis
series = ["cases", "hosp", "deaths"]
data = {"cases": i_data, "deaths": d_data, "hosp": h_data}
label = {"cases": "New cases per week", "hosp": "Admissions per week", "deaths": "Deaths per week"}
title = {"cases": "Cases", "hosp": "Hospital admissions", "deaths": "Deaths"}

forecast_dir = "../../forecasts"
fig_dir = "../../report_figures/real_data"
os.makedirs(os.path.join(fig_dir, "tables"), exist_ok=True)
### categorical colours: the default validated palette, slots 1-4 in fixed order
models = {
    "PINN": {"file": "PINN.pkl", "color": "#2a78d6"},
    "LSTM": {"file": "preds_lstm.pkl", "color": "#eb6834"},
    "Transformer": {"file": "preds_transformer.pkl", "color": "#1baf7a"},
    "Prophet": {"file": "preds_prophet.pkl", "color": "#eda100"},
}

loaded = {}
for name, cfg in models.items():
    path = os.path.join(forecast_dir, cfg["file"])
    try:
        with open(path, "rb") as f:
            loaded[name] = pickle.load(f)
        print(f"loaded {name:12s} <- {path}")
    except FileNotFoundError:
        print(f"skip   {name:12s} ({path} not found)")
if not loaded:
    raise SystemExit("no prediction dumps found in ../../forecasts/")
names = list(loaded.keys())
keys = [set(loaded[n]["all_pred"][s]) for n in names for s in series]
assert all(k == keys[0] for k in keys), "models were not scored on the same forecasts"


def horizon_series(store, s, h):
    pairs = sorted((fidx, pv) for (te, fidx), pv in store["all_pred"][s].items() if fidx - te == h)
    return np.array([p[0] for p in pairs], dtype=int), np.array([p[1] for p in pairs], dtype=float)


def mase(store, s, h):
    idx, pred = horizon_series(store, s, h)
    obs = data[s][idx]
    naive = data[s][idx - h]
    return np.mean(np.abs(pred - obs)) / (np.mean(np.abs(naive - obs)) + 1e-12)


def apply_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')
    ax.set_xlim(study_dates[first_train_weeks - 1], study_dates[-1])


def panel_label(ax, i):
    ax.text(-0.08, 1.02, f"({chr(97 + i)})", transform=ax.transAxes,
            fontsize=13, verticalalignment='bottom', horizontalalignment='left')


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved {name}")


### MASE table
horizons = [1, 2, 3, 4]
table = pd.DataFrame([{"model": n, "series": s, "horizon": h, "MASE": mase(loaded[n], s, h)}
                      for n in names for s in series for h in horizons])
table.to_csv(os.path.join(fig_dir, "tables", "model_comparison_mase.csv"), index=False)
print(table.pivot_table(index=["series", "horizon"], columns="model", values="MASE")[names].round(3).to_string())

### MASE of each model by horizon
x = np.arange(len(horizons))
width = 0.8 / len(names)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for j, (ax, s) in enumerate(zip(axes, series)):
    for i, n in enumerate(names):
        vals = table[(table.model == n) & (table.series == s)].MASE.values
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.9, color=models[n]["color"], label=n)
    ax.axhline(1.0, color="#444444", lw=1.2, ls="--", label="Naive (persistence)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} wk" for h in horizons])
    ax.set_title(title[s])
    ax.set_xlabel("Forecast horizon")
    ax.grid(False, axis="x")
    panel_label(ax, j)
axes[0].set_ylabel("MASE (lower is better)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(names) + 1, frameon=False, bbox_to_anchor=(0.5, 1.07))
save(fig, "model_comparison_mase")

### 1-week-ahead forecasts of every model against the observations
compare_h = 1
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for j, (ax, s) in enumerate(zip(axes, series)):
    ax.plot(study_dates, data[s] * n_val, color="#222222", lw=0, marker='o', markersize=3, label="Observed",
            zorder=4)
    for name, store in loaded.items():
        idx, pred = horizon_series(store, s, compare_h)
        ax.plot(study_dates[idx], pred * n_val, color=models[name]["color"], lw=1.8, label=name, zorder=3)
    ax.set_ylabel(label[s])
    apply_date_axis(ax)
    panel_label(ax, j)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(names) + 1, frameon=False, bbox_to_anchor=(0.5, 1.03))
save(fig, "model_comparison_forecasts_1week")
