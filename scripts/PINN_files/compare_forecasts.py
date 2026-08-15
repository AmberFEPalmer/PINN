import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'lines.linewidth': 2, 'axes.grid': True, 'grid.alpha': 0.3,
})

### data / dates
n_val = 56_000_000
i_data = np.load("../../data/I_data_study.npy").reshape(-1).astype("float64")
h_data = np.load("../../data/H_data_study.npy").reshape(-1).astype("float64")
d_data = np.load("../../data/D_data_study.npy").reshape(-1).astype("float64")
study_dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")
n_total_points = len(i_data)
study_dates = study_dates[:n_total_points]

first_train_weeks = 17                      # forecasts begin here — used to crop the x-axis
series = ["cases", "deaths","hosp"]
data = {"cases": i_data, "deaths": d_data, "hosp": h_data}
label = {"cases": "Cases", "hosp": "Hospital Admissions", "deaths": "Deaths"}

forecast_dir = "../../forecasts"
models = {
    "LSTM": {"file": "preds_lstm.pkl", "color": "#ceb7e6"},
    "Prophet": {"file": "preds_prophet.pkl", "color": "#f1f890"},
    "Transformer": {"file": "preds_transformer.pkl", "color": "#92f49a"},
    "PINN": {"file": "PINN.pkl", "color": "#f199c6"},
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

def horizon_series(store, s, h):
    pairs = [(fidx, pv) for (te, fidx), pv in store["all_pred"][s].items() if fidx - te == h]
    pairs.sort()
    idx = np.array([p[0] for p in pairs], dtype=int)
    pred = np.array([p[1] for p in pairs], dtype=float)
    return idx, pred

def mase(store, s, h):
    idx, pred = horizon_series(store, s, h)
    if len(idx) == 0:
        return np.nan
    obs = data[s][idx]; naive = data[s][idx - h]
    return np.mean(np.abs(pred - obs)) / (np.mean(np.abs(naive - obs)) + 1e-12)

def apply_date_axis(ax, crop=True):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right')
    if crop:
        ax.set_xlim(study_dates[first_train_weeks - 1], study_dates[-1])

###  MASE of each model
horizons = [1, 2, 3, 4]
x = np.arange(len(horizons))
width = 0.8 / len(names)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, s in zip(axes, series):
    for i, n in enumerate(names):
        vals = [mase(loaded[n], s, h) for h in horizons]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               color=models[n]["color"], label=n)
    ax.axhline(1.0, color="black", lw=1.2, ls="--", label="naive (=1)")
    ax.set_xticks(x); ax.set_xticklabels([f"{h}wk" for h in horizons])
    ax.set_title(label[s]); ax.set_xlabel("Horizon"); ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylabel("MASE")
axes[0].legend(fontsize=9, loc="upper left")
fig.suptitle("Model Accuracy by Forecasting Horizon", fontsize=14)
plt.tight_layout(); plt.savefig("compare_mase_bars.png", dpi=150); plt.show()
 
### Model predictions 1 week
compare_h = 1   # headline 1-week-ahead; set to 4 to stress-test the hardest horizon
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, series):
    ax.plot(study_dates, data[s] * n_val, color="#004F94", lw=1.2, ls=":",
            label="observed", zorder=1)
    for name, store in loaded.items():
        idx, pred = horizon_series(store, s, compare_h)
        order = np.argsort(idx)
        ax.plot(study_dates[idx[order]], pred[order] * n_val,
                color=models[name]["color"], lw=1.6, alpha=0.9, label=name, zorder=3)
    ax.set_ylabel(label[s]); ax.legend(fontsize=8, loc="upper right"); apply_date_axis(ax)
axes[0].set_title(f"england: model comparison — {compare_h}-week-ahead, all series")
plt.tight_layout(); plt.savefig("compare_allseries_h1.png", dpi=150); plt.show()