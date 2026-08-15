import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import warnings
warnings.filterwarnings("ignore")

### prophet model for 1, 2, 3 and 4 week predictions
### forecasting cases, deaths and hospitalisations
### https://facebook.github.io/prophet/docs/quick_start.html

### silence prophet / cmdstanpy chatter
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

### plot style — fixed across all scripts
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

colours = {
    "S": "#2ca02c",
    "E": "#ff7f0e",
    "I": "#d62728",
    "R": "#1f33b4",
}

### load in data
n_val = 56_000_000
i_data = np.load("../../data/I_data_study.npy").reshape(-1).astype("float64")
h_data = np.load("../../data/H_data_study.npy").reshape(-1).astype("float64")
d_data = np.load("../../data/D_data_study.npy").reshape(-1).astype("float64")
study_dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")
n_total_points = len(i_data)
study_dates = study_dates[:n_total_points]

series = ["cases", "deaths", "hosp"]
data = {"cases": i_data, "deaths": d_data, "hosp": h_data}
label = {"cases": "daily cases", "hosp": "daily admissions", "deaths": "daily deaths"}

### configure prophet
first_train_weeks = 17
forecast_horizon = 4
log_transform = True            # fit on log1p scale (counts are strictly positive, multiplicative dynamics)
yearly_seasonality = False      # series too short / irregular for a stable yearly component
changepoint_prior_scale = 0.5
n_changepoints_cap = 25

print(f"study grid: {n_total_points} weeks ({study_dates[0]} -> {study_dates[-1]})")

def to_frame(values, dates):
    """build the ds/y dataframe prophet expects"""
    y = np.clip(values, 0, None)
    if log_transform:
        y = np.log1p(y)
    return pd.DataFrame({"ds": pd.to_datetime(dates), "y": y})


def fit_prophet(df):
    """fit one prophet model on the training history (weekly data)"""
    m = Prophet(
        growth="linear",
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,   # data is already weekly — no intra-week cycle to learn
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        n_changepoints=min(n_changepoints_cap, max(1, len(df) // 2)),
        uncertainty_samples=0,      # point forecasts only — big speed-up
    )
    m.fit(df)
    return m


def forecast_horizons(m, last_date, n_ahead):
    """native multi-step forecast: 1..n_ahead weeks beyond the last training date"""
    future = pd.DataFrame({
        "ds": pd.date_range(start=pd.Timestamp(last_date) + pd.Timedelta(days=7),
                            periods=n_ahead, freq="7D")
    })
    fc = m.predict(future)
    yhat = fc["yhat"].to_numpy()
    if log_transform:
        yhat = np.expm1(yhat)
    return np.clip(yhat, 1e-6, None)


### rolling forecast (train once per origin per series, horizons 1-4 in one shot)
all_pred  = {s: {} for s in series}
all_naive = {s: {} for s in series}
all_obs   = {s: {} for s in series}

full3 = np.column_stack([data["cases"], data["deaths"], data["hosp"]])

for train_end in range(first_train_weeks, n_total_points - forecast_horizon + 1):
    origin = train_end - 1
    n_ahead = min(forecast_horizon, n_total_points - 1 - origin)

    for ci, s in enumerate(series):
        df = to_frame(full3[:train_end, ci], study_dates[:train_end])
        model = fit_prophet(df)
        preds = forecast_horizons(model, study_dates[origin], n_ahead)

        for h in range(1, n_ahead + 1):
            fidx = origin + h
            all_pred[s][(origin, fidx)]  = float(preds[h - 1])
            all_naive[s][(origin, fidx)] = float(full3[origin, ci])
            all_obs[s][fidx]             = float(full3[fidx, ci])

    print(f"origin {origin:3d} (train weeks 1-{train_end}) done | "
          f"1wk cases={all_pred['cases'][(origin, origin+1)]*n_val:,.0f}")

### collect / mase / plots
def collect(s_name):
    hres = {h: {"idx": [], "pred": [], "obs": [], "naive": []} for h in range(1, 5)}
    for (te, fidx), pv in all_pred[s_name].items():
        h = fidx - te
        hres[h]["idx"].append(fidx); hres[h]["pred"].append(pv)
        hres[h]["obs"].append(all_obs[s_name][fidx]); hres[h]["naive"].append(all_naive[s_name][(te, fidx)])
    return hres

def apply_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right')

print("\nprophet mase by horizon (model mae / naive mae; <1 beats naive):")
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, series):
    hres = collect(s)
    ax.plot(study_dates, data[s] * n_val, color="#004F94", lw=1.5, ls=":", label="observed")
    for h in range(1, 5):
        idx = np.array(hres[h]["idx"], dtype=int)
        order = np.argsort(idx)
        ax.plot(study_dates[idx[order]], np.array(hres[h]["pred"])[order] * n_val,
                lw=1.2, alpha=0.85, label=f"{h} wk ahead")
    ax.set_ylabel(label[s]); ax.legend(fontsize=8, loc="upper right"); apply_date_axis(ax)
    line = f"  {s:8s}"
    for h in range(1, 5):
        p = np.array(hres[h]["pred"]); o = np.array(hres[h]["obs"]); n = np.array(hres[h]["naive"])
        mase = np.mean(np.abs(p - o)) / (np.mean(np.abs(n - o)) + 1e-12)
        line += f"  {h}wk={mase:.3f}"
    print(line)
axes[0].set_title("england: prophet per-horizon vs observed")
plt.tight_layout(); plt.savefig("prophet_perhorizon.png", dpi=150); plt.show()

### per-horizon 2x2 grid for cases
plot_series, label_str = "cases", "new cases per week"
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for h, ax in zip(range(1, 5), axes.flatten()):
    hd, hp, ho, hn = [], [], [], []
    for (te, fidx), pv in all_pred[plot_series].items():
        if fidx - te == h:
            hd.append(study_dates[fidx]); hp.append(pv)
            ho.append(all_obs[plot_series][fidx]); hn.append(all_naive[plot_series][(te, fidx)])
    if hd:
        order = np.argsort(np.array(hd, dtype='datetime64[D]'))
        hd = np.array(hd, dtype='datetime64[D]')[order]
        hp = np.array(hp)[order] * n_val; ho = np.array(ho)[order] * n_val; hn = np.array(hn)[order] * n_val
        ax.plot(hd, ho, color="#004F94", lw=1.5, label="observed")
        ax.plot(hd, hp, color="#23e623", lw=1.5, label=f"prophet {h}-week")
        ax.plot(hd, hn, color="orange", lw=1.0, ls="--", label="naive baseline")
    ax.set_title(f"{h}-week-ahead forecast"); ax.set_ylabel(label_str)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3); apply_date_axis(ax)
fig.suptitle("prophet: cases (1-4 weeks ahead)", fontsize=14)
plt.tight_layout(); plt.savefig("prophet_cases_grid.png", dpi=150); plt.show()

### Save
import os, pickle
os.makedirs("../../forecasts", exist_ok=True)
with open("../../forecasts/preds_prophet.pkl", "wb") as f:
    pickle.dump({"all_pred": all_pred, "all_obs": all_obs, "all_naive": all_naive}, f)