import os
import pickle
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

### Report figures for the England (UKHSA) data: the weekly data, and beta(t) / R(t) from the 7-compartment
### SEIR-PINN (PINN_hospitalisations_deaths.py), then every real-data figure collected into one folder in
### report order. Run after PINN_hospitalisations_deaths.py, conformal_PINN.py and compare_forecasts.py.
### No ground truth for beta on real data, so the reconstruction is shown two ways:
###   in-sample beta(t) from the last training window's fit (all weeks except the final forecast horizon)
###   R(t) = beta/gamma at 1 week ahead from every rolling window - the value each forecast actually used

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

N_val = 56_000_000
SERIES = {"cases": ("I", "New cases per week", "#d62728"),
          "hosp": ("H", "Admissions per week", "#ff7f0e"),
          "deaths": ("D", "Deaths per week", "#444444")}
BETA_COLOUR = "#7a85d6"
fig_dir = "../../report_figures/real_data"
dst = "../../report_figures_final/4_real_data"
os.makedirs(fig_dir, exist_ok=True)

dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")
with open("../../forecasts/PINN.pkl", "rb") as f:
    det = pickle.load(f)


def apply_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')


def panel_label(ax, i):
    ax.text(-0.06, 1.02, f"({chr(97 + i)})", transform=ax.transAxes,
            fontsize=13, verticalalignment='bottom', horizontalalignment='left')


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"saved {name}")


### the data the PINN is fitted to
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
for i, (ax, (s, (k, label, colour))) in enumerate(zip(axes, SERIES.items())):
    y = np.load(f"../../data/{k}_data_study.npy").ravel() * N_val
    ax.plot(dates[:len(y)], y, color=colour, marker='o', markersize=3, linewidth=1.5)
    ax.set_ylabel(label)
    panel_label(ax, i)
apply_date_axis(axes[-1])
fig.supxlabel("Week", fontsize=13)
save(fig, "real_data_england_weekly")

### beta(t) and R(t)
gamma = det["gamma"]
train_end = det["final_train_end"]
beta = det["beta_final"]
keys = sorted(k for k in det["all_beta"] if k[1] - k[0] == 1)
rt_dates = dates[[k[1] for k in keys]]
rt = np.array([det["all_beta"][k] for k in keys]) / gamma

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax = axes[0]
ax.plot(dates[:train_end], beta[:train_end], color=BETA_COLOUR, linewidth=2, label=r"$\beta(t)$, fitted weeks")
ax.plot(dates[train_end - 1:len(beta)], beta[train_end - 1:], color=BETA_COLOUR, linewidth=2, linestyle='--',
        label=r"$\beta(t)$, forecast weeks")
ax.axvline(dates[train_end - 1], color='gray', linestyle='--', linewidth=1.5, label='End of training data')
ax.set_ylabel(r"$\beta(t)$ (per week)")
ax.set_title(r"$\beta(t)$ from the final training window")
ax.legend(fontsize=9, loc='upper left')
panel_label(ax, 0)
ax = axes[1]
ax.plot(rt_dates, rt, color=BETA_COLOUR, marker='o', markersize=3, linewidth=1.5,
        label=r"$R(t) = \beta/\gamma$, 1 week ahead of each origin")
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label=r"$R = 1$")
ax.set_ylabel(r"$R(t)$")
ax.set_title(r"$R(t)$ used by each rolling forecast")
ax.legend(fontsize=9, loc='upper left')
panel_label(ax, 1)
apply_date_axis(axes[-1])
save(fig, "real_data_beta_Rt")

### sanity check against the published UKHSA/SPI-M-O R ranges for England (90% CI, contemporary, not revised).
### Each published range describes transmission 2-3 weeks before its publication date, so it is drawn over
### that window rather than at one shifted date. The PINN's R = beta/gamma takes no account of susceptible
### depletion or vaccination, so it is closer to an intrinsic than an effective R.
LAG_DAYS = (14, 21)
ukhsa = pd.read_csv("../../data/external/ukhsa_R_england.csv", comment="#", parse_dates=["publication_date"])
ukhsa["ref_start"] = ukhsa.publication_date - pd.Timedelta(days=LAG_DAYS[1])
ukhsa["ref_end"] = ukhsa.publication_date - pd.Timedelta(days=LAG_DAYS[0])
period = (pd.Timestamp(dates[0]), pd.Timestamp(dates[len(beta) - 1]))
ukhsa = ukhsa[(ukhsa.ref_end >= period[0]) & (ukhsa.ref_start <= period[1])]

rt_ahead = pd.Series(rt, index=pd.to_datetime(rt_dates))

### the R used by each 1-week-ahead forecast (data up to that week only) against the UKHSA ribbon, drawn at
### the middle of each estimate's 2-3 week reference window; the ribbon breaks where publications are more
### than 2 weeks apart. Both are real-time estimates, so this is the like-for-like comparison.
ref_mid = ukhsa.publication_date - pd.Timedelta(days=sum(LAG_DAYS) / 2)
gap = ref_mid.diff().dt.days.fillna(7).gt(14).cumsum()
fig, ax = plt.subplots(figsize=(12, 5))
for _, g in ukhsa.assign(mid=ref_mid).groupby(gap):
    ax.fill_between(g.mid, g.R_lower, g.R_upper, color="#c8c8c8", linewidth=0,
                    label="UKHSA published R range (90%)" if _ == 0 else None)
ax.plot(rt_ahead.index, rt_ahead.values, color="#3b4cc0", marker='o', markersize=4, linewidth=2,
        label=r"PINN $R = \beta/\gamma$ used by each 1-week-ahead forecast")
ax.axhline(1.0, color='#444444', linestyle='--', linewidth=1)
ax.set_ylabel(r"$R(t)$")
ax.legend(fontsize=10, loc='upper left', frameon=False)
ax.set_xlim(rt_ahead.index[0] - pd.Timedelta(days=14), period[1])
apply_date_axis(ax)
save(fig, "real_data_Rt_vs_ukhsa")

### agreement: PINN R averaged over each published estimate's reference window
rows = []
for r in ukhsa.itertuples():
    for name, s in (("one_week_ahead", rt_ahead),):
        w = s[(s.index >= r.ref_start) & (s.index <= r.ref_end + pd.Timedelta(days=6))]
        if len(w):
            m = float(w.mean())
            ukhsa_side = "above" if r.R_lower > 1 else "below" if r.R_upper < 1 else "spans"
            rows.append({"publication_date": r.publication_date.date(), "estimate": name, "R_lower": r.R_lower,
                         "R_upper": r.R_upper, "pinn_R": m, "in_range": r.R_lower <= m <= r.R_upper,
                         "ukhsa_side_of_1": ukhsa_side,
                         "same_side_of_1": ukhsa_side == "spans" or (m > 1) == (ukhsa_side == "above")})
agree = pd.DataFrame(rows)
os.makedirs(os.path.join(fig_dir, "tables"), exist_ok=True)
agree.to_csv(os.path.join(fig_dir, "tables", "Rt_vs_ukhsa.csv"), index=False)
print(agree.groupby("estimate")[["in_range", "same_side_of_1"]].mean().mul(100).round(0).assign(
    n=agree.groupby("estimate").size()).to_string())

### collect in report order (conformal figures from conformal_PINN.py, ACI on the scaled score)
FIGURES = [
    (os.path.join(fig_dir, "real_data_england_weekly"), "real_data_england_weekly"),
    ("conformal_pinn_fan_h1", "forecast_conformal_aci_1week"),
    ("conformal_pinn_coverage", "conformal_coverage_by_horizon"),
    ("conformal_pinn_sharpness", "conformal_sharpness_vs_coverage"),
    ("conformal_pinn_aci_level", "conformal_aci_level"),
    (os.path.join(fig_dir, "real_data_beta_Rt"), "beta_Rt_reconstruction"),
    (os.path.join(fig_dir, "real_data_Rt_vs_ukhsa"), "Rt_vs_ukhsa_published"),
    (os.path.join(fig_dir, "model_comparison_mase"), "model_comparison_mase"),                 ### compare_forecasts.py
    (os.path.join(fig_dir, "model_comparison_forecasts_1week"), "model_comparison_forecasts_1week"),
]
if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst)
for k, (src, new) in enumerate(FIGURES, start=1):
    for ext in ("pdf", "png"):
        if os.path.exists(f"{src}.{ext}"):
            shutil.copy(f"{src}.{ext}", os.path.join(dst, f"{k:02d}_{new}.{ext}"))
shutil.copy("conformal_pinn_metrics.csv", dst)
shutil.copy(os.path.join(fig_dir, "tables", "Rt_vs_ukhsa.csv"), dst)
shutil.copy(os.path.join(fig_dir, "tables", "model_comparison_mase.csv"), dst)
print(f"copied {len(FIGURES)} figures to {dst}")
