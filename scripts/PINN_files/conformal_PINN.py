import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import matplotlib.dates as mdates

### Conformal prediction intervals for the deterministic SEIR-PINN

### Keep the deterministic PINN's point forecast and use a conformal prediction wrapper
### contrasts two methods - split conformal (fixed level intervals from past errors) and adaptive (ACI) (online adjusted intervals)

### Plot style - fixed across all scripts
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

### england pop.
N_val = 56_000_000
SERIES = ["cases", "hosp", "deaths"]
LABEL = {"cases": "New cases per week", "hosp": "Admissions per week", "deaths": "Deaths per week"}
Forecast_horizon = 4

### calibration settings
MIN_CAL = 20              ### smallest calibration set we will score on. 
GAMMA_ACI = 0.02          ### ACI step size
SCORE_FLOOR = 1e-7        ### ~5.6 counts; stops the scaled score dividing by ~0
QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
LEVELS = {0.5: (0.25, 0.75), 0.9: (0.1, 0.9), 0.95: (0.025, 0.975)}   ### maps coverage level to the pair of quantiles

### load the deterministic PINN's rolling-origin forecasts
### keys are (origin, target index); values are proportions of N
with open("../../forecasts/PINN.pkl", "rb") as f:
    det = pickle.load(f)
all_pred, all_obs, all_naive = det["all_pred"], det["all_obs"], det["all_naive"]

### extract all unique forecast origins, prints summary and loads callendar dates 
keys_all = sorted(all_pred["cases"].keys())
ORIGINS = sorted({k[0] for k in keys_all})
print(f"deterministic PINN: {len(ORIGINS)} origins ({ORIGINS[0]}-{ORIGINS[-1]}), "
      f"horizons {sorted({k[1] - k[0] for k in keys_all})}, {len(keys_all)} forecasts per series")

study_dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")

def apply_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right')

### nonconformity score
### score turns an observation/prediction pair into a residual
def score(y, yhat, kind):
    if kind == "scaled":
        return (y - yhat) / max(yhat, SCORE_FLOOR)
    return y - yhat

def invert(yhat, r, kind):
    if kind == "scaled":
        return yhat + r * max(yhat, SCORE_FLOOR)
    return yhat + r

### check whether magnitude of error correlates with prediction size 
for s in SERIES:
    yh = np.array([all_pred[s][k] for k in keys_all])
    y = np.array([all_obs[s][k[1]] for k in keys_all])
    print(f"  {s:7s} corr(|resid|, pred) = {np.corrcoef(np.abs(y - yh), yh)[0, 1]:+.3f}")

### causal calibration set
### avoids leaking future info.
def calibration_residuals(s, t, h, kind):
    return np.array([score(all_obs[s][t2 + h], all_pred[s][(t2, t2 + h)], kind)
                     for t2 in ORIGINS if t2 + h <= t and (t2, t2 + h) in all_pred[s]])

### conformal quantiles with the finite-sample correction
def conformal_bounds(r, alpha):
    n = len(r)
    rs = np.sort(r)
    k_lo = int(np.floor((n + 1) * (alpha / 2)))
    k_hi = int(np.ceil((n + 1) * (1 - alpha / 2)))
    if k_lo >= 1 and k_hi <= n:
        return rs[k_lo - 1], rs[k_hi - 1], "asymmetric"
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k <= n:
        q = np.sort(np.abs(r))[k - 1]
        return -q, q, "symmetric"
    ### below the finite-sample floor for this level - widest available, flagged
    return rs[0], rs[-1], "unguaranteed"

### building the intervals
def build(kind="scaled", adaptive=False):
    qs = {s: {} for s in SERIES}
    used = {"asymmetric": 0, "symmetric": 0, "unguaranteed": 0}
    alpha_t = {(s, h, lvl): 1 - lvl for s in SERIES for h in range(1, Forecast_horizon + 1)
               for lvl in LEVELS}
    alpha_trace = {k: [] for k in alpha_t}
    ### chronological sweep so ACI only ever sees the past
    for t in ORIGINS:
        for h in range(1, Forecast_horizon + 1):
            key = (t, t + h)
            if key not in all_pred[SERIES[0]]:
                continue
            for s in SERIES:
                r = calibration_residuals(s, t, h, kind)
                if len(r) < MIN_CAL:
                    continue
                yhat = all_pred[s][key]
                row = {}
                for lvl, (ql, qh) in LEVELS.items():
                    a = alpha_t[(s, h, lvl)] if adaptive else 1 - lvl
                    a = float(np.clip(a, 1e-3, 0.999))
                    lo, hi, form = conformal_bounds(r, a)
                    used[form] += 1
                    row[ql] = max(invert(yhat, lo, kind), 0.0)
                    row[qh] = max(invert(yhat, hi, kind), 0.0)
                    alpha_trace[(s, h, lvl)].append(a)
                ### median = the PINN's own point forecast
                row[0.5] = max(yhat, 0.0)
                row["median_biascorr"] = max(invert(yhat, float(np.median(r)), kind), 0.0)
                qs[s][key] = row
                if adaptive:
                    for lvl, (ql, qh) in LEVELS.items():
                        y = all_obs[s][t + h]
                        err = 0.0 if row[ql] <= y <= row[qh] else 1.0
                        alpha_t[(s, h, lvl)] += GAMMA_ACI * ((1 - lvl) - err)
    return qs, used, alpha_trace

### weighted interval score 
PI_PAIRS = [(0.025, 0.975), (0.1, 0.9), (0.25, 0.75)]

def wis(y, q):
    total = 0.5 * abs(y - q[0.5])
    for lo_q, hi_q in PI_PAIRS:
        alpha = 2 * lo_q
        l, u = q[lo_q], q[hi_q]
        s = (u - l)
        if y < l:
            s += (2 / alpha) * (l - y)
        if y > u:
            s += (2 / alpha) * (y - u)
        total += (alpha / 2) * s
    return total / (len(PI_PAIRS) + 0.5)

def evaluate(qs, tag, restrict=None):
    rows = []
    for s in SERIES:
        for h in range(1, Forecast_horizon + 1):
            hk = sorted([k for k in qs[s] if k[1] - k[0] == h], key=lambda k: k[1])
            if restrict is not None:
                hk = [k for k in hk if k in restrict]
            if not hk:
                continue
            obs = np.array([all_obs[s][k[1]] for k in hk])
            mu = np.array([all_pred[s][k] for k in hk])
            med = np.array([qs[s][k]["median_biascorr"] for k in hk])
            naive = np.array([all_naive[s][k] for k in hk])
            mae_naive = np.mean(np.abs(naive - obs)) + 1e-12
            w = {lvl: np.mean([qs[s][k][qh] - qs[s][k][ql] for k in hk]) * N_val
                 for lvl, (ql, qh) in LEVELS.items()}
            cov = {lvl: np.mean([qs[s][k][ql] <= all_obs[s][k[1]] <= qs[s][k][qh] for k in hk])
                   for lvl, (ql, qh) in LEVELS.items()}
            obs_mean = float(obs.mean()) * N_val
            rows.append({
                "method": tag, "series": s, "horizon": h, "n_forecasts": len(hk),
                "mase_point": np.mean(np.abs(mu - obs)) / mae_naive,
                "mase_biascorr": np.mean(np.abs(med - obs)) / mae_naive,
                "coverage_50": cov[0.5], "coverage_90": cov[0.9], "coverage_95": cov[0.95],
                "width_50": w[0.5], "width_90": w[0.9], "width_95": w[0.95],
                "rel_width_50": w[0.5] / obs_mean, "rel_width_90": w[0.9] / obs_mean,
                "rel_width_95": w[0.95] / obs_mean, "mean_obs": obs_mean,
                "wis": np.mean([wis(all_obs[s][k[1]], qs[s][k]) for k in hk]) * N_val,
            })
    return pd.DataFrame(rows)

### build all four variants
variants = {}
for kind in ["scaled", "additive"]:
    for adaptive in [False, True]:
        tag = f"{'ACI' if adaptive else 'split'}-{kind}"
        qs, used, atr = build(kind, adaptive)
        variants[tag] = (qs, atr)
        print(f"{tag:16s} | {len(qs['cases'])} scored forecasts/series | bounds: "
              f"{used['asymmetric']} asymmetric, {used['symmetric']} symmetric fallback, "
              f"{used['unguaranteed']} unguaranteed")
        if used["unguaranteed"]:
            print(f"{'':16s}   ({used['unguaranteed']}/{sum(used.values())} = "
                  f"{used['unguaranteed'] / sum(used.values()):.1%} of bounds sat below the "
                  f"finite-sample floor after ACI lowered alpha; widest residual used)")

### score every variant on the SAME forecast set, so widths are comparable
common = set.intersection(*[set(v[0]["cases"].keys()) for v in variants.values()])
print(f"\ncommon scored set: {len(common)} forecasts per series "
      f"(origins {min(k[0] for k in common)}-{max(k[0] for k in common)}, "
      f"first {MIN_CAL} calibration points per horizon held back)")

metrics = pd.concat([evaluate(qs, tag, restrict=common) for tag, (qs, _) in variants.items()],
                    ignore_index=True)

try:
    with open("../../forecasts/PINN_variational.pkl", "rb") as f:
        vi = pickle.load(f)
    vq = vi["quantiles"]
    vi_rows = []
    for s in SERIES:
        for h in range(1, Forecast_horizon + 1):
            hk = [k for k in common if k[1] - k[0] == h and k in vq[s]]
            if not hk:
                continue
            obs = np.array([all_obs[s][k[1]] for k in hk])
            naive = np.array([all_naive[s][k] for k in hk])
            mae_naive = np.mean(np.abs(naive - obs)) + 1e-12
            mu = np.array([vi["all_pred"][s][k] for k in hk])
            med = np.array([vq[s][k][0.5] for k in hk])
            w = {lvl: np.mean([vq[s][k][qh] - vq[s][k][ql] for k in hk]) * N_val
                 for lvl, (ql, qh) in LEVELS.items()}
            cov = {lvl: np.mean([vq[s][k][ql] <= all_obs[s][k[1]] <= vq[s][k][qh] for k in hk])
                   for lvl, (ql, qh) in LEVELS.items()}
            obs_mean = float(obs.mean()) * N_val
            vi_rows.append({
                "method": f"B-PINN VI (KL={vi['config']['kl_max']})", "series": s, "horizon": h,
                "n_forecasts": len(hk),
                "mase_point": np.mean(np.abs(mu - obs)) / mae_naive,
                "mase_biascorr": np.mean(np.abs(med - obs)) / mae_naive,
                "coverage_50": cov[0.5], "coverage_90": cov[0.9], "coverage_95": cov[0.95],
                "width_50": w[0.5], "width_90": w[0.9], "width_95": w[0.95],
                "rel_width_50": w[0.5] / obs_mean, "rel_width_90": w[0.9] / obs_mean,
                "rel_width_95": w[0.95] / obs_mean, "mean_obs": obs_mean,
                "wis": np.mean([wis(all_obs[s][k[1]], vq[s][k]) for k in hk]) * N_val,
            })
    metrics = pd.concat([metrics, pd.DataFrame(vi_rows)], ignore_index=True)
except FileNotFoundError:
    print("PINN_variational.pkl not found - skipping the B-PINN comparison")

metrics.to_csv("conformal_pinn_metrics.csv", index=False)

pd.set_option("display.width", 250)
print("\n=== mean over the 12 series x horizon cells (nominal 0.50 / 0.90 / 0.95) ===")
summ = (metrics.groupby("method")[["mase_point", "coverage_50", "coverage_90",
                                   "coverage_95", "rel_width_90", "wis"]].mean())
print(summ.round(3).to_string())
print("\nmedian-residual bias correction, mean MASE (recorded, not used): "
      + ", ".join(f"{m}={metrics[metrics.method == m].mase_biascorr.mean():.3f}"
                  for m in metrics.method.unique()
                  if metrics[metrics.method == m].mase_biascorr.notna().any()))

print("\n=== per-series coverage and 90% width, in counts ===")
print(metrics.pivot_table(index=["series", "horizon"], columns="method",
                          values=["coverage_95", "width_90"]).round(3).to_string())
print("\nmetrics written to conformal_pinn_metrics.csv")

### primary method for reporting: ACI on the scaled score
PRIMARY = "ACI-scaled"
qs_p = variants[PRIMARY][0]

def horizon_arrays(s, h, qs):
    hk = sorted([k for k in qs[s] if k[1] - k[0] == h and k in common], key=lambda k: k[1])
    idx = np.array([k[1] for k in hk], dtype=int)
    g = lambda q: np.array([qs[s][k][q] for k in hk]) * N_val
    return (study_dates[idx], np.array([all_pred[s][k] for k in hk]) * N_val,
            np.array([all_obs[s][k[1]] for k in hk]) * N_val,
            g(0.25), g(0.75), g(0.1), g(0.9), g(0.025), g(0.975))

### Plot 1: fan chart, one row per series, 1-week-ahead
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, SERIES):
    d, mu, obs, lo50, hi50, lo90, hi90, lo95, hi95 = horizon_arrays(s, 1, qs_p)
    ax.fill_between(d, lo95, hi95, color="#7397de", alpha=0.20, label="95% PI")
    ax.fill_between(d, lo90, hi90, color="#7397de", alpha=0.30, label="90% PI")
    ax.fill_between(d, lo50, hi50, color="#7397de", alpha=0.55, label="50% PI")
    ax.plot(d, mu, color="#1f3d99", lw=1.8, label="PINN point forecast")
    ax.plot(d, obs, color="#004F94", lw=1.3, ls=":", label="Observed")
    ax.set_ylabel(LABEL[s]); ax.legend(fontsize=9, loc="upper right"); apply_date_axis(ax)
axes[0].set_title(f"England: deterministic SEIR-PINN with {PRIMARY} prediction intervals, "
                  "1-week-ahead")
plt.tight_layout(); plt.savefig("conformal_pinn_fan_h1.png", dpi=150); plt.show()

### Plot 2: calibration - every method, nominal vs empirical
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
meths = list(metrics.method.unique())
for ax, lvl in zip(axes, [0.5, 0.9, 0.95]):
    col = f"coverage_{int(lvl * 100)}"
    x = np.arange(Forecast_horizon)
    width = 0.8 / len(meths)
    for i, m in enumerate(meths):
        sub = metrics[metrics.method == m].groupby("horizon")[col].mean()
        ax.bar(x + i * width - 0.4 + width / 2, sub.values, width, label=m)
    ax.axhline(lvl, color="k", lw=1.2, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"{h}wk" for h in range(1, Forecast_horizon + 1)])
    ax.set_title(f"{int(lvl * 100)}% interval"); ax.set_xlabel("Horizon"); ax.set_ylim(0, 1.05)
axes[0].set_ylabel("Empirical coverage"); axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Coverage: conformal variants vs the variational B-PINN "
             "(dashed = nominal, averaged over series)", fontsize=14)
plt.tight_layout(); plt.savefig("conformal_pinn_coverage.png", dpi=150); plt.show()

### Plot 3: sharpness against calibration - width is only meaningful next to coverage
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, s in zip(axes, SERIES):
    for m in meths:
        sub = metrics[(metrics.method == m) & (metrics.series == s)]
        ax.scatter(sub.coverage_95, sub.rel_width_95, s=55, label=m)
    ax.axvline(0.95, color="k", lw=1.2, ls="--")
    ax.set_xlabel("Empirical 95% coverage"); ax.set_title(LABEL[s])
    ax.set_xlim(0, 1.05)
axes[0].set_ylabel("95% width / mean observation"); axes[0].legend(fontsize=8)
fig.suptitle("Sharpness vs calibration, one point per horizon "
             "(dashed = nominal 95%; ideal is on the line and low)", fontsize=14)
plt.tight_layout(); plt.savefig("conformal_pinn_sharpness.png", dpi=150); plt.show()

### Plot 4: the ACI level adapting over time - shows where the error
### distribution shifted enough that the nominal level had to move
fig, ax = plt.subplots(figsize=(12, 5))
for s in SERIES:
    tr = variants["ACI-scaled"][1][(s, 1, 0.95)]
    ax.plot(range(len(tr)), 1 - np.array(tr), label=f"{s} (h=1)")
ax.axhline(0.95, color="k", lw=1.2, ls="--", label="nominal 0.95")
ax.set_xlabel("Forecast origin (chronological)"); ax.set_ylabel("Level used by ACI")
ax.set_title(f"ACI adapting the 95% level online (gamma={GAMMA_ACI})")
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("conformal_pinn_aci_level.png", dpi=150); plt.show()

with open("../../forecasts/PINN_conformal.pkl", "wb") as f:
    pickle.dump({
        "all_pred": all_pred, "all_obs": all_obs, "all_naive": all_naive,
        "quantiles": {tag: qs for tag, (qs, _) in variants.items()},
        "primary": PRIMARY,
        "scored_keys": sorted(common),
        "metrics": metrics,
        "config": {"min_cal": MIN_CAL, "gamma_aci": GAMMA_ACI,
                   "score_floor": SCORE_FLOOR, "quantiles": QUANTILES,
                   "source": "forecasts/PINN.pkl"},
    }, f)
print("\nsaved conformal intervals to ../../forecasts/PINN_conformal.pkl")
