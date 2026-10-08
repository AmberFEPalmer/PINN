import argparse
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

import PINN_simulated_data as pc
import synthetic_scenarios as st

### Conformal prediction intervals for SEIR-PINN forecasts of I(t)
###
### 1. Rolling origin: for each forecast origin o (day 20 .. day 99) the PINN is fitted to days 0..o and
###    forecasts days o+1 .. o+H. The first origin is a full fit from random weights; each later origin
###    is warm-started from the previous fit, as in the England
###    pipeline, with a shorter Adam top-up and an L-BFGS polish.
### 2. Conformal intervals, per horizon h, calibrated on the PINN's own past errors at that horizon,
###    using only forecasts whose target day had already been observed at origin o (no look-ahead):
###      - scaled score r = (y - yhat) / yhat, since errors grow with the size of the epidemic
###      - finite-sample conformal bounds: asymmetric signed order statistics where the calibration set
###        is large enough, symmetric |r| otherwise, flagged "unguaranteed" below that
###      - adaptive conformal inference (Gibbs & Candes 2021) on each level, with each miss fed back
###        only once its target day is observed
###    The point forecast is the PINN's own prediction; intervals at 50%, 90% and 95%.

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", nargs="+", default=["beta_spline_three_waves", "beta_spline_gradual_decline", "noise_10"])
ap.add_argument("--first-origin", type=int, default=20)
ap.add_argument("--horizon", type=int, default=7)
ap.add_argument("--first-steps", type=int, default=50_000, help="Adam steps for the first origin")
ap.add_argument("--warm-steps", type=int, default=5000, help="Adam steps for each later, warm-started origin")
ap.add_argument("--lbfgs", type=int, default=2000)
ap.add_argument("--min-cal", type=int, default=15, help="smallest calibration set to issue intervals")
ap.add_argument("--gamma-aci", type=float, default=0.02)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-dir", default="../../results/PINN_simulated_data/conformal")
args = ap.parse_args()

LEVELS = {0.5: (0.25, 0.75), 0.9: (0.1, 0.9), 0.95: (0.025, 0.975)}
SCORE_FLOOR = 1e-7                     ### stops the scaled score dividing by ~0 (~0.01 people)
H = args.horizon


### --- 1. rolling-origin forecasts ---------------------------------------------------------------
def rolling_forecasts(task, fit, w0):
    t, I = task["t"], task["I"]
    last = len(t) - 1
    preds, betas, t_start = {}, {}, time.time()
    for o in range(args.first_origin, last):
        fit.set_task(t[:o + 1], I[:o + 1])
        if o == args.first_origin:
            fit.reset(w0)
            loss = fit.fit(args.first_steps, args.lbfgs)
        else:                              ### warm start from the previous origin's fit
            fit.reset_optimiser()
            loss = fit.fit(args.warm_steps, args.lbfgs)
        out = fit.predict(t)
        for h in range(1, H + 1):
            if o + h <= last:
                preds[(o, o + h)] = float(out["I"][o + h])
        betas[o] = out["beta"][:o + 1]
        if (o - args.first_origin) % 10 == 0:
            print(f"  origin day {o:3d} | loss {loss:.2e} | {time.time() - t_start:.0f}s", flush=True)
    return preds, betas


### --- 2. conformal intervals --------------------------------------------------------------------
def conformal_bounds(r, alpha):
    n = len(r); rs = np.sort(r)
    k_lo = int(np.floor((n + 1) * alpha / 2)); k_hi = int(np.ceil((n + 1) * (1 - alpha / 2)))
    if k_lo >= 1 and k_hi <= n:
        return rs[k_lo - 1], rs[k_hi - 1], "asymmetric"
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k <= n:
        q = np.sort(np.abs(r))[k - 1]
        return -q, q, "symmetric"
    return rs[0], rs[-1], "unguaranteed"


def conformal_intervals(preds, obs):
    """preds[(origin, target)] -> quantiles per forecast. Calibration at origin o uses only
    forecasts with target <= o; ACI updates are applied once a target is observed."""
    origins = sorted({k[0] for k in preds})
    alpha = {(h, lvl): 1 - lvl for h in range(1, H + 1) for lvl in LEVELS}
    pending, out, used = [], {}, {"asymmetric": 0, "symmetric": 0, "unguaranteed": 0}
    for o in origins:
        for (avail, key, err) in [p for p in pending if p[0] <= o]:
            alpha[key] += args.gamma_aci * ((1 - key[1]) - err)
        pending = [p for p in pending if p[0] > o]
        for h in range(1, H + 1):
            if (o, o + h) not in preds:
                continue
            r = np.array([(obs[k[1]] - v) / max(v, SCORE_FLOOR)
                          for k, v in preds.items() if k[1] - k[0] == h and k[1] <= o])
            if len(r) < args.min_cal:
                continue
            yhat = preds[(o, o + h)]
            row = {0.5: yhat}
            for lvl, (ql, qh) in LEVELS.items():
                a = float(np.clip(alpha[(h, lvl)], 1e-3, 0.999))
                lo, hi, form = conformal_bounds(r, a); used[form] += 1
                row[ql] = max(yhat + lo * max(yhat, SCORE_FLOOR), 0.0)
                row[qh] = max(yhat + hi * max(yhat, SCORE_FLOOR), 0.0)
            ### levels can use different bound types on small calibration sets; widen so the intervals
            ### are nested (95% contains 90% contains 50%). Widening only raises coverage.
            row[0.1], row[0.9] = min(row[0.1], row[0.25]), max(row[0.9], row[0.75])
            row[0.025], row[0.975] = min(row[0.025], row[0.1]), max(row[0.975], row[0.9])
            y = obs[o + h]
            for lvl, (ql, qh) in LEVELS.items():
                pending.append((o + h, (h, lvl), 0.0 if row[ql] <= y <= row[qh] else 1.0))
            out[(o, o + h)] = row
    return out, used


def wis(y, q):
    total = 0.5 * abs(y - q[0.5])
    for lo_q, hi_q in ((0.025, 0.975), (0.1, 0.9), (0.25, 0.75)):
        a = 2 * lo_q; l, u = q[lo_q], q[hi_q]
        total += (a / 2) * ((u - l) + (2 / a) * max(l - y, 0) + (2 / a) * max(y - u, 0))
    return total / 3.5


def evaluate(intervals, preds, obs):
    rows = []
    for h in range(1, H + 1):
        keys = sorted(k for k in intervals if k[1] - k[0] == h)
        if not keys:
            continue
        y = np.array([obs[k[1]] for k in keys]); yhat = np.array([preds[k] for k in keys])
        row = {"horizon": h, "n": len(keys), "mae_pct": 100 * np.mean(np.abs(yhat - y)) / np.mean(y)}
        for lvl, (ql, qh) in LEVELS.items():
            lo = np.array([intervals[k][ql] for k in keys]); hi = np.array([intervals[k][qh] for k in keys])
            row[f"coverage_{int(lvl * 100)}"] = float(np.mean((lo <= y) & (y <= hi)))
            row[f"rel_width_{int(lvl * 100)}"] = float(np.mean(hi - lo) / np.mean(y))
        row["wis_rel"] = float(np.mean([wis(obs[k[1]], intervals[k]) for k in keys]) / np.mean(y))
        rows.append(row)
    return pd.DataFrame(rows)


### --- run ---------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10,
                     'lines.linewidth': 2, 'axes.grid': True, 'grid.alpha': 0.3})
import os
os.makedirs(args.out_dir, exist_ok=True)
fit = pc.Fitter()
tf.random.set_seed(args.seed)
w0 = pc.create_pinn_model().get_weights()

for name in args.scenarios:
    print(f"\n=== {name} ===", flush=True)
    task = st.load_task(name)
    obs = {i: float(v) for i, v in enumerate(task["I"])}
    preds, betas = rolling_forecasts(task, fit, w0)
    intervals, used = conformal_intervals(preds, obs)
    metrics = evaluate(intervals, preds, obs)
    total = sum(used.values())
    print(f"bounds: {used['asymmetric']} asymmetric, {used['symmetric']} symmetric, {used['unguaranteed']} unguaranteed"
          f" ({100 * used['unguaranteed'] / max(total, 1):.1f}%)")
    print(metrics.round(3).to_string(index=False))
    with open(f"{args.out_dir}/{name}.pkl", "wb") as f:
        pickle.dump({"preds": preds, "intervals": intervals, "metrics": metrics, "betas": betas,
                     "obs": obs, "bounds_used": used, "config": vars(args)}, f)

    ### fan charts at 1 day and H days ahead, plus coverage by horizon
    N = st.N_VAL
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, h in zip(axes[:2], (1, H)):
        keys = sorted(k for k in intervals if k[1] - k[0] == h)
        d = np.array([k[1] for k in keys])
        g = lambda q: np.array([intervals[k][q] for k in keys]) * N
        ax.fill_between(d, g(0.025), g(0.975), color="#7397de", alpha=0.20, label="95% PI")
        ax.fill_between(d, g(0.1), g(0.9), color="#7397de", alpha=0.30, label="90% PI")
        ax.fill_between(d, g(0.25), g(0.75), color="#7397de", alpha=0.55, label="50% PI")
        ax.plot(d, g(0.5), color="#1f3d99", lw=1.8, label="PINN forecast")
        ax.scatter(np.arange(len(task["I"])), task["I"] * N, s=8, color="k", alpha=0.6, label="observed", zorder=3)
        ax.set_xlim(args.first_origin, len(task["I"]) - 1)
        ax.set_title(f"{name}: {h}-day-ahead"); ax.set_xlabel("Time (days)"); ax.set_ylabel("Infected individuals")
        ax.legend(fontsize=9, loc="upper left")
    for lvl in (50, 90, 95):
        axes[2].plot(metrics.horizon, metrics[f"coverage_{lvl}"], marker="o", label=f"{lvl}% interval")
        axes[2].axhline(lvl / 100, ls="--", lw=1, color="k")
    axes[2].set_ylim(0, 1.05); axes[2].set_xlabel("Horizon (days)"); axes[2].set_ylabel("Empirical coverage")
    axes[2].set_title("Coverage (dashed = nominal)"); axes[2].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{args.out_dir}/{name}.png", dpi=130); plt.close()
    print(f"saved {args.out_dir}/{name}.pkl and .png", flush=True)
