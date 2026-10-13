import os
import numpy as np
import pandas as pd
import forecast_origins as fo

### Conformal prediction intervals for the synthetic multi-origin forecasts (forecast_origins.py).
### Same score and quantiles as conformal_PINN.py (scaled residual, asymmetric conformal quantiles with
### the finite-sample correction, 50/90/95% levels). The synthetic runs have only 4 origins per scenario,
### too few for conformal_PINN.py's causal calibration, so calibration is leave-one-scenario-out instead:
### intervals for a scenario use the errors of every OTHER synthetic scenario (same sigma/gamma setup, same
### horizon, all origins), and nothing from the scenario itself. The point forecast is the median over seeds.
### Coverage is scored against the observations and reported by scenario group, so any failure of the
### exchangeability assumption (very different scenarios sharing one calibration pool) shows up there.

MIN_CAL = 20              ### smallest calibration set we will score on, as in conformal_PINN.py
SCORE_FLOOR = 1e-5        ### one person in N = 100001; stops the scaled score dividing by ~0
LEVELS = {0.5: (0.25, 0.75), 0.9: (0.1, 0.9), 0.95: (0.025, 0.975)}
OUT = os.path.join(fo.OUT, "conformal")
os.makedirs(OUT, exist_ok=True)


def score(y, yhat):
    return (y - yhat) / max(yhat, SCORE_FLOOR)


def invert(yhat, r):
    return yhat + r * max(yhat, SCORE_FLOOR)


def conformal_bounds(r, alpha):
    """As conformal_PINN.py: asymmetric quantiles if n allows, else symmetric |r|, else widest (flagged)."""
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
    return rs[0], rs[-1], "unguaranteed"


def main():
    ### median-over-seeds forecast for every scenario, setup and origin
    preds, tasks = {}, {}
    for group, label, task, beta_fit, learn, origin, seed, pred in fo.runs():
        preds.setdefault((group, label, learn, origin), []).append(pred["I_pred"].values)
        tasks[label] = task
    median = {k: np.median(np.stack(v), 0) for k, v in preds.items()}

    ### every (scenario, setup, origin, horizon) forecast with its score
    rows = []
    for (group, label, learn, origin), I_hat in median.items():
        obs = tasks[label]["I"]
        for h in range(1, len(obs) - origin):
            rows.append({"group": group, "scenario": label, "learned": learn, "origin": origin, "horizon": h,
                         "day": origin + h, "y": obs[origin + h], "yhat": I_hat[origin + h],
                         "score": score(obs[origin + h], I_hat[origin + h])})
    f = pd.DataFrame(rows)

    ### leave-one-scenario-out intervals
    out = []
    for (learn, h), g in f.groupby(["learned", "horizon"]):
        for label in g.scenario.unique():
            r = g[g.scenario != label].score.values
            for _, row in g[g.scenario == label].iterrows():
                rec = row.to_dict()
                rec["n_cal"] = len(r)
                if len(r) >= MIN_CAL:
                    for lvl, (ql, qh) in LEVELS.items():
                        lo, hi, form = conformal_bounds(r, 1 - lvl)
                        rec[f"q{ql}"] = max(invert(row.yhat, lo), 0.0)
                        rec[f"q{qh}"] = max(invert(row.yhat, hi), 0.0)
                        rec[f"form_{int(100 * lvl)}"] = form
                        rec[f"covered_{int(100 * lvl)}"] = rec[f"q{ql}"] <= row.y <= rec[f"q{qh}"]
                out.append(rec)
    iv = pd.DataFrame(out)
    for lvl in LEVELS:                                   ### True/False, NaN where there is no interval
        iv[f"covered_{int(100 * lvl)}"] = iv[f"covered_{int(100 * lvl)}"].astype(float)
    iv["sigma_gamma"] = np.where(iv.learned, "learned", "provided")
    iv.drop(columns="learned").to_csv(os.path.join(OUT, "conformal_intervals.csv"), index=False)

    ### empirical coverage by group, setup and horizon band (only horizons with intervals)
    scored = iv[iv.n_cal >= MIN_CAL].copy()
    scored["horizon_band"] = pd.cut(scored.horizon, [0, 5, 10, 20, 40], labels=["1-5", "6-10", "11-20", "21-40"])
    cov = scored.groupby(["sigma_gamma", "group", "horizon_band"], observed=True)[
        ["covered_50", "covered_90", "covered_95"]].mean().mul(100).round(1)
    cov["n"] = scored.groupby(["sigma_gamma", "group", "horizon_band"], observed=True).size()
    cov.to_csv(os.path.join(OUT, "conformal_coverage.csv"))
    print(cov.to_string())
    print(f"\ninterval form used (90%): {scored.form_90.value_counts().to_dict()}")
    print(f"horizons without intervals (n_cal < {MIN_CAL}): {sorted(iv[iv.n_cal < MIN_CAL].horizon.unique())}")


if __name__ == "__main__":
    main()
