import glob
import os
import re
import numpy as np
import pandas as pd
import synthetic_scenarios as sc

### Forecast skill by forecast origin and horizon, from the multi-origin runs (5 seeds each).
### Origin = last training day (day 60/70/80/90 for split 0.6/0.7/0.8/0.9). For every origin:
###   MAE10_pct - MAE% over the first 10 days after the origin (a horizon every origin has)
###   AE_h{1,5,10}_pct - absolute % error at 1, 5 and 10 days ahead
###   MAE_full_pct - MAE% over the whole remaining window (its length differs between origins)
###   beta_MAPE_train_pct / beta_MAPE10_pct - beta(t) error over days 0..origin / the first 10 forecast days
### I(t) errors are against the observations; noise scenarios also against the noise-free curve (*_clean).
### Every seed is kept; summaries are median, min and max over seeds.

RES = "../../results/PINN_simulated_data"
OUT = os.path.join(RES, "origins")
data_folder = "../../data"
TIME_VARYING = ["beta_spline_slow_rise", "beta_spline_gradual_decline", "beta_spline_three_waves",
                "beta_spline_escalating", "beta_spline_rapid"]

### group -> (scenarios, beta fit, {split suffix: run directory}); day 80 for constant, noise and
### forecastable comes from the earlier 5-seed runs
GROUPS = {
    "constant": (sc.CONSTANT, "constant", {"60_40": OUT + "/constant", "70_30": OUT + "/constant",
                                           "80_20": RES + "/const_beta_seeds", "90_10": OUT + "/constant"}),
    "noise": (sc.NOISE, "constant", {"60_40": OUT + "/noise", "70_30": OUT + "/noise",
                                     "80_20": RES + "/const_beta_noise_seeds", "90_10": OUT + "/noise"}),
    "time_varying": (TIME_VARYING, "network", {s: OUT + "/time_varying" for s in ("60_40", "70_30", "80_20", "90_10")}),
    "forecastable": (sc.FORECASTABLE, "network", {"60_40": OUT + "/forecastable", "70_30": OUT + "/forecastable",
                                                  "80_20": RES + "/forecastable_seeds", "90_10": OUT + "/forecastable"}),
}


def pred_prefix(beta_fit, learn):
    if beta_fit == "constant":
        return "PINN_all_learnable_beta_const" if learn else "PINN_beta_const"
    return "PINN_all_learnable" if learn else "PINN"


def tag(label):
    """File tag used by run_scenarios: noise scenarios are Gaussian_noise_{p}percent."""
    return f"Gaussian_noise_{label.split('_')[1]}percent" if label.startswith("noise_") else label


def errors(I_pred, obs, origin):
    f = slice(origin + 1, None)
    p, o = I_pred[f], obs[f]
    out = {"MAE10_pct": 100 * np.mean(np.abs(p[:10] - o[:10])) / np.mean(o[:10]),
           "MAE_full_pct": 100 * np.mean(np.abs(p - o)) / np.mean(o)}
    for h in (1, 5, 10):
        out[f"AE_h{h}_pct"] = 100 * abs(p[h - 1] - o[h - 1]) / o[h - 1]
    return out


def beta_errors(beta_pred, beta_true, origin):
    """beta(t) error: MAPE over the training window (days 0..origin) and over the first 10 forecast days."""
    ape = 100 * np.abs(beta_pred - beta_true) / beta_true
    return {"beta_MAPE_train_pct": ape[:origin + 1].mean(), "beta_MAPE10_pct": ape[origin + 1:origin + 11].mean()}


def runs():
    """Every fit: (group, label, task, beta_fit, learn, origin, seed, predictions with I_pred and beta_pred)."""
    for group, (scenarios, beta_fit, dirs) in GROUPS.items():
        for label in scenarios:
            task = sc.load_task(label, data_folder)
            for learn in (False, True):
                for suffix, run_dir in dirs.items():
                    origin = int(round(int(suffix[:2]) / 100 * len(task["t"]))) - 1
                    files = sorted(glob.glob(os.path.join(
                        run_dir, f"{pred_prefix(beta_fit, learn)}_predictions_{tag(label)}_seed*_{suffix}.csv")))
                    for f in files:
                        yield (group, label, task, beta_fit, learn, origin,
                               int(re.search(r"_seed(\d+)_", f).group(1)), pd.read_csv(f))


def main():
    rows, horizon_rows = [], []
    for group, label, task, beta_fit, learn, origin, seed, pred in runs():
        I_pred, beta_pred = pred["I_pred"].values, pred["beta_pred"].values
        key = {"group": group, "scenario": label, "beta_fit": beta_fit,
               "sigma_gamma": "learned" if learn else "provided", "origin": origin, "seed": seed}
        row = dict(key, **errors(I_pred, task["I"], origin))
        if "I_clean" in task:
            row.update({f"{k}_clean": v for k, v in errors(I_pred, task["I_clean"], origin).items()})
        row.update(beta_errors(beta_pred, task["beta"], origin))
        rows.append(row)
        for h in range(1, 11):
            o = task["I"][origin + h]
            b = task["beta"][origin + h]
            horizon_rows.append(dict(key, horizon=h, AE_pct=100 * abs(I_pred[origin + h] - o) / o,
                                     AE_beta_pct=100 * abs(beta_pred[origin + h] - b) / b))

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(OUT, "forecast_by_origin_per_seed.csv"), index=False)
    pd.DataFrame(horizon_rows).to_csv(os.path.join(OUT, "forecast_by_horizon_per_seed.csv"), index=False)

    metrics = [c for c in per_seed.columns if c.endswith("_pct") or c.endswith("_pct_clean")]
    keys = ["group", "scenario", "beta_fit", "sigma_gamma", "origin"]
    summary = per_seed.groupby(keys, sort=False)[metrics].agg(["median", "min", "max"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_seeds", per_seed.groupby(keys, sort=False).size())
    summary.reset_index().to_csv(os.path.join(OUT, "forecast_by_origin.csv"), index=False)
    print(f"{len(per_seed)} fits -> {OUT}/forecast_by_origin.csv")


if __name__ == "__main__":
    main()
