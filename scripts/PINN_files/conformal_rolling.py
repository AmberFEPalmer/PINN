import glob
import os
import sys
import numpy as np
import pandas as pd
import forecast_origins as fo
import conformal_synthetic as cs

### Causal rolling-origin conformal intervals for the synthetic scenarios: the same procedure as
### conformal_PINN.py on the UKHSA data, applied to each synthetic scenario on its own.
### Fits (run_rolling.py): forecast origins every 2 days (day 30..94), one seed per origin, either sigma/gamma
### provided (results/PINN_simulated_data/rolling) or learned from the infection data only (rolling_learned). For origin t and horizon h, the calibration set is that scenario's
### own scores from earlier origins t' whose target is already observed (t' + h <= t), so nothing from the
### future is used. Scaled score, asymmetric conformal quantiles, MIN_CAL = 20. Two variants:
###   split - fixed level 1 - alpha
###   aci   - adaptive level (Gibbs & Candes 2021), updated only once the target day is observed
### Coverage and WIS are scored against the observations.
### Usage: python conformal_rolling.py [provided|learned]

SETUP = sys.argv[1] if len(sys.argv) > 1 else "provided"
LEARN = SETUP == "learned"
RUN_DIR = os.path.join(fo.RES, {"provided": "rolling", "learned": "rolling_learned"}[SETUP])
OUT = os.path.join(RUN_DIR, "conformal")
os.makedirs(OUT, exist_ok=True)
GAMMA_ACI = 0.02          ### ACI step size, as in conformal_PINN.py
MIN_CAL = cs.MIN_CAL
LEVELS = cs.LEVELS
PI_PAIRS = [(0.025, 0.975), (0.1, 0.9), (0.25, 0.75)]


def load_forecasts():
    """{(group, scenario): {origin: I_pred}} from the rolling runs."""
    out = {}
    for group, (scenarios, beta_fit, _) in fo.GROUPS.items():
        prefix = fo.pred_prefix(beta_fit, LEARN)
        for label in scenarios:
            for d in sorted(glob.glob(os.path.join(RUN_DIR, group, "origin_*"))):
                origin = int(d.rsplit("_", 1)[1])
                f = glob.glob(os.path.join(d, f"{prefix}_predictions_{fo.tag(label)}_*.csv"))
                if f:
                    out.setdefault((group, label), {})[origin] = pd.read_csv(f[0])["I_pred"].values
    return out


def wis(y, q):
    total = 0.5 * abs(y - q[0.5])
    for lo_q, hi_q in PI_PAIRS:
        alpha = 2 * lo_q
        l, u = q[lo_q], q[hi_q]
        s = u - l
        if y < l:
            s += (2 / alpha) * (l - y)
        if y > u:
            s += (2 / alpha) * (y - u)
        total += (alpha / 2) * s
    return total / (len(PI_PAIRS) + 0.5)


def intervals(group, label, preds, obs, adaptive):
    origins = sorted(preds)
    days = len(obs)
    alpha_t = {(h, lvl): 1 - lvl for h in range(1, days) for lvl in LEVELS}
    pending, rows = [], []
    for t in origins:
        ### ACI: apply each miss only once its target day has been observed
        ready = [p for p in pending if p[0] <= t]
        pending = [p for p in pending if p[0] > t]
        for _, key, err in ready:
            alpha_t[key] += GAMMA_ACI * ((1 - key[1]) - err)
        for h in range(1, days - t):
            r = np.array([cs.score(obs[t2 + h], preds[t2][t2 + h]) for t2 in origins if t2 + h <= t])
            if len(r) < MIN_CAL:
                continue
            yhat, y = preds[t][t + h], obs[t + h]
            q = {0.5: max(yhat, 0.0)}
            rec = {"group": group, "scenario": label, "origin": t, "horizon": h, "day": t + h, "y": y,
                   "yhat": yhat, "n_cal": len(r)}
            for lvl, (ql, qh) in LEVELS.items():
                a = float(np.clip(alpha_t[(h, lvl)] if adaptive else 1 - lvl, 1e-3, 0.999))
                lo, hi, form = cs.conformal_bounds(r, a)
                q[ql], q[qh] = max(cs.invert(yhat, lo), 0.0), max(cs.invert(yhat, hi), 0.0)
                covered = float(q[ql] <= y <= q[qh])
                rec.update({f"q{ql}": q[ql], f"q{qh}": q[qh], f"covered_{int(100 * lvl)}": covered,
                            f"form_{int(100 * lvl)}": form})
                if adaptive:
                    pending.append((t + h, (h, lvl), 1.0 - covered))
            rec["wis"] = wis(y, q) / max(np.mean(obs), cs.SCORE_FLOOR)      ### relative to the mean count
            rows.append(rec)
    return rows


def main():
    forecasts = load_forecasts()
    frames = []
    for variant, adaptive in (("split", False), ("aci", True)):
        rows = []
        for (group, label), preds in forecasts.items():
            rows += intervals(group, label, preds, fo.sc.load_task(label, fo.data_folder)["I"], adaptive)
        d = pd.DataFrame(rows)
        d.insert(0, "variant", variant)
        frames.append(d)
    iv = pd.concat(frames, ignore_index=True)
    iv.to_csv(os.path.join(OUT, "rolling_conformal_intervals.csv"), index=False)

    iv["horizon_band"] = pd.cut(iv.horizon, [0, 5, 10, 20], labels=["1-5", "6-10", "11-20"])
    keys = ["variant", "group", "horizon_band"]
    cov = iv.groupby(keys, observed=True)[["covered_50", "covered_90", "covered_95"]].mean().mul(100).round(1)
    ### mean WIS is the standard summary but a few non-converged single-seed fits dominate it, so the
    ### median is reported alongside; every fit is kept in both
    cov["rel_WIS_mean"] = iv.groupby(keys, observed=True).wis.mean().round(4)
    cov["rel_WIS_median"] = iv.groupby(keys, observed=True).wis.median().round(4)
    cov["n"] = iv.groupby(keys, observed=True).size()
    cov.to_csv(os.path.join(OUT, "rolling_conformal_coverage.csv"))
    print(cov.to_string())
    print(f"\norigins with intervals: {sorted(iv.origin.unique())}")
    print(f"interval form (aci, 90%): {iv[iv.variant == 'aci'].form_90.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
