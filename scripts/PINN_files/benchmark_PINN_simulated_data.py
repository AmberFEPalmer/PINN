import argparse
import pickle
import time
import numpy as np
import tensorflow as tf

import PINN_simulated_data as pc
import synthetic_scenarios as st

### Fits the SEIR-PINN to synthetic scenarios and scores it:
###   forecast    - train on days 0-80, score I(t) MAE% on days 81-100
###   reconstruct - train on all 101 days, score beta(t) (and sigma, gamma if learned) against the truth
### Test tasks: the 10 named spline scenarios (data/beta_spline*.csv), random spline epidemics,
### out-of-distribution scenarios (st.OOD), the constant-beta scenarios and the noise scenarios. The fixed
### scenarios are fitted from several random seeds, since single-run PINN results vary several-fold across
### seeds. Each Adam budget is scored on the best weights so far; the final result adds L-BFGS.

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="../../results/PINN_simulated_data/benchmark_shard0.pkl")
ap.add_argument("--n-random", type=int, default=20)
ap.add_argument("--random-seed", type=int, default=999)
ap.add_argument("--seeds", type=int, default=3, help="random seeds per fixed scenario")
ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 5000, 20000, 50000])
ap.add_argument("--lbfgs", type=int, default=5000, help="L-BFGS iterations after the last Adam budget")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--learn-sg", action="store_true", help="sigma and gamma are learnable constants, started at 0.5")
ap.add_argument("--beta-const", action="store_true", help="beta is one learnable constant (started at 1.0), not a network")
ap.add_argument("--kinds", nargs="+", default=["named", "ood", "random"])
ap.add_argument("--modes", nargs="+", default=["forecast", "reconstruct"])
args = ap.parse_args()

SPLIT = 81                                         ### days 0-80 train, 81-100 test

tasks = [dict(st.named_task(l), name=l, kind="named") for l in st.NAMED]
tasks += [dict(st.ood_task(l), name=l, kind="ood") for l in st.OOD]
tasks += [dict(st.constant_task(l), name=l, kind="constant") for l in st.CONSTANT]
tasks += [dict(st.noise_task(l), name=l, kind="noise") for l in st.NOISE]
tasks += [dict(t, name=f"random_{i:02d}", kind="random")
          for i, t in enumerate(st.random_tasks(args.n_random, args.random_seed))]

jobs = []
for i, task in enumerate(tasks):
    if task["kind"] not in args.kinds:
        continue
    n_seeds = 1 if task["kind"] == "random" else args.seeds
    for mode in args.modes:
        jobs += [(i, mode, s) for s in range(n_seeds)]
jobs = jobs[args.shard::args.n_shards]
print(f"{len(tasks)} tasks, {len(jobs)} fits in shard {args.shard}/{args.n_shards}", flush=True)


def mae_pct(pred, obs):
    return float(100 * np.mean(np.abs(pred - obs)) / (np.mean(obs) + 1e-12))


def score(task, mode, out):
    I, b = task["I"], task["beta"]
    ape = np.abs(out["beta"] - b) / b
    ### beta only enters the dynamics through beta*S*I/N; report its error separately where infections
    ### are substantial (I >= 5% of peak) and in the low-incidence period
    signal = I >= 0.05 * I.max()
    row = {"I_mae_pct_full": mae_pct(out["I"], I),
           ### against the noise-free curve, where there is one (noise scenarios)
           "I_mae_pct_clean": mae_pct(out["I"], task["I_clean"]) if "I_clean" in task else np.nan,
           "beta_mape": float(np.mean(ape)),
           "beta_mae": float(np.mean(np.abs(out["beta"] - b))),
           "sigma": out["sigma"], "gamma": out["gamma"],
           "beta_mape_signal": float(np.mean(ape[signal])),
           "beta_mape_lowsignal": float(np.mean(ape[~signal])) if (~signal).any() else np.nan}
    if mode == "forecast":
        row["I_mae_pct_test"] = mae_pct(out["I"][SPLIT:], I[SPLIT:])
        row["beta_mape_train"] = float(np.mean(np.abs(out["beta"][:SPLIT] - b[:SPLIT]) / b[:SPLIT]))
    return row


fit = pc.Fitter(learn_sg=args.learn_sg, beta_const=args.beta_const)
rows, preds = [], []
t0 = time.time()
for j, (i, mode, seed) in enumerate(jobs):
    task = tasks[i]
    n = SPLIT if mode == "forecast" else len(task["t"])
    fit.set_task(task["t"][:n], task["I"][:n])
    tf.random.set_seed(10_000 * seed + 17 * i + (mode == "forecast"))
    fit.reset(pc.create_pinn_model().get_weights())

    done, best = 0, None
    for budget in sorted(args.budgets):
        best = fit.run_keep_best(budget - done, best)
        done = budget
        current = fit.get_state()
        fit.set_state(best[1])
        out = fit.predict(task["t"])
        rows.append({"task": task["name"], "kind": task["kind"], "mode": mode, "lbfgs": False,
                     "seed": seed, "budget": budget, "loss": best[0], **score(task, mode, out)})
        if budget != max(args.budgets):
            fit.set_state(current)               ### keep training Adam from where it was
    loss = best[0]
    if args.lbfgs:                               ### polish the best weights
        _, loss = fit.lbfgs(args.lbfgs)
        out = fit.predict(task["t"])
        rows.append({"task": task["name"], "kind": task["kind"], "mode": mode, "lbfgs": True,
                     "seed": seed, "budget": budget, "loss": loss, **score(task, mode, out)})
    preds.append({"task": task["name"], "mode": mode, "seed": seed,
                  "I": out["I"], "beta": out["beta"], "sigma": out["sigma"], "gamma": out["gamma"]})

    r = rows[-1]
    extra = f"I test MAE% {r['I_mae_pct_test']:.2f}" if mode == "forecast" else ""
    print(f"[{j + 1}/{len(jobs)}] {task['name']:28s} {mode:11s} s{seed} | loss {loss:.2e} | "
          f"beta MAPE {r['beta_mape']:.3f} | sigma {r['sigma']:.3f} gamma {r['gamma']:.3f} | {extra} | "
          f"{time.time() - t0:.0f}s", flush=True)
    with open(args.out, "wb") as f:
        pickle.dump({"rows": rows, "preds": preds, "config": vars(args)}, f)

print(f"saved {len(rows)} rows to {args.out}")
