import os
import sys
import PINN_simulated_data as p
import synthetic_scenarios as sc

### Rolling-origin fits for causal conformal (conformal_rolling.py): origins every 2 days (day 30..94), 1 seed.
### Origin d = last training day; split = (d + 1) / 101 gives n_train = d + 1.
### Method unchanged: constant beta for constant/noise scenarios, beta(t) network for the time-varying ones.
### Usage: python run_rolling.py WORKER N_WORKERS [provided|learned]
###   provided - sigma/gamma provided -> results/PINN_simulated_data/rolling
###   learned  - sigma/gamma learned (infection data only) -> results/PINN_simulated_data/rolling_learned

RES = "../../results/PINN_simulated_data"
TV = ["beta_spline_slow_rise", "beta_spline_gradual_decline", "beta_spline_three_waves",
      "beta_spline_escalating", "beta_spline_rapid"]
GROUPS = {"constant": (sc.CONSTANT, True), "noise": (sc.NOISE, True),
          "time_varying": (TV, False), "forecastable": (sc.FORECASTABLE, False)}
ORIGINS = list(range(30, 95, 2))
RUN_DIRS = {"provided": os.path.join(RES, "rolling"), "learned": os.path.join(RES, "rolling_learned")}


def prefixes(beta_const, learn):
    """(metrics/plot prefix, prediction prefix), as in the multi-origin runs."""
    if beta_const:
        return (("PINN_all_learnable_parameters_beta_const", "PINN_all_learnable_beta_const") if learn
                else ("PINN_beta_const", "PINN_beta_const"))
    return ("PINN_all_learnable_parameters", "PINN_all_learnable") if learn else ("PINN", "PINN")


def main():
    worker, n_workers = int(sys.argv[1]), int(sys.argv[2])
    setup = sys.argv[3] if len(sys.argv) > 3 else "provided"
    learn = setup == "learned"
    out = RUN_DIRS[setup]

    ### greedy balance by number of fits, largest first
    jobs = [(g, scen, bconst, d) for g, (scen, bconst) in GROUPS.items() for d in ORIGINS]
    load, mine = [0] * n_workers, []
    for j in sorted(jobs, key=lambda j: -len(j[1])):
        k = load.index(min(load))
        load[k] += len(j[1])
        if k == worker:
            mine.append(j)
    print(f"worker {worker} ({setup}): {len(mine)} jobs, {sum(len(j[1]) for j in mine)} fits", flush=True)

    for group, scenarios, bconst, d in mine:
        run_dir = f"{out}/{group}/origin_{d:02d}"
        prefix, pred_prefix = prefixes(bconst, learn)
        split = (d + 1) / 101
        done = [s for s in scenarios if os.path.exists(os.path.join(
            run_dir, f"{pred_prefix}_predictions_{p.report_label(s)}_{p.split_suffix(split, learn)}.csv"))]
        todo = [s for s in scenarios if s not in done]
        print(f"\n### job {group} origin {d} ({len(done)} already done)", flush=True)
        if todo:
            p.run_scenarios(todo, split=split, seeds=1, learn_sg=learn, beta_const=bconst, prefix=prefix,
                            pred_prefix=pred_prefix, data_folder="../../data", output_dir=run_dir)
    print(f"worker {worker} done", flush=True)


if __name__ == "__main__":
    main()
