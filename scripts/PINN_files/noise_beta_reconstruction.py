import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import synthetic_scenarios as st

### beta reconstruction under multiplicative Gaussian observation noise (5-20%), constant beta = 0.75.
### The noise draws are the report's (seed 42), applied to the accurate solve_ivp curve so that 0.75
### is the correct answer (the report's own noise files use a forward-Euler curve; see
### synthetic_scenarios.noise_task). 3 seeds per noise level.

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14, 'legend.fontsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'lines.linewidth': 2,
    'axes.grid': True, 'grid.alpha': 0.3,
})
RES = "../../results/PINN_simulated_data/noise"
rows, preds = [], []
for f in sorted(glob.glob(f"{RES}/*.pkl")):
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    rows += d["rows"]; preds += d["preds"]
df = pd.DataFrame(rows)
df = df[(df.budget == 50000) & df.lbfgs.astype(bool)]
df["level"] = df.task.str.split("_").str[1].astype(int)

for metric, label in (("beta_mape", "beta MAPE, whole series"),
                      ("beta_mape_signal", "beta MAPE, I >= 5% of peak"),
                      ("beta_mape_lowsignal", "beta MAPE, I < 5% of peak"),
                      ("I_mae_pct_clean", "I(t) error vs the noise-free curve, %")):
    g = df.groupby("level")[metric]
    print(f"\n=== {label} (median [min-max] over 3 seeds) ===")
    for lvl in sorted(df.level.unique()):
        print(f"  noise {lvl:2d}%  {g.median()[lvl]:.3f} [{g.min()[lvl]:.3f}-{g.max()[lvl]:.3f}]")

P = pd.DataFrame(preds)
t_days = np.arange(101)
fig, axes = plt.subplots(2, len(st.NOISE), figsize=(5 * len(st.NOISE), 8.5))
for j, name in enumerate(st.NOISE):
    task = st.noise_task(name)
    for k, (_, r) in enumerate(P[P.task == name].iterrows()):
        axes[0, j].plot(t_days, r["beta"], color="#d62728", lw=1.5, alpha=0.85, label="β(t) estimated" if k == 0 else None)
        axes[1, j].plot(t_days, r["I"] * st.N_VAL, color="#d62728", lw=1.5, alpha=0.85, label="PINN" if k == 0 else None)
    axes[0, j].axhline(0.75, color="#1f33b4", ls="--", lw=1.5, label="true β = 0.75")
    axes[0, j].set_ylim(0, 1.6); axes[0, j].set_title(f"{name.split('_')[1]}% noise: β(t)", fontsize=12)
    axes[1, j].scatter(t_days, task["I"] * st.N_VAL, s=8, color="#1f33b4", alpha=0.5, label="noisy data")
    axes[1, j].plot(t_days, task["I_clean"] * st.N_VAL, color="k", ls=":", lw=1.2, label="noise-free")
    axes[1, j].set_title(f"{name.split('_')[1]}% noise: I(t)", fontsize=12)
    axes[1, j].set_xlabel("Time (days)")
axes[0, 0].set_ylabel(r"$\beta(t)$"); axes[1, 0].set_ylabel("Infected individuals")
axes[0, 0].legend(fontsize=9); axes[1, 0].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{RES}/beta_noise.png", dpi=130)
print(f"\nwrote {RES}/beta_noise.png")
