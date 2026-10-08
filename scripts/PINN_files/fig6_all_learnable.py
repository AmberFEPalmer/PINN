import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import synthetic_scenarios as st

### The report's Fig 6 setting (constant beta, PINN given infection data only): sigma and gamma are two
### learnable constants started at 0.5 (true 0.25). Two versions of beta:
###   learn_sg   - beta(t) a network of time: non-identifiable, a free beta(t) absorbs a wrong sigma, gamma
###   beta_const - beta one learnable constant: identifiable

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14, 'legend.fontsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'lines.linewidth': 2,
    'axes.grid': True, 'grid.alpha': 0.3,
})
RES = "../../results/PINN_simulated_data/fig6"
VARIANTS = {"learn_sg": "β(t) network + learnable σ, γ (non-identifiable)",
            "beta_const": "constant β + learnable σ, γ (identifiable)"}


def load(prefix):
    rows, preds = [], []
    for f in sorted(glob.glob(f"{RES}/{prefix}_*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        rows += d["rows"]; preds += d["preds"]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[(df.budget == 50000) & df.lbfgs.astype(bool)]
    return df, pd.DataFrame(preds)


data = {v: load(v) for v in VARIANTS}
for v, (df, _) in data.items():
    if df.empty:
        continue
    print(f"\n=== {VARIANTS[v]} - true sigma = gamma = 0.25 ===")
    for task, g in df.groupby("task"):
        for _, r in g.sort_values("seed").iterrows():
            print(f"  {task:10s} seed {r.seed}: beta MAPE {100*r.beta_mape:6.2f}% | sigma {r.sigma:.4f} | gamma {r.gamma:.4f}")
        print(f"  {task:10s} MEDIAN: beta MAPE {100*g.beta_mape.median():.2f}%, sigma {g.sigma.median():.4f}, gamma {g.gamma.median():.4f}")

t_days = np.arange(101)
present = [v for v in VARIANTS if not data[v][0].empty]
fig, axes = plt.subplots(len(present), 3, figsize=(16, 4.6 * len(present)), squeeze=False)
for i, v in enumerate(present):
    P = data[v][1]
    for ax, name in zip(axes[i], st.CONSTANT):
        task = st.constant_task(name, "../../data")
        ax.scatter(t_days, task["beta"], s=10, color="#1f33b4", alpha=0.5, label="β true")
        for _, r in P[P.task == name].sort_values("seed").iterrows():
            ax.plot(t_days, r["beta"], color="#d62728", lw=1.6, alpha=0.85,
                    label=f"seed {r.seed}: σ={r.sigma:.3f}, γ={r.gamma:.3f}")
        ax.set_ylim(0, 1.6); ax.set_title(f"β = {name.split('_')[1]}  |  {VARIANTS[v]}", fontsize=10.5)
        ax.set_xlabel("Time (days)"); ax.legend(fontsize=8.5, loc="upper right")
    axes[i, 0].set_ylabel(r"$\beta(t)$")
fig.suptitle("Infection data only (report Fig 6 setting): σ and γ learned, true σ = γ = 0.25, started at 0.5",
             fontsize=13)
plt.tight_layout(); plt.savefig(f"{RES}/fig6_all_learnable.png", dpi=140)
print(f"\nwrote {RES}/fig6_all_learnable.png")
