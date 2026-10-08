import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import synthetic_scenarios as st

### beta(t) reconstruction in the report's Fig 10 setting: time-varying beta, sigma and gamma provided,
### all 100 days of I(t) observed. Five scenarios, 3 seeds each.

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14, 'legend.fontsize': 11,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'lines.linewidth': 2,
    'axes.grid': True, 'grid.alpha': 0.3,
})
RES = "../../results/PINN_simulated_data"
SCEN = [("beta_spline_slow_rise", "Slow Rise"), ("beta_spline_gradual_decline", "Gradual Decline"),
        ("beta_spline_three_waves", "Three Waves"), ("beta_spline_escalating", "Escalating"),
        ("beta_spline_rapid", "Rapid")]

preds = []
for f in sorted(glob.glob(f"{RES}/reconstruction/*.pkl")):
    with open(f, "rb") as fh:
        preds += [p for p in pickle.load(fh)["preds"] if p["mode"] == "reconstruct" and p.get("arm", "cold") == "cold"]
P = pd.DataFrame(preds)

t = np.arange(101)
fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
for k, (ax, (name, title)) in enumerate(zip(axes.ravel(), SCEN)):
    task = st.named_task(name, "../../data")
    ax.scatter(t, task["beta"], s=12, color="#1f33b4", alpha=0.5, label=r"$\beta(t)$ true", zorder=3)
    sub = P[P.task == name]
    err = np.median([np.mean(np.abs(b - task["beta"]) / task["beta"]) for b in sub["beta"]])
    for j, (_, r) in enumerate(sub.iterrows()):
        ax.plot(t, r["beta"], color="#d62728", lw=1.5, alpha=0.9, label=r"$\beta(t)$ estimated" if j == 0 else None)
    ax.set_title(f"({chr(97 + k)}) {title}")
    ax.text(0.02, 0.03, f"median error {100 * err:.1f}%", transform=ax.transAxes, fontsize=10)
    ax.set_xlabel("Time (days)"); ax.set_ylabel(r"$\beta(t)$")
    ax.set_ylim(0, max(1.0, 1.3 * task["beta"].max()))
axes[0, 0].legend(fontsize=10)
axes[1, 2].axis("off")
fig.suptitle("β(t) reconstruction, σ and γ provided, 3 seeds each", fontsize=14)
plt.tight_layout(); plt.savefig(f"{RES}/fig10_beta_reconstruction.png", dpi=140)
print(f"wrote {RES}/fig10_beta_reconstruction.png")
