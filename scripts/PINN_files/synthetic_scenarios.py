import os
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.integrate import solve_ivp

### SEIR epidemics with spline beta(t), for benchmarking the synthetic PINN

DAYS = 100
N_VAL = 100001
SIGMA = 0.25
GAMMA = 0.25
I0, E0, R0 = 1, 0, 0
S0 = N_VAL - I0
t_grid = np.linspace(0, DAYS, DAYS + 1)

K_CHOICES = (5, 7, 9, 13)
PSI_RANGE = (-1.7, -0.2)

NAMED = ["beta_spline", "beta_spline_slow_rise", "beta_spline_gradual_decline",
         "beta_spline_single_pulse", "beta_spline_two_waves", "beta_spline_three_waves",
         "beta_spline_rapid", "beta_spline_very_rapid", "beta_spline_escalating",
         "beta_spline_damped"]

_basis = {}


def basis(K):
    if K not in _basis:
        _basis[K] = np.asarray(dmatrix(f"cr(t, df={K}) - 1", {"t": t_grid}))
    return _basis[K]


def ode_model(t, y, beta_t):
    S, E, I, R = y
    beta = beta_t[min(int(np.floor(t)), len(t_grid) - 1)]
    return [-beta * S * I / N_VAL,
            beta * S * I / N_VAL - SIGMA * E,
            SIGMA * E - GAMMA * I,
            GAMMA * I]


def simulate(beta_t):
    sol = solve_ivp(ode_model, (t_grid[0], t_grid[-1]), [S0, E0, I0, R0],
                    args=(beta_t,), t_eval=t_grid)
    S, E, I, R = sol.y / N_VAL
    return {"t": t_grid / DAYS, "S": S, "E": E, "I": I, "R": R, "beta": beta_t}


def random_task(rng):
    K = int(rng.choice(K_CHOICES))
    psi = rng.uniform(*PSI_RANGE, size=K)
    task = simulate(np.exp(basis(K) @ psi))
    task.update({"K": K, "psi": psi})
    return task


def random_tasks(n, seed, min_peak_count=20):
    """n random epidemics. Draws whose infections never exceed min_peak_count people are
    re-drawn: with I0 = 1 and low beta the epidemic can die out, leaving no signal to fit."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        task = random_task(rng)
        if task["I"].max() * N_VAL >= min_peak_count:
            out.append(task)
    return out


def named_task(label, data_folder="../../data"):
    d = pd.read_csv(os.path.join(data_folder, f"{label}.csv"))
    task = {k: d[k].values for k in ("S", "E", "I", "R", "beta")}
    task["t"] = d["time"].values
    task["label"] = label
    return task


### Out-of-distribution test scenarios: beta(t) or observations the random spline family never
### produces. Spline draws keep beta in ~[0.18, 0.82] and are smooth and noise-free.
### exp_decay and piecewise follow scripts/synthetic_data_generation/SEIR_exponential_decay_beta.py
### and SEIR_beta_peicewise.py (change_day = 92 as in its code), evaluated on the daily grid used
### here; the noise scenarios are the report's SEIR_Gaussian_noise_*percent.csv (true beta = 0.75).
def _ood_beta(name):
    t = t_grid
    if name == "ood_exp_decay":
        return 0.8 + 0.2 * np.exp(-0.1 * t)                 ### starts above the spline range
    if name == "ood_constant_1.0":
        return np.full_like(t, 1.0)                         ### R0 = 4, above the range
    if name == "ood_piecewise":
        return np.where(t < 92, 0.75, 0.55)                 ### sharp step
    if name == "ood_lockdown_step":
        return np.where(t < 60, 0.8, 0.3)                   ### large sharp drop mid-epidemic
    raise KeyError(name)


OOD = ["ood_exp_decay", "ood_constant_1.0", "ood_piecewise", "ood_lockdown_step",
       "ood_noise_10", "ood_noise_20"]


def ood_task(name, data_folder="../../data"):
    if name.startswith("ood_noise_"):
        pct = name.split("_")[-1]
        d = pd.read_csv(os.path.join(data_folder, f"SEIR_Gaussian_noise_{pct}percent.csv"))
        task = {k: d[k].values for k in ("S", "E", "I", "R")}
        task["t"] = d["time"].values
        task["beta"] = np.full(len(task["t"]), 0.75)
    else:
        task = simulate(_ood_beta(name))
    task["label"] = name
    return task


### The report's constant-beta scenarios (Sec 3.1/3.3, Figs 5-6): data/SEIR_data_beta_{b}.csv,
### same ODE, I0 = 1, N = 100001 (verified identical to simulate() here)
CONSTANT = ["beta_0.75", "beta_0.5", "beta_0.4"]


def constant_task(label, data_folder="../../data"):
    b = float(label.split("_")[1])
    d = pd.read_csv(os.path.join(data_folder, f"SEIR_data_{label}.csv"))
    task = {k: d[k].values for k in ("S", "E", "I", "R")}
    task["t"] = d["time"].values
    task["beta"] = np.full(len(task["t"]), b)
    task["label"] = label
    return task


### The report's Gaussian observation noise scenarios (Sec 3.10-3.14): constant beta = 0.75 and
### I_noisy = I * (1 + level * N(0, 1)), multiplicative, with np.random.seed(42) and levels 1 - 20% 
NOISE = ["noise_5", "noise_10", "noise_15", "noise_20"]


def _report_noise_draws():
    rng = np.random.RandomState(42)
    return {pct: rng.normal(0, 1, len(t_grid)) for pct in range(1, 21)}


def _euler_I(beta=0.75):
    S, E, I, R = S0 * 1.0, E0 * 1.0, I0 * 1.0, R0 * 1.0
    out = [I]
    for _ in range(DAYS):
        dS = -beta * S * I / N_VAL; dE = beta * S * I / N_VAL - SIGMA * E
        dI = SIGMA * E - GAMMA * I; dR = GAMMA * I
        S += dS; E += dE; I += dI; R += dR
        out.append(I)
    return np.array(out) / N_VAL


def noise_task(label, euler=False):
    pct = int(label.split("_")[1])
    clean = _euler_I() if euler else simulate(np.full(len(t_grid), 0.75))["I"]
    task = simulate(np.full(len(t_grid), 0.75)) if not euler else {"t": t_grid / DAYS}
    task["I_clean"] = clean
    task["I"] = clean * (1 + pct / 100 * _report_noise_draws()[pct])
    task["beta"] = np.full(len(t_grid), 0.75)
    task["label"] = label
    return task


def load_task(name, data_folder="../../data"):
    """Any named scenario: spline (beta_spline*), constant (beta_0.75 etc.), OOD (ood_*), noise (noise_*)."""
    if name.startswith("noise_"):
        return noise_task(name)
    if name.startswith("ood_"):
        return ood_task(name, data_folder)
    if name in CONSTANT:
        return constant_task(name, data_folder)
    return named_task(name, data_folder)
