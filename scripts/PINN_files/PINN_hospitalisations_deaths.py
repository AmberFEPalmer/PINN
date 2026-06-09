import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import regularizers
import pandas as pd
import os
import matplotlib.dates as mdates

### 7 compartment ODE structure
### State variables - susceptible, exposed, infectious, recovered, reported cases, hospital admissions, hospital occupancy , reported deaths
### 2 sub-networks - one for state variables, seperate one for beta
### data fitted to fluxes of reporting compartments - this varied from my implementation with data fitted directly to compartments
### i.e. data fitted to reported cases per week, admissions fitted to admissions per week
### model parameters copied from Qian et al. 2025 table 2

### Plot style — fixed across all scripts
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLOURS = {
    "S": "#2ca02c",
    "E": "#ff7f0e",
    "I": "#d62728",
    "R": "#1f33b4",
}

### set seed for reproducibility 
np.random.seed(42)
tf.random.set_seed(42)

### Population size for England – used for normalisation and unnormalisation
### https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/bulletins/annualmidyearpopulationestimates/mid2021
N_val = 56_000_000               
### Make N a tensorflow constant so it can be used in the loss function
### Python floats can't be used in the @tensorflow graph function, without which the model is too slow          
N = tf.constant(float(N_val), dtype=tf.float32)
### N^2 saved here for scale normalisation
### SEIR compartments in absolute counts so losses are on O(N^2) scale, but want them on O(1) scale for stability
N_sq = N ** 2   

### get the preprocessed data from csv files
t_data_norm = np.load("../../data/t_data_study.npy").reshape(-1, 1)
I_data = np.load("../../data/I_data_study.npy").reshape(-1, 1)
H_data = np.load("../../data/H_data_study.npy").reshape(-1, 1)
D_data = np.load("../../data/D_data_study.npy").reshape(-1, 1)

END_DATE = None

_dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")[:len(t_data_norm)]
if END_DATE is not None:
    keep = _dates <= np.datetime64(END_DATE)
    T_cap = int(keep.sum())
else:
    T_cap = len(t_data_norm)

### cap data so each data source is the same length
t_data_norm = t_data_norm[:T_cap]
I_data = I_data[:T_cap]
H_data = H_data[:T_cap]
D_data = D_data[:T_cap]
assert len(I_data) == len(H_data) == len(D_data) == len(t_data_norm), \
    "series length mismatch - re-run data processing (cases, then deaths/hosp)"

### Print total study period
N_total_points = len(t_data_norm)
print(f"study grid: {N_total_points} weeks "
      f"({_dates[0]} -> {_dates[T_cap - 1]})"
      f"{'  [clipped]' if END_DATE else '  [full period]'}")

### scale infection, hospitalisation + death data
I_scale = float(I_data.max())
H_scale = float(H_data.max())
D_scale = float(D_data.max())
print(f"scales  cases={I_scale:.3e}  hosp={H_scale:.3e}  deaths={D_scale:.3e}")

try:
    study_dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")[:N_total_points]
    if len(study_dates) < N_total_points:
        raise ValueError("dates file shorter than data")
except (FileNotFoundError, ValueError):
    print("dates_study.npy not found/short - reconstructing weekly date grid.")
    study_dates = (pd.date_range("2020-08-01", periods=N_total_points, freq="W-SAT")
                   .values.astype("datetime64[D]"))

def apply_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right')

### fixed parameters from Qian et al. 2025 table 2
### scaled to weekly time units (multiply by 7)
ETA = 0.25 * 7 ### latent to infectious
GAMMA = 0.25 * 7 ### infectious to recovery
GAMMA_ZW = 1.0 * 7 ### case reporting
GAMMA_H = 0.1 * 7 ### leaving hospital
GAMMA_A = 1.0  * 7 ### hospital admissisons reporting
GAMMA_DW = 0.1 * 7 ### death reporting
RHO = 0.5

### Define PINN
### L2 regularisation for hidden layers -> helps to prevent overfitting
### Add penalty proportional to the sum of squared coefficients to the loss
### Reduce model complexity, penalise large weights
### https://keras.io/api/layers/regularizers/
def create_pinn_model():
    t_input = Input(shape=(1,), name='time_input')
    
    ### hidden layers compartmental model, 3, 50 neurons
    ### tanh activation for nonlinearity, L2 regularisation to prevent overfitting
    core = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    core = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(core)
    core = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(core)
    
    ### 7 output layers
    ### latent, infectious, removed, case-box, hospitalisations, admissions, deaths
    ### softplus activation to ensure non-negativity of compartments and fluxes
    L = Dense(1, activation='softplus', name='L')(core)
    Y = Dense(1, activation='softplus', name='Y')(core)
    R = Dense(1, activation='softplus', name='R')(core)
    Z = Dense(1, activation='softplus', name='Z')(core)
    H = Dense(1, activation='softplus', name='H')(core)
    A = Dense(1, activation='softplus', name='A')(core)
    D = Dense(1, activation='softplus', name='D')(core)
    X = Lambda(lambda z: 1.0 - z[0] - z[1] - z[2], name='X')([L, Y, R])
    
    ### hidden beta subnetwork, 3 layers, 50 neurons, tanh activation, L2 regularisation
    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_h)
    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_h)
    
    ### beta output
    ### no activation function
    ### in log form
    log_beta = Dense(1, activation=None, name='log_beta')(beta_h)
    log_beta = Lambda(lambda x: tf.clip_by_value(x, -3.0, 1.5), name='clipped_log_beta')(log_beta)
    beta = Lambda(lambda x: tf.exp(x), name='beta')(log_beta)
    
    return Model(inputs=t_input, outputs=[X, L, Y, R, Z, H, A, D, beta])

### define loss function
def compute_loss(t_col, t_data, I_d, H_d, D_d, net, logit_p_h, logit_p_d,
                 total_weeks, L0, R0):
    
    ### convert to column
    if len(t_col.shape) == 1:
        t_col = tf.reshape(t_col, (-1, 1))
    t_data = tf.cast(tf.reshape(t_data, (-1, 1)), tf.float32)
    I_d = tf.cast(tf.reshape(I_d, (-1, 1)), tf.float32)
    H_d = tf.cast(tf.reshape(H_d, (-1, 1)), tf.float32)
    D_d = tf.cast(tf.reshape(D_d, (-1, 1)), tf.float32)
    
    ### model parameters
    eta = tf.constant(ETA, dtype=tf.float32)
    gamma = tf.constant(GAMMA, dtype=tf.float32)
    gamma_zw = tf.constant(GAMMA_ZW, dtype=tf.float32)
    gamma_h = tf.constant(GAMMA_H, dtype=tf.float32)
    gamma_a = tf.constant(GAMMA_A, dtype=tf.float32)
    gamma_dw = tf.constant(GAMMA_DW, dtype=tf.float32)
    rho = tf.constant(RHO, dtype=tf.float32)
    p_h = tf.sigmoid(logit_p_h)
    p_d = tf.sigmoid(logit_p_d)
    asc = p_h + rho * (1.0 - p_h)
    
    T = tf.cast(total_weeks, tf.float32)
    
    ### https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        X, L, Y, R, Z, H, A, D, beta = net(t_col)
        
    dL_dt = tape.gradient(L, t_col)
    dY_dt = tape.gradient(Y, t_col)
    dR_dt = tape.gradient(R, t_col)
    dZ_dt = tape.gradient(Z, t_col)
    dH_dt = tape.gradient(H, t_col)
    dA_dt = tape.gradient(A, t_col)
    dD_dt = tape.gradient(D, t_col)
    del tape
    
    ### ODE compartmental model
    X_abs, L_abs, Y_abs = X * N, L * N, Y * N
    R_abs, Z_abs, H_abs, A_abs, D_abs = R * N, Z * N, H * N, A * N, D * N
    dL_phys = T * (beta * X_abs * Y_abs / N - eta * L_abs)
    dY_phys = T * (eta * L_abs - gamma * Y_abs)
    dR_phys = T * (gamma * Y_abs)
    dZ_phys = T * (asc * gamma * Y_abs - gamma_zw * Z_abs)
    dH_phys = T * (p_h * gamma * Y_abs - gamma_h * H_abs)
    dA_phys = T * (p_h * gamma * Y_abs - gamma_a * A_abs)
    dD_phys = T * (p_d * gamma_h * H_abs - gamma_dw * D_abs)
    
    ode_loss = (
        tf.reduce_mean(tf.square((dL_dt * N) - dL_phys)) +
        tf.reduce_mean(tf.square((dY_dt * N) - dY_phys)) +
        tf.reduce_mean(tf.square((dR_dt * N) - dR_phys)) +
        tf.reduce_mean(tf.square((dZ_dt * N) - dZ_phys)) +
        tf.reduce_mean(tf.square((dH_dt * N) - dH_phys)) +
        tf.reduce_mean(tf.square((dA_dt * N) - dA_phys)) +
        tf.reduce_mean(tf.square((dD_dt * N) - dD_phys))
    ) / N_sq
    
    Y0 = eta * L0 / gamma
    Z0 = asc * gamma * Y0 / gamma_zw
    H0 = p_h * gamma * Y0 / gamma_h
    A0 = p_h * gamma * Y0 / gamma_a
    D0 = p_d * gamma_h * H0 / gamma_dw
    
    t_zero = tf.constant([[0.0]], dtype=tf.float32)
    _, L_0, Y_0, R_0, Z_0, H_0, A_0, D_0, _ = net(t_zero)
    ic_loss = tf.reduce_mean(
        tf.square(L_0 - L0) + tf.square(Y_0 - Y0) + tf.square(R_0 - R0) +
        tf.square(Z_0 - Z0) + tf.square(H_0 - H0) +
        tf.square(A_0 - A0) + tf.square(D_0 - D0)
    )
    
    _, _, Y_p, _, Z_p, _, A_p, D_p, _ = net(t_data)
    case_p = gamma_zw * Z_p
    adm_p = gamma_a * A_p
    death_p = gamma_dw * D_p
    
    ### data loss
    case_loss = tf.reduce_mean(tf.square((case_p - I_d) / I_scale))
    hosp_loss = tf.reduce_mean(tf.square((adm_p - H_d) / H_scale))
    death_loss = tf.reduce_mean(tf.square((death_p - D_d) / D_scale))
    data_loss = case_loss + hosp_loss + death_loss
    total = 1.0 * data_loss + 0.1 * ode_loss + 0.1 * ic_loss
    return total, {"data_loss": data_loss, "case": case_loss, "hosp": hosp_loss,
                   "death": death_loss, "IC_loss": ic_loss, "ODE_loss": ode_loss}

### define single window training
def train_window(t_train_norm, I_tr, H_tr, D_tr, total_weeks, inc0, R0,
                 n_iter=50_000, warm=None):
    model = create_pinn_model()
    logit_p_h = tf.Variable(-3.0, dtype=tf.float32) ### learnable parameter - probability infection leads to hospitalisation
    logit_p_d = tf.Variable(-1.0, dtype=tf.float32) ### learnable parameter - probability hospitalisation leds to death
    asc0 = 0.5 + 0.047 * 0.5 ### one off estimate of ascertainment 
    L0_seed = max(inc0 / (asc0 * GAMMA) * (GAMMA / ETA), 1e-6)
    log_L0 = tf.Variable(np.log(L0_seed), dtype=tf.float32)
    
    ### warm start model
    ### https://arxiv.org/abs/1910.08475
    if warm is not None:
        model.set_weights(warm[0])
        logit_p_h.assign(warm[1]); logit_p_d.assign(warm[2]); log_L0.assign(warm[3])
        
    ### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
    optm = Adam(learning_rate=0.001)
    t_col = tf.convert_to_tensor(np.linspace(0, 1.0, 500).reshape(-1, 1), dtype=tf.float32)
    t_tr = tf.convert_to_tensor(t_train_norm, dtype=tf.float32)
    I_t = tf.convert_to_tensor(I_tr, dtype=tf.float32)
    H_t = tf.convert_to_tensor(H_tr, dtype=tf.float32)
    D_t = tf.convert_to_tensor(D_tr, dtype=tf.float32)
    
    ### @tf.function makes faster
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            L0 = tf.exp(log_L0)
            loss, ld = compute_loss(t_col, t_tr, I_t, H_t, D_t, model,
                                    logit_p_h, logit_p_d, total_weeks, L0, R0)
        vlist = model.trainable_variables + [logit_p_h, logit_p_d, log_L0]
        grads = tape.gradient(loss, vlist)
        optm.apply_gradients(zip(grads, vlist))
        return loss, ld
    best, min_loss = None, np.inf
    
    #### printing while model runs
    for itr in range(n_iter):
        loss, ld = step()
        if itr % 1000 == 0:
            lv = float(loss)
            if lv < min_loss:
                min_loss = lv
                best = (model.get_weights(), float(logit_p_h.numpy()),
                        float(logit_p_d.numpy()), float(log_L0.numpy()))
            if itr % 5000 == 0:
                print(f"  iter {itr:5d} | total {lv:.2e} | "
                      f"case {float(ld['case']):.2e} hosp {float(ld['hosp']):.2e} "
                      f"death {float(ld['death']):.2e} | ODE {float(ld['ODE_loss']):.2e} | "
                      f"IC {float(ld['IC_loss']):.2e} | p_h {float(tf.sigmoid(logit_p_h)):.4f} "
                      f"p_d {float(tf.sigmoid(logit_p_d)):.4f} L0 {float(tf.exp(log_L0)):.2e}")
    if best is not None:
        model.set_weights(best[0])
        logit_p_h.assign(best[1]); logit_p_d.assign(best[2]); log_L0.assign(best[3])
    return model, logit_p_h, logit_p_d, log_L0

First_train_weeks = 17
Forecast_horizon = 4
SERIES = ["cases", "hosp", "deaths"]
DATA = {"cases": I_data, "hosp": H_data, "deaths": D_data}

inc0 = float(I_data[0, 0])
R0 = 0.06

all_pred = {s: {} for s in SERIES}
all_naive = {s: {} for s in SERIES}
all_obs = {s: {} for s in SERIES}
all_beta = {}

warm = None
for train_end in range(First_train_weeks, N_total_points - Forecast_horizon + 1):
    total_weeks = float(N_total_points - 1)
    n_iter = 50_000
    print(f"Train weeks 1-{train_end} | forecast {train_end+1}-{train_end+Forecast_horizon} "
          f"| {'warm' if warm else 'cold'} start")

    model, lph, lpd, lL0 = train_window(
        t_data_norm[:train_end], I_data[:train_end], H_data[:train_end], D_data[:train_end],
        total_weeks=total_weeks, inc0=inc0, R0=R0, n_iter=n_iter, warm=warm)
    warm = (model.get_weights(), float(lph.numpy()), float(lpd.numpy()), float(lL0.numpy()))

    p_h = float(tf.sigmoid(lph)); p_d = float(tf.sigmoid(lpd))

    origin = train_end - 1
    if train_end == First_train_weeks:
        d_last = pd.Timestamp(study_dates[origin])
        d_h1 = pd.Timestamp(study_dates[origin + 1])
        gap_days = (d_h1 - d_last).days
        print(f"[index check] last train date = {d_last.date()} (idx {origin}) | "
              f"h=1 score date = {d_h1.date()} (idx {origin + 1}) | "
              f"gap = {gap_days} days {'OK' if 5 <= gap_days <= 9 else 'WRONG - not 1 week'}")
    for h in range(1, Forecast_horizon + 1):
        fidx = origin + h
        if fidx >= N_total_points:
            continue
        t_fc = tf.constant([[float(t_data_norm[fidx, 0])]], dtype=tf.float32)
        X, L, Y, R, Z, H, A, D, beta = model(t_fc)
        Z_v = float(np.clip(Z.numpy()[0, 0], 0.0, 1.0))
        A_v = float(np.clip(A.numpy()[0, 0], 0.0, 1.0))
        D_v = float(np.clip(D.numpy()[0, 0], 0.0, 1.0))
        preds = {
            "cases": max(0.0, GAMMA_ZW * Z_v),
            "hosp": max(0.0, GAMMA_A * A_v),
            "deaths": max(0.0, GAMMA_DW * D_v),
        }
        key = (origin, fidx)
        for s in SERIES:
            all_pred[s][key] = preds[s]
            all_naive[s][key] = float(DATA[s][origin, 0])   # persistence: last observed
            all_obs[s][fidx] = float(DATA[s][fidx, 0])
        all_beta[key] = float(np.clip(beta.numpy()[0, 0], 0.0, None))

### ensembling done in Qian et al. 2025 paper
ensemble_weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}

def collect(series):
    hres = {h: {"idx": [], "pred": [], "obs": [], "naive": []} for h in range(1, 5)}
    enum_ = np.zeros(N_total_points); eden = np.zeros(N_total_points)
    enaive = np.full(N_total_points, np.nan); eobs = np.full(N_total_points, np.nan)
    for (te, fidx), pv in all_pred[series].items():
        h = fidx - te
        hres[h]["idx"].append(fidx); hres[h]["pred"].append(pv)
        hres[h]["obs"].append(all_obs[series][fidx]); hres[h]["naive"].append(all_naive[series][(te, fidx)])
        enum_[fidx] += pv * ensemble_weights[h]; eden[fidx] += ensemble_weights[h]
        eobs[fidx] = all_obs[series][fidx]
        if h == 1:
            enaive[fidx] = all_naive[series][(te, fidx)]
    return hres, enum_, eden, enaive, eobs

LABEL = {"cases": "daily cases", "hosp": "daily admissions", "deaths": "daily deaths"}

### Plot per-horizon forecasts vs observed
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, SERIES):
    hres, *_ = collect(s)
    for h in range(1, 5):
        d = study_dates[np.array(hres[h]["idx"], dtype=int)]
        ax.plot(d, np.array(hres[h]["obs"]) * N_val, ".", color="tab:green",
                label="Observed" if h == 1 else None, zorder=5)
        ax.plot(d, np.array(hres[h]["pred"]) * N_val, "-", alpha=0.85, label=f"{h} wk ahead")
    ax.set_ylabel(LABEL[s]); ax.legend(fontsize=8, loc="upper right"); apply_date_axis(ax)
axes[0].set_title("England: Qian-replication SEIR-PINN per-horizon forecast vs observed")
plt.tight_layout(); plt.savefig("qian_perhorizon.png", dpi=150); plt.show()

### MASE by horizon (model MAE / naive MAE; <1 beats naive)
print("\nMASE by horizon (model MAE / naive MAE; <1 beats naive):")
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, SERIES):
    hres, enum_, eden, enaive, eobs = collect(s)
    valid = eden > 0
    ax.plot(study_dates, DATA[s].reshape(-1) * N_val, color="#004F94", lw=1.3, label="Observed")
    ax.plot(study_dates[valid], enum_[valid] * N_val, color="#ff7ee3", lw=1.8, label="PINN ensemble")
    nv = valid & ~np.isnan(enaive)
    ax.plot(study_dates[nv], enaive[nv] * N_val, color="orange", lw=1.0, ls="--", label="Naive")
    ax.set_ylabel(LABEL[s]); ax.legend(fontsize=9, loc="upper right"); apply_date_axis(ax)
    line = f"  {s:8s}"
    for h in range(1, 5):
        p = np.array(hres[h]["pred"]); o = np.array(hres[h]["obs"]); n = np.array(hres[h]["naive"])
        mase = np.mean(np.abs(p - o)) / (np.mean(np.abs(n - o)) + 1e-12)
        line += f"  {h}wk={mase:.3f}"
    print(line)
axes[0].set_title("England: Qian-replication SEIR-PINN ensemble forecast vs observed")
plt.tight_layout(); plt.savefig("qian_ensemble.png", dpi=150); plt.show()

### Plot R(t) = beta/gamma over time 
fig, ax = plt.subplots(figsize=(12, 5))
rt = [all_beta[k] / GAMMA for k in sorted(all_beta) if (k[1] - k[0]) == 1]
rt_d = [study_dates[k[1]] for k in sorted(all_beta) if (k[1] - k[0]) == 1]
ax.plot(rt_d, rt, color="#ff7ee3", lw=1.5, label="R(t)=beta/gamma (1-wk origin)")
ax.axhline(1.0, color="gray", lw=1, ls="--", label="R=1")
ax.set_ylabel("R(t)"); ax.legend(); apply_date_axis(ax)
ax.set_title("England: Qian-replication SEIR-PINN effective reproduction number")
plt.tight_layout(); plt.savefig("qian_Rt.png", dpi=150); plt.show()

def apply_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right')

### Plot per-horizon forecasts in 2x2 grid (cases only)
series = "cases"
LABEL_STR = "New cases per week"

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

for h, ax in zip(range(1, 5), axes.flatten()):
    ### Collect predictions for horizon h
    h_pred = []
    h_obs = []
    h_naive = []
    h_dates = []
    
    for (te, fidx), pred_val in all_pred[series].items():
        actual_h = fidx - te
        if actual_h == h:
            h_pred.append(pred_val)
            h_obs.append(all_obs[series][fidx])
            h_naive.append(all_naive[series][(te, fidx)])
            h_dates.append(study_dates[fidx])
    
    # Convert to arrays and dates
    if len(h_dates) > 0:
        h_dates = np.array(h_dates, dtype='datetime64[D]')
        h_pred = np.array(h_pred) * N_val
        h_obs = np.array(h_obs) * N_val
        h_naive = np.array(h_naive) * N_val
        
        # Plot
        ax.plot(h_dates, h_obs, color="#004F94", lw=1.5, label="Observed")
        ax.plot(h_dates, h_pred, color="#ff7ee3", lw=1.5, label=f"PINN {h}-week")
        ax.plot(h_dates, h_naive, color="orange", lw=1.0, linestyle="--", label="Naive baseline")
        
    ax.set_title(f"{h}-week-ahead forecast")
    ax.set_ylabel(LABEL_STR)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    apply_date_axis(ax)

fig.suptitle("SEIR-PINN rolling window forecasts (1–4 weeks ahead)", fontsize=14)
plt.tight_layout()
plt.savefig("7odemodel_perhorizon_grid.png", dpi=150)
plt.show()

### Also plot ensemble (weighted average across horizons)
print("\nForecast evaluation by horizon (MASE; <1 beats naive):")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

ensemble_weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}

for h, ax in zip(range(1, 5), axes.flatten()):
    h_pred = []
    h_obs = []
    h_naive = []
    h_dates = []
    
    for (te, fidx), pred_val in all_pred[series].items():
        actual_h = fidx - te
        if actual_h == h:
            h_pred.append(pred_val)
            h_obs.append(all_obs[series][fidx])
            h_naive.append(all_naive[series][(te, fidx)])
            h_dates.append(study_dates[fidx])
    
    if len(h_dates) > 0:
        h_dates = np.array(h_dates, dtype='datetime64[D]')
        h_pred = np.array(h_pred) * N_val
        h_obs = np.array(h_obs) * N_val
        h_naive = np.array(h_naive) * N_val
        
        # Metrics
        mae_pinn = np.mean(np.abs(h_pred - h_obs))
        mae_naive = np.mean(np.abs(h_naive - h_obs))
        mase = mae_pinn / (mae_naive + 1e-10)
        
        print(f"  {h}-week | PINN MAE={mae_pinn:.2e}  Naive MAE={mae_naive:.2e}  MASE={mase:.3f}")
        
        ax.plot(h_dates, h_obs, color="#004F94", lw=1.5, label="Observed")
        ax.plot(h_dates, h_pred, color="#ff7ee3", lw=1.5, label=f"PINN {h}-week")
        ax.plot(h_dates, h_naive, color="orange", lw=1.0, linestyle="--", label="Naive baseline")
        
    ax.set_title(f"{h}-week-ahead forecast")
    ax.set_ylabel(LABEL_STR)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    apply_date_axis(ax)

fig.suptitle("7-Compartment SEIR-PINN: Cases (1–4 weeks ahead)", fontsize=14)
plt.tight_layout()
plt.savefig("7comp_cases_grid.png", dpi=150)
plt.show()