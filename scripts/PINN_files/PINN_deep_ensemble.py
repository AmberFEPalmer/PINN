import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import regularizers
import pandas as pd
import os
import pickle
import matplotlib.dates as mdates
from statistics import NormalDist

### Deep ensemble of the 7-compartment SEIR-PINN in PINN_hospitalisations_deaths.py
### Uncertainty quantification by independently training M copies of the same PINN,
### each from a different random initialisation, and treating the spread of their
### forecasts as the epistemic (model/optimisation) uncertainty.
### Lakshminarayanan et al. 2017, "Simple and Scalable Predictive Uncertainty
### Estimation using Deep Ensembles" - https://arxiv.org/abs/1612.01474

### ensemble settings
N_MEMBERS = 5              ### number of independently initialised PINNs
MEMBER_SEEDS = [42, 7, 123, 2024, 31337][:N_MEMBERS]
N_ITER = 50_000            ### Adam iterations per window per member 
QUICK_TEST = False         ### True = 2 members x 2000 iterations, for checking the pipeline runs
if QUICK_TEST:
    N_MEMBERS = 2
    MEMBER_SEEDS = MEMBER_SEEDS[:2]
    N_ITER = 2_000

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
GAMMA_A = 1.0  * 7 ### hospital admissisons reporting
GAMMA_DW = 0.1 * 7 ### death reporting
RHO = 0.5

### Define PINN
### identical architecture to PINN_hospitalisations_deaths.py
### L2 regularisation for hidden layers -> helps to prevent overfitting
### https://keras.io/api/layers/regularizers/
def create_pinn_model():
    t_input = Input(shape=(1,), name='time_input')

    ### hidden layers compartmental model, 3, 50 neurons
    ### tanh activation for nonlinearity, L2 regularisation to prevent overfitting
    core = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    core = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(core)
    core = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(core)

    ### 7 output layers
    ### latent, infectious, removed, case-box, hospitalisations, admissions, deaths
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
                 n_iter=50_000, warm=None, init_jitter=(0.0, 0.0, 0.0), verbose=False):
    model = create_pinn_model()
    logit_p_h = tf.Variable(-3.0 + init_jitter[0], dtype=tf.float32) ### learnable parameter - probability infection leads to hospitalisation
    logit_p_d = tf.Variable(-1.0 + init_jitter[1], dtype=tf.float32) ### learnable parameter - probability hospitalisation leds to death
    asc0 = 0.5 + 0.047 * 0.5 ### one off estimate of ascertainment
    L0_seed = max(inc0 / (asc0 * GAMMA) * (GAMMA / ETA), 1e-6)
    log_L0 = tf.Variable(np.log(L0_seed) + init_jitter[2], dtype=tf.float32)

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
            if verbose and itr % 10000 == 0:
                print(f"    iter {itr:5d} | total {lv:.2e} | "
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
FLUX = {"cases": GAMMA_ZW, "hosp": GAMMA_A, "deaths": GAMMA_DW}
LABEL = {"cases": "New cases per week", "hosp": "Admissions per week", "deaths": "Deaths per week"}

inc0 = float(I_data[0, 0])
R0 = 0.06
total_weeks = float(N_total_points - 1)
WINDOWS = list(range(First_train_weeks, N_total_points - Forecast_horizon + 1))

### in-sample residual sd on the training window, per series
### this is the aleatoric (observation-noise) term sigma_m for the member
def training_residual_sd(model, train_end):
    t_tr = tf.convert_to_tensor(t_data_norm[:train_end], dtype=tf.float32)
    _, _, _, _, Z, _, A, D, _ = model(t_tr)
    fitted = {
        "cases": GAMMA_ZW * Z.numpy().reshape(-1),
        "hosp": GAMMA_A * A.numpy().reshape(-1),
        "deaths": GAMMA_DW * D.numpy().reshape(-1),
    }
    out = {}
    for s in SERIES:
        resid = fitted[s] - DATA[s][:train_end, 0]
        out[s] = float(np.std(resid))
    return out

### train one ensemble member over the full rolling-window protocol
def run_member(member_idx, seed):
    print(f"\n{'='*70}\nensemble member {member_idx + 1}/{N_MEMBERS} (seed {seed})\n{'='*70}")
    ### different seed -> different Glorot initialisation for every layer
    np.random.seed(seed)
    tf.random.set_seed(seed)
    ### member 0 uses the exact initialisation of the single-model script
    if member_idx == 0:
        jitter = (0.0, 0.0, 0.0)
    else:
        rng = np.random.default_rng(seed)
        jitter = tuple(rng.normal(0.0, [0.5, 0.5, 0.3]))

    m_pred = {s: {} for s in SERIES}
    m_beta = {}
    m_sigma = {}   ### (train_end) -> {series: sd}

    warm = None
    for train_end in WINDOWS:
        model, lph, lpd, lL0 = train_window(
            t_data_norm[:train_end], I_data[:train_end], H_data[:train_end], D_data[:train_end],
            total_weeks=total_weeks, inc0=inc0, R0=R0, n_iter=N_ITER, warm=warm,
            init_jitter=jitter, verbose=(train_end == WINDOWS[0]))
        warm = (model.get_weights(), float(lph.numpy()), float(lpd.numpy()), float(lL0.numpy()))

        origin = train_end - 1
        m_sigma[origin] = training_residual_sd(model, train_end)

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
                m_pred[s][key] = preds[s]
            m_beta[key] = float(np.clip(beta.numpy()[0, 0], 0.0, None))

        msg = f"  [member {member_idx + 1}] train_end={train_end:3d} done"
        if (origin, origin + 1) in m_pred["cases"]:
            msg += (f" | 1wk cases pred={m_pred['cases'][(origin, origin + 1)] * N_val:,.0f}"
                    f" obs={I_data[origin + 1, 0] * N_val:,.0f}")
        print(msg)

    return {"pred": m_pred, "beta": m_beta, "sigma": m_sigma, "seed": seed, "jitter": jitter}

### train the ensemble
### saved incrementally so a long run can be inspected (or resumed) if interrupted
os.makedirs("../../forecasts", exist_ok=True)
DUMP_PATH = "../../forecasts/PINN_deep_ensemble.pkl"

print(f"\ndeep ensemble: {N_MEMBERS} members x {len(WINDOWS)} windows x {N_ITER:,} iterations")
members = []
for i, seed in enumerate(MEMBER_SEEDS):
    members.append(run_member(i, seed))
    with open(DUMP_PATH, "wb") as f:
        pickle.dump({"members": members, "n_members_done": len(members)}, f)
    print(f"  -> checkpoint written after member {i + 1} ({DUMP_PATH})")

### aggregate members into a predictive distribution
### mu = mean over members
### var = mean member variance (aleatoric) + variance of member means (epistemic)
all_pred = {s: {} for s in SERIES}      ### ensemble mean 
all_naive = {s: {} for s in SERIES}
all_obs = {s: {} for s in SERIES}
ens_sd = {s: {} for s in SERIES}        ### total predictive sd
ens_sd_epi = {s: {} for s in SERIES}    ### between-member sd only
ens_sd_alea = {s: {} for s in SERIES}   ### observation-noise sd only
member_pred = {s: {} for s in SERIES}   ### key -> array of member forecasts
all_beta = {}
beta_sd = {}

keys = sorted(members[0]["pred"]["cases"].keys())
for key in keys:
    origin, fidx = key
    for s in SERIES:
        mu_m = np.array([m["pred"][s][key] for m in members], dtype=float)
        sd_m = np.array([m["sigma"][origin][s] for m in members], dtype=float)
        mu = float(mu_m.mean())
        var_epi = float(mu_m.var(ddof=1)) if len(mu_m) > 1 else 0.0
        var_alea = float((sd_m ** 2).mean())
        all_pred[s][key] = mu
        member_pred[s][key] = mu_m
        ens_sd_epi[s][key] = np.sqrt(var_epi)
        ens_sd_alea[s][key] = np.sqrt(var_alea)
        ens_sd[s][key] = np.sqrt(var_epi + var_alea)
        all_naive[s][key] = float(DATA[s][origin, 0])   # persistence: last observed
        all_obs[s][fidx] = float(DATA[s][fidx, 0])
    b_m = np.array([m["beta"][key] for m in members], dtype=float)
    all_beta[key] = float(b_m.mean())
    beta_sd[key] = float(b_m.std(ddof=1)) if len(b_m) > 1 else 0.0

### quantiles of the moment-matched normal, truncated at zero
QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
Z_Q = {q: NormalDist().inv_cdf(q) for q in QUANTILES}

def pred_quantiles(s, key):
    mu, sd = all_pred[s][key], ens_sd[s][key]
    return {q: max(0.0, mu + Z_Q[q] * sd) for q in QUANTILES}

quantile_store = {s: {k: pred_quantiles(s, k) for k in keys} for s in SERIES}

### weighted interval score (Bracher et al. 2021, PLoS Comput Biol)
### WIS = 1/(K+1/2) * [ 1/2*|y-median| + sum_k alpha_k/2 * IS_alpha_k ]
### lower is better; on the same scale as MAE, so directly comparable to it
PI_PAIRS = [(0.025, 0.975), (0.1, 0.9), (0.25, 0.75)]

def wis(y, qs):
    total = 0.5 * abs(y - qs[0.5])
    for lo_q, hi_q in PI_PAIRS:
        alpha = 2 * lo_q
        l, u = qs[lo_q], qs[hi_q]
        interval_score = (u - l)
        if y < l:
            interval_score += (2 / alpha) * (l - y)
        if y > u:
            interval_score += (2 / alpha) * (y - u)
        total += (alpha / 2) * interval_score
    return total / (len(PI_PAIRS) + 0.5)

### per-horizon evaluation
### MASE for the point forecast, empirical coverage + WIS for the intervals
rows = []
print("\n=== deep-ensemble evaluation (values in counts, N scaled back up) ===")
for s in SERIES:
    for h in range(1, Forecast_horizon + 1):
        hkeys = [k for k in keys if k[1] - k[0] == h]
        if not hkeys:
            continue
        obs = np.array([all_obs[s][k[1]] for k in hkeys])
        mu = np.array([all_pred[s][k] for k in hkeys])
        naive = np.array([all_naive[s][k] for k in hkeys])
        mae_naive = np.mean(np.abs(naive - obs)) + 1e-12
        mase_ens = np.mean(np.abs(mu - obs)) / mae_naive
        ### mean MASE of the individual members, to show what the ensemble buys us
        mase_members = [
            np.mean(np.abs(np.array([m["pred"][s][k] for k in hkeys]) - obs)) / mae_naive
            for m in members
        ]
        cov50 = np.mean([(quantile_store[s][k][0.25] <= all_obs[s][k[1]] <= quantile_store[s][k][0.75])
                         for k in hkeys])
        cov90 = np.mean([(quantile_store[s][k][0.1] <= all_obs[s][k[1]] <= quantile_store[s][k][0.9])
                         for k in hkeys])
        cov95 = np.mean([(quantile_store[s][k][0.025] <= all_obs[s][k[1]] <= quantile_store[s][k][0.975])
                         for k in hkeys])
        wis_h = np.mean([wis(all_obs[s][k[1]], quantile_store[s][k]) for k in hkeys]) * N_val
        sd_epi = np.mean([ens_sd_epi[s][k] for k in hkeys]) * N_val
        sd_alea = np.mean([ens_sd_alea[s][k] for k in hkeys]) * N_val
        rows.append({
            "series": s, "horizon": h, "n_forecasts": len(hkeys),
            "mase_ensemble": mase_ens,
            "mase_member_mean": float(np.mean(mase_members)),
            "mase_member_best": float(np.min(mase_members)),
            "mase_member_worst": float(np.max(mase_members)),
            "coverage_50": cov50, "coverage_90": cov90, "coverage_95": cov95,
            "wis": wis_h, "mean_sd_epistemic": sd_epi, "mean_sd_aleatoric": sd_alea,
        })
        print(f"  {s:7s} h={h} | MASE ens={mase_ens:.3f} "
              f"(members {np.mean(mase_members):.3f} avg, {np.min(mase_members):.3f}-{np.max(mase_members):.3f}) "
              f"| cov50={cov50:.2f} cov90={cov90:.2f} cov95={cov95:.2f} | WIS={wis_h:.3e}")

metrics = pd.DataFrame(rows)
metrics.to_csv("deep_ensemble_metrics.csv", index=False)
print("\nmetrics written to deep_ensemble_metrics.csv")

### helper: pull a horizon's forecast + interval as sorted arrays, in count units
def horizon_arrays(s, h):
    hkeys = sorted([k for k in keys if k[1] - k[0] == h], key=lambda k: k[1])
    idx = np.array([k[1] for k in hkeys], dtype=int)
    mu = np.array([all_pred[s][k] for k in hkeys]) * N_val
    obs = np.array([all_obs[s][k[1]] for k in hkeys]) * N_val
    lo50 = np.array([quantile_store[s][k][0.25] for k in hkeys]) * N_val
    hi50 = np.array([quantile_store[s][k][0.75] for k in hkeys]) * N_val
    lo90 = np.array([quantile_store[s][k][0.1] for k in hkeys]) * N_val
    hi90 = np.array([quantile_store[s][k][0.9] for k in hkeys]) * N_val
    mem = np.array([[m["pred"][s][k] for k in hkeys] for m in members]) * N_val
    return study_dates[idx], mu, obs, lo50, hi50, lo90, hi90, mem

### Plot 1: fan chart, one row per series, 1-week-ahead
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, SERIES):
    d, mu, obs, lo50, hi50, lo90, hi90, _ = horizon_arrays(s, 1)
    ax.fill_between(d, lo90, hi90, color="#f199c6", alpha=0.30, label="80% PI")
    ax.fill_between(d, lo50, hi50, color="#f199c6", alpha=0.55, label="50% PI")
    ax.plot(d, mu, color="#c2185b", lw=1.8, label="Deep ensemble mean")
    ax.plot(d, obs, color="#004F94", lw=1.3, ls=":", label="Observed")
    ax.set_ylabel(LABEL[s]); ax.legend(fontsize=9, loc="upper right"); apply_date_axis(ax)
axes[0].set_title("England: deep-ensemble SEIR-PINN, 1-week-ahead with prediction intervals")
plt.tight_layout(); plt.savefig("deep_ensemble_fan_h1.png", dpi=150); plt.show()

### Plot 2: cases, 2x2 grid over horizons
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for h, ax in zip(range(1, Forecast_horizon + 1), axes.flatten()):
    d, mu, obs, lo50, hi50, lo90, hi90, _ = horizon_arrays("cases", h)
    ax.fill_between(d, lo90, hi90, color="#f199c6", alpha=0.30, label="80% PI")
    ax.fill_between(d, lo50, hi50, color="#f199c6", alpha=0.55, label="50% PI")
    ax.plot(d, mu, color="#c2185b", lw=1.6, label=f"Ensemble {h}-week")
    ax.plot(d, obs, color="#004F94", lw=1.4, label="Observed")
    ax.set_title(f"{h}-week-ahead forecast")
    ax.set_ylabel(LABEL["cases"]); ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3); apply_date_axis(ax)
fig.suptitle(f"Deep-ensemble SEIR-PINN ({N_MEMBERS} members): cases, 1-4 weeks ahead", fontsize=14)
plt.tight_layout(); plt.savefig("deep_ensemble_cases_grid.png", dpi=150); plt.show()

### Plot 3: individual member trajectories - shows where members disagree
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, SERIES):
    d, mu, obs, *_, mem = horizon_arrays(s, 1)
    for i in range(mem.shape[0]):
        ax.plot(d, mem[i], color="#f199c6", lw=1.0, alpha=0.7,
                label=f"members (n={mem.shape[0]})" if i == 0 else None)
    ax.plot(d, mu, color="#c2185b", lw=2.0, label="Ensemble mean")
    ax.plot(d, obs, color="#004F94", lw=1.3, ls=":", label="Observed")
    ax.set_ylabel(LABEL[s]); ax.legend(fontsize=9, loc="upper right"); apply_date_axis(ax)
axes[0].set_title("England: individual deep-ensemble members, 1-week-ahead")
plt.tight_layout(); plt.savefig("deep_ensemble_members_h1.png", dpi=150); plt.show()

### Plot 4: calibration - nominal vs empirical coverage
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
nominal = [0.5, 0.9, 0.95]
x = np.arange(Forecast_horizon)
width = 0.8 / len(nominal)
for ax, s in zip(axes, SERIES):
    sub = metrics[metrics["series"] == s].sort_values("horizon")
    for i, (lvl, col) in enumerate(zip(nominal, ["coverage_50", "coverage_90", "coverage_95"])):
        ax.bar(x + i * width - 0.4 + width / 2, sub[col].values, width,
               label=f"{int(lvl * 100)}% PI")
        ax.axhline(lvl, color="grey", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"{h}wk" for h in range(1, Forecast_horizon + 1)])
    ax.set_title(LABEL[s]); ax.set_xlabel("Horizon"); ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylabel("Empirical coverage")
axes[0].legend(fontsize=9, loc="lower left")
fig.suptitle("Deep-ensemble interval calibration (dashed = nominal level)", fontsize=14)
plt.tight_layout(); plt.savefig("deep_ensemble_coverage.png", dpi=150); plt.show()

### Plot 5: variance decomposition - how much uncertainty is between-member vs noise
fig, ax = plt.subplots(figsize=(12, 5))
sub = metrics[metrics["series"] == "cases"].sort_values("horizon")
ax.bar(sub["horizon"] - 0.2, sub["mean_sd_epistemic"], 0.4, color="#c2185b", label="Epistemic (between members)")
ax.bar(sub["horizon"] + 0.2, sub["mean_sd_aleatoric"], 0.4, color="#f1c40f", label="Aleatoric (residual noise)")
ax.set_xticks(list(range(1, Forecast_horizon + 1)))
ax.set_xlabel("Horizon (weeks ahead)"); ax.set_ylabel("Mean predictive sd (cases per week)")
ax.set_title("Deep-ensemble uncertainty decomposition, cases")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("deep_ensemble_variance_split.png", dpi=150); plt.show()

### Plot 6: R(t) with between-member band
fig, ax = plt.subplots(figsize=(12, 5))
h1 = sorted([k for k in keys if k[1] - k[0] == 1], key=lambda k: k[1])
rt_d = [study_dates[k[1]] for k in h1]
rt = np.array([all_beta[k] for k in h1]) / GAMMA
rt_sd = np.array([beta_sd[k] for k in h1]) / GAMMA
ax.fill_between(rt_d, rt - 1.96 * rt_sd, rt + 1.96 * rt_sd, color="#f199c6", alpha=0.35,
                label="±1.96 sd across members")
ax.plot(rt_d, rt, color="#c2185b", lw=1.5, label="R(t)=beta/gamma (ensemble mean)")
ax.axhline(1.0, color="gray", lw=1, ls="--", label="R=1")
ax.set_ylabel("R(t)"); ax.legend(); apply_date_axis(ax)
ax.set_title("England: deep-ensemble SEIR-PINN effective reproduction number")
plt.tight_layout(); plt.savefig("deep_ensemble_Rt.png", dpi=150); plt.show()

### Save
### all_pred/all_obs/all_naive keep the PINN.pkl layout so compare_forecasts.py can
### read this file directly; the extra keys carry the uncertainty information
with open(DUMP_PATH, "wb") as f:
    pickle.dump({
        "all_pred": all_pred,          ### ensemble mean point forecast
        "all_obs": all_obs,
        "all_naive": all_naive,
        "member_pred": member_pred,    ### key -> array of per-member forecasts
        "quantiles": quantile_store,   ### key -> {quantile level: value}
        "sd_total": ens_sd,
        "sd_epistemic": ens_sd_epi,
        "sd_aleatoric": ens_sd_alea,
        "all_beta": all_beta,
        "beta_sd": beta_sd,
        "members": members,
        "metrics": metrics,
        "config": {"n_members": N_MEMBERS, "seeds": MEMBER_SEEDS, "n_iter": N_ITER,
                   "first_train_weeks": First_train_weeks,
                   "forecast_horizon": Forecast_horizon, "quantiles": QUANTILES},
    }, f)
print(f"\nsaved deep ensemble to {DUMP_PATH}")
