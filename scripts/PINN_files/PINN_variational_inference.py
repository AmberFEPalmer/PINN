import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow_probability as tfp
import pandas as pd
import os
import pickle
import matplotlib.dates as mdates

tfd = tfp.distributions
tfpl = tfp.layers

### Bayesian version of the 7-compartment SEIR-PINN fitted by variational inference
### Every weight is a random variable. The exact posterior p(w | data, physics) is
### intractable, so it is approximated by a mean-field Gaussian q(w) whose means and
### standard deviations are learned by maximising the ELBO:
###     ELBO = E_q[ log p(data | w) ] - KL( q(w) || p(w) )

### Forecast uncertainty is then Monte Carlo: N_MC draws w ~ q(w) give N_MC forecast
### trajectories (epistemic), + learned observation noise
### sigma_s (aleatoric). 

### VI settings
N_ITER = 50_000            ### Adam iterations per window 
N_MC = 500                 ### posterior draws used to form each forecast distribution
KL_MAX = 0.05              ### final weight on the KL term (KL is also divided by n_data)
KL_RAMP_ITERS = 5_000      ### KL annealing: ramp 0 -> KL_MAX, stops the posterior collapsing early
                           ### https://arxiv.org/abs/1903.10145
SIGMA_MIN, SIGMA_MAX = 1e-2, 1.0   ### observation-noise sd bounds, in units of each series' scale
QUICK_TEST = False         ### True = 2000 iterations, 50 draws, for checking the pipeline runs
if QUICK_TEST:
    N_ITER = 2_000
    N_MC = 50

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
GAMMA_A = 1.0  * 7 ### hospital admissisons reporting
GAMMA_DW = 0.1 * 7 ### death reporting
RHO = 0.5

### Define Bayesian PINN
### same topology as the deterministic PINN (3x50 SEIR trunk + 3x50 beta trunk),
### but every Dense layer is replaced by DenseFlipout: the kernel is a mean-field
### Gaussian q(w) = N(loc, softplus(rho)) with a standard normal prior, and each
### forward pass draws a fresh weight sample
### https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout
### L2 regularisation is dropped - the KL term to the N(0,1) prior now plays that role
def variational_dense(units, activation=None, name=None):
    return tfpl.DenseFlipout(units, activation=activation, name=name)

def create_bayesian_pinn_model():
    t_input = Input(shape=(1,), name='time_input')

    ### hidden layers compartmental model, 3, 50 neurons, tanh activation
    core = variational_dense(50, activation='tanh')(t_input)
    core = variational_dense(50, activation='tanh')(core)
    core = variational_dense(50, activation='tanh')(core)

    ### 7 output layers
    ### latent, infectious, removed, case-box, hospitalisations, admissions, deaths
    ### softplus activation to ensure non-negativity of compartments and fluxes
    L = variational_dense(1, activation='softplus', name='L')(core)
    Y = variational_dense(1, activation='softplus', name='Y')(core)
    R = variational_dense(1, activation='softplus', name='R')(core)
    Z = variational_dense(1, activation='softplus', name='Z')(core)
    H = variational_dense(1, activation='softplus', name='H')(core)
    A = variational_dense(1, activation='softplus', name='A')(core)
    D = variational_dense(1, activation='softplus', name='D')(core)
    X = Lambda(lambda z: 1.0 - z[0] - z[1] - z[2], name='X')([L, Y, R])

    ### hidden beta subnetwork, 3 layers, 50 neurons, tanh activation
    beta_h = variational_dense(50, activation='tanh')(t_input)
    beta_h = variational_dense(50, activation='tanh')(beta_h)
    beta_h = variational_dense(50, activation='tanh')(beta_h)

    ### beta output
    ### no activation function
    ### in log form
    log_beta = variational_dense(1, activation=None, name='log_beta')(beta_h)
    log_beta = Lambda(lambda x: tf.clip_by_value(x, -3.0, 1.5), name='clipped_log_beta')(log_beta)
    beta = Lambda(lambda x: tf.exp(x), name='beta')(log_beta)

    return Model(inputs=t_input, outputs=[X, L, Y, R, Z, H, A, D, beta])

### KL( q(w) || p(w) ) summed over every variational layer
### computed directly from each layer's posterior/prior 
def total_kl(net):
    kl = tf.constant(0.0, dtype=tf.float32)
    for layer in net.layers:
        if getattr(layer, "kernel_prior", None) is not None:
            kl += tf.reduce_sum(tfd.kl_divergence(layer.kernel_posterior, layer.kernel_prior))
    return kl

### the data term is now a Gaussian log-likelihood with learnable observation noise,
def compute_loss(t_col, t_data, I_d, H_d, D_d, net, logit_p_h, logit_p_d,
                 total_weeks, L0, R0, log_sig, kl_weight, n_data):

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
    ### one weight sample is drawn here; the physics residual is therefore evaluated
    ### on a draw from q(w), which is what makes the physics constrain the posterior
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        X, L, Y, R, Z, H, A, D, beta = net(t_col, training=True)

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
    _, L_0, Y_0, R_0, Z_0, H_0, A_0, D_0, _ = net(t_zero, training=True)
    ic_loss = tf.reduce_mean(
        tf.square(L_0 - L0) + tf.square(Y_0 - Y0) + tf.square(R_0 - R0) +
        tf.square(Z_0 - Z0) + tf.square(H_0 - H0) +
        tf.square(A_0 - A0) + tf.square(D_0 - D0)
    )

    _, _, Y_p, _, Z_p, _, A_p, D_p, _ = net(t_data, training=True)
    case_p = gamma_zw * Z_p
    adm_p = gamma_a * A_p
    death_p = gamma_dw * D_p

    ### Gaussian NLL per series on scale-normalised residuals
    ### sigma is learnable (clipped) - this is the aleatoric / observation-noise term
    sig = tf.clip_by_value(tf.exp(log_sig), SIGMA_MIN, SIGMA_MAX)   # [sig_cases, sig_hosp, sig_deaths]
    def nll(pred, obs, scale, s):
        r = (pred - obs) / scale
        return tf.reduce_mean(0.5 * tf.square(r / s) + tf.math.log(s))
    case_loss = nll(case_p, I_d, I_scale, sig[0])
    hosp_loss = nll(adm_p, H_d, H_scale, sig[1])
    death_loss = nll(death_p, D_d, D_scale, sig[2])
    data_loss = case_loss + hosp_loss + death_loss

    ### KL is over the whole network
    kl = total_kl(net)
    kl_term = kl_weight * kl / tf.cast(n_data, tf.float32)

    total = 1.0 * data_loss + 0.1 * ode_loss + 0.1 * ic_loss + kl_term
    return total, {"data_loss": data_loss, "case": case_loss, "hosp": hosp_loss,
                   "death": death_loss, "IC_loss": ic_loss, "ODE_loss": ode_loss,
                   "KL": kl, "KL_term": kl_term}

### define single window training
### returns the fitted variational posterior (the model) and the learned sigmas
def train_window(t_train_norm, I_tr, H_tr, D_tr, total_weeks, inc0, R0,
                 n_iter=50_000, warm=None, verbose=False):
    model = create_bayesian_pinn_model()
    logit_p_h = tf.Variable(-3.0, dtype=tf.float32) ### learnable parameter - probability infection leads to hospitalisation
    logit_p_d = tf.Variable(-1.0, dtype=tf.float32) ### learnable parameter - probability hospitalisation leds to death
    asc0 = 0.5 + 0.047 * 0.5 ### one off estimate of ascertainment
    L0_seed = max(inc0 / (asc0 * GAMMA) * (GAMMA / ETA), 1e-6)
    log_L0 = tf.Variable(np.log(L0_seed), dtype=tf.float32)
    ### one observation-noise sd per series, started at 10% of the series scale
    log_sig = tf.Variable(np.log([0.1, 0.1, 0.1]), dtype=tf.float32)

    ### warm start: carry the previous window's variational parameters (both the
    ### posterior means and the posterior sds) into this window
    ### https://arxiv.org/abs/1910.08475
    if warm is not None:
        model.set_weights(warm[0])
        logit_p_h.assign(warm[1]); logit_p_d.assign(warm[2]); log_L0.assign(warm[3])
        log_sig.assign(warm[4])

    ### Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. 2017
    optm = Adam(learning_rate=0.001)
    t_col = tf.convert_to_tensor(np.linspace(0, 1.0, 500).reshape(-1, 1), dtype=tf.float32)
    t_tr = tf.convert_to_tensor(t_train_norm, dtype=tf.float32)
    I_t = tf.convert_to_tensor(I_tr, dtype=tf.float32)
    H_t = tf.convert_to_tensor(H_tr, dtype=tf.float32)
    D_t = tf.convert_to_tensor(D_tr, dtype=tf.float32)
    n_data = int(len(t_train_norm))
    kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    ### @tf.function makes faster
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            L0 = tf.exp(log_L0)
            loss, ld = compute_loss(t_col, t_tr, I_t, H_t, D_t, model,
                                    logit_p_h, logit_p_d, total_weeks, L0, R0,
                                    log_sig, kl_weight, n_data)
        vlist = model.trainable_variables + [logit_p_h, logit_p_d, log_L0, log_sig]
        grads = tape.gradient(loss, vlist)
        optm.apply_gradients(zip(grads, vlist))
        return loss, ld

    loss_ema = None
    for itr in range(n_iter):
        kl_weight.assign(min(KL_MAX, KL_MAX * itr / KL_RAMP_ITERS))
        loss, ld = step()
        if itr % 1000 == 0:
            lv = float(loss)
            loss_ema = lv if loss_ema is None else 0.9 * loss_ema + 0.1 * lv
            if verbose and itr % 10000 == 0:
                sig = np.clip(np.exp(log_sig.numpy()), SIGMA_MIN, SIGMA_MAX)
                print(f"    iter {itr:5d} | -ELBO {lv:.2e} (ema {loss_ema:.2e}) | "
                      f"case {float(ld['case']):.2e} hosp {float(ld['hosp']):.2e} "
                      f"death {float(ld['death']):.2e} | ODE {float(ld['ODE_loss']):.2e} | "
                      f"IC {float(ld['IC_loss']):.2e} | KL {float(ld['KL']):.2e} "
                      f"(term {float(ld['KL_term']):.2e}) | p_h {float(tf.sigmoid(logit_p_h)):.4f} "
                      f"p_d {float(tf.sigmoid(logit_p_d)):.4f} L0 {float(tf.exp(log_L0)):.2e} | "
                      f"sig {sig[0]:.3f}/{sig[1]:.3f}/{sig[2]:.3f}")
    return model, logit_p_h, logit_p_d, log_L0, log_sig

First_train_weeks = 17
Forecast_horizon = 4
SERIES = ["cases", "hosp", "deaths"]
DATA = {"cases": I_data, "hosp": H_data, "deaths": D_data}
SCALE = {"cases": I_scale, "hosp": H_scale, "deaths": D_scale}
LABEL = {"cases": "New cases per week", "hosp": "Admissions per week", "deaths": "Deaths per week"}

inc0 = float(I_data[0, 0])
R0 = 0.06
total_weeks = float(N_total_points - 1)
WINDOWS = list(range(First_train_weeks, N_total_points - Forecast_horizon + 1))

### Monte Carlo posterior sampling
def posterior_draws(model, t_query, n_samples=N_MC):
    t = tf.convert_to_tensor(np.asarray(t_query, dtype=np.float32).reshape(-1, 1))

    @tf.function
    def one_draw():
        _, _, _, _, Z, _, A, D, beta = model(t, training=True)
        return Z, A, D, beta

    draws = {s: [] for s in SERIES}
    beta_draws = []
    for _ in range(n_samples):
        Z, A, D, beta = one_draw()
        draws["cases"].append(GAMMA_ZW * np.clip(Z.numpy().reshape(-1), 0.0, 1.0))
        draws["hosp"].append(GAMMA_A * np.clip(A.numpy().reshape(-1), 0.0, 1.0))
        draws["deaths"].append(GAMMA_DW * np.clip(D.numpy().reshape(-1), 0.0, 1.0))
        beta_draws.append(np.clip(beta.numpy().reshape(-1), 0.0, None))
    return ({s: np.asarray(v, dtype=np.float32) for s, v in draws.items()},   ### (n_samples, n_times)
            np.asarray(beta_draws, dtype=np.float32))

### rolling-window fit
### one variational posterior per window, warm-started from the previous window
all_pred = {s: {} for s in SERIES}       ### posterior predictive mean - same format as PINN.pkl
all_naive = {s: {} for s in SERIES}
all_obs = {s: {} for s in SERIES}
f_draws = {s: {} for s in SERIES}        ### key -> (N_MC,) posterior draws of the mean function
sigma_obs = {s: {} for s in SERIES}      ### key -> observation-noise sd, data units
ens_sd = {s: {} for s in SERIES}         ### total predictive sd
ens_sd_epi = {s: {} for s in SERIES}     ### posterior (weight) uncertainty
ens_sd_alea = {s: {} for s in SERIES}    ### observation noise
all_beta = {}
beta_draws_store = {}

os.makedirs("../../forecasts", exist_ok=True)
DUMP_PATH = "../../forecasts/PINN_variational.pkl"

print(f"\nvariational Bayesian PINN: {len(WINDOWS)} windows x {N_ITER:,} iterations, "
      f"{N_MC} posterior draws per forecast")

warm = None
for train_end in WINDOWS:
    print(f"Train weeks 1-{train_end} | forecast {train_end+1}-{train_end+Forecast_horizon} "
          f"| {'warm' if warm else 'cold'} start")

    model, lph, lpd, lL0, lsig = train_window(
        t_data_norm[:train_end], I_data[:train_end], H_data[:train_end], D_data[:train_end],
        total_weeks=total_weeks, inc0=inc0, R0=R0, n_iter=N_ITER, warm=warm,
        verbose=(train_end == WINDOWS[0]))
    warm = (model.get_weights(), float(lph.numpy()), float(lpd.numpy()),
            float(lL0.numpy()), lsig.numpy())

    sig_scaled = np.clip(np.exp(lsig.numpy()), SIGMA_MIN, SIGMA_MAX)
    sig_data = {s: float(sig_scaled[i] * SCALE[s]) for i, s in enumerate(SERIES)}

    origin = train_end - 1
    if train_end == First_train_weeks:
        d_last = pd.Timestamp(study_dates[origin])
        d_h1 = pd.Timestamp(study_dates[origin + 1])
        gap_days = (d_h1 - d_last).days
        print(f"[index check] last train date = {d_last.date()} (idx {origin}) | "
              f"h=1 score date = {d_h1.date()} (idx {origin + 1}) | "
              f"gap = {gap_days} days {'OK' if 5 <= gap_days <= 9 else 'WRONG - not 1 week'}")

    ### all horizons of this window sampled together, one coherent draw at a time
    fidxs = [origin + h for h in range(1, Forecast_horizon + 1) if origin + h < N_total_points]
    if not fidxs:
        continue
    t_query = [float(t_data_norm[fi, 0]) for fi in fidxs]
    draws, b_draws = posterior_draws(model, t_query)

    for j, fidx in enumerate(fidxs):
        key = (origin, fidx)
        for s in SERIES:
            d = draws[s][:, j]
            all_pred[s][key] = float(d.mean())
            f_draws[s][key] = d
            sigma_obs[s][key] = sig_data[s]
            ens_sd_epi[s][key] = float(d.std(ddof=1))
            ens_sd_alea[s][key] = sig_data[s]
            ens_sd[s][key] = float(np.sqrt(d.var(ddof=1) + sig_data[s] ** 2))
            all_naive[s][key] = float(DATA[s][origin, 0])   # persistence: last observed
            all_obs[s][fidx] = float(DATA[s][fidx, 0])
        all_beta[key] = float(b_draws[:, j].mean())
        beta_draws_store[key] = b_draws[:, j]

    k1 = (origin, origin + 1)
    if k1 in all_pred["cases"]:
        print(f"  train_end={train_end:3d} done | 1wk cases mean={all_pred['cases'][k1] * N_val:,.0f} "
              f"(sd {ens_sd['cases'][k1] * N_val:,.0f}) obs={I_data[origin + 1, 0] * N_val:,.0f}")

    ### checkpoint so a long run can be inspected if interrupted
    with open(DUMP_PATH, "wb") as f:
        pickle.dump({"all_pred": all_pred, "all_obs": all_obs, "all_naive": all_naive,
                     "f_draws": f_draws, "sigma_obs": sigma_obs,
                     "last_train_end": train_end}, f)

keys = sorted(all_pred["cases"].keys())

### posterior predictive quantiles
QUANTILES = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
rng = np.random.default_rng(0)

def predictive_sample(s, key):
    f = f_draws[s][key]
    return np.clip(f + rng.normal(0.0, sigma_obs[s][key], size=f.shape), 0.0, None)

quantile_store = {s: {} for s in SERIES}
for s in SERIES:
    for key in keys:
        y = predictive_sample(s, key)
        quantile_store[s][key] = {q: float(np.quantile(y, q)) for q in QUANTILES}

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
### MASE for the posterior mean, empirical coverage + WIS for the credible intervals
rows = []
print("\n=== variational Bayesian PINN evaluation (values in counts, N scaled back up) ===")
for s in SERIES:
    for h in range(1, Forecast_horizon + 1):
        hkeys = [k for k in keys if k[1] - k[0] == h]
        if not hkeys:
            continue
        obs = np.array([all_obs[s][k[1]] for k in hkeys])
        mu = np.array([all_pred[s][k] for k in hkeys])
        med = np.array([quantile_store[s][k][0.5] for k in hkeys])
        naive = np.array([all_naive[s][k] for k in hkeys])
        mae_naive = np.mean(np.abs(naive - obs)) + 1e-12
        mase_mean = np.mean(np.abs(mu - obs)) / mae_naive
        mase_median = np.mean(np.abs(med - obs)) / mae_naive
        cov50 = np.mean([(quantile_store[s][k][0.25] <= all_obs[s][k[1]] <= quantile_store[s][k][0.75])
                         for k in hkeys])
        cov90 = np.mean([(quantile_store[s][k][0.1] <= all_obs[s][k[1]] <= quantile_store[s][k][0.9])
                         for k in hkeys])
        cov95 = np.mean([(quantile_store[s][k][0.025] <= all_obs[s][k[1]] <= quantile_store[s][k][0.975])
                         for k in hkeys])
        wis_h = np.mean([wis(all_obs[s][k[1]], quantile_store[s][k]) for k in hkeys]) * N_val
        sd_epi = np.mean([ens_sd_epi[s][k] for k in hkeys]) * N_val
        sd_alea = np.mean([ens_sd_alea[s][k] for k in hkeys]) * N_val
        ### mean credible-interval width, in counts - the sharpness half of the
        ### calibration/sharpness pair, so coverage can be read against interval size
        def mean_width(lo_q, hi_q):
            return float(np.mean([quantile_store[s][k][hi_q] - quantile_store[s][k][lo_q]
                                 for k in hkeys])) * N_val
        w50 = mean_width(0.25, 0.75)
        w90 = mean_width(0.1, 0.9)
        w95 = mean_width(0.025, 0.975)
        ### widths relative to the mean observation, so series are comparable
        obs_mean = float(np.mean(obs)) * N_val + 1e-12
        rows.append({
            "series": s, "horizon": h, "n_forecasts": len(hkeys),
            "mase_posterior_mean": mase_mean, "mase_posterior_median": mase_median,
            "coverage_50": cov50, "coverage_90": cov90, "coverage_95": cov95,
            "wis": wis_h, "mean_sd_epistemic": sd_epi, "mean_sd_aleatoric": sd_alea,
            "width_50": w50, "width_90": w90, "width_95": w95,
            "rel_width_50": w50 / obs_mean, "rel_width_90": w90 / obs_mean,
            "rel_width_95": w95 / obs_mean, "mean_obs": obs_mean,
        })
        print(f"  {s:7s} h={h} | MASE mean={mase_mean:.3f} median={mase_median:.3f} "
              f"| cov50={cov50:.2f} cov90={cov90:.2f} cov95={cov95:.2f} "
              f"| width50={w50:.3e} width90={w90:.3e} ({w90 / obs_mean:.2f}x mean obs) "
              f"| WIS={wis_h:.3e}")

metrics = pd.DataFrame(rows)
metrics.to_csv("variational_pinn_metrics.csv", index=False)
print("\nmetrics written to variational_pinn_metrics.csv")

### helper: pull a horizon's forecast + credible interval as sorted arrays, in count units
def horizon_arrays(s, h):
    hkeys = sorted([k for k in keys if k[1] - k[0] == h], key=lambda k: k[1])
    idx = np.array([k[1] for k in hkeys], dtype=int)
    mu = np.array([all_pred[s][k] for k in hkeys]) * N_val
    obs = np.array([all_obs[s][k[1]] for k in hkeys]) * N_val
    lo50 = np.array([quantile_store[s][k][0.25] for k in hkeys]) * N_val
    hi50 = np.array([quantile_store[s][k][0.75] for k in hkeys]) * N_val
    lo90 = np.array([quantile_store[s][k][0.1] for k in hkeys]) * N_val
    hi90 = np.array([quantile_store[s][k][0.9] for k in hkeys]) * N_val
    dr = np.array([f_draws[s][k] for k in hkeys]).T * N_val          ### (N_MC, n_times)
    return study_dates[idx], mu, obs, lo50, hi50, lo90, hi90, dr

### Plot 1: fan chart, one row per series, 1-week-ahead
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, SERIES):
    d, mu, obs, lo50, hi50, lo90, hi90, _ = horizon_arrays(s, 1)
    ax.fill_between(d, lo90, hi90, color="#7397de", alpha=0.30, label="80% CrI")
    ax.fill_between(d, lo50, hi50, color="#7397de", alpha=0.55, label="50% CrI")
    ax.plot(d, mu, color="#1f3d99", lw=1.8, label="Posterior mean")
    ax.plot(d, obs, color="#004F94", lw=1.3, ls=":", label="Observed")
    ax.set_ylabel(LABEL[s]); ax.legend(fontsize=9, loc="upper right"); apply_date_axis(ax)
axes[0].set_title("England: variational Bayesian SEIR-PINN, 1-week-ahead with credible intervals")
plt.tight_layout(); plt.savefig("variational_pinn_fan_h1.png", dpi=150); plt.show()

### Plot 2: cases, 2x2 grid over horizons
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for h, ax in zip(range(1, Forecast_horizon + 1), axes.flatten()):
    d, mu, obs, lo50, hi50, lo90, hi90, _ = horizon_arrays("cases", h)
    ax.fill_between(d, lo90, hi90, color="#7397de", alpha=0.30, label="80% CrI")
    ax.fill_between(d, lo50, hi50, color="#7397de", alpha=0.55, label="50% CrI")
    ax.plot(d, mu, color="#1f3d99", lw=1.6, label=f"Posterior mean {h}-week")
    ax.plot(d, obs, color="#004F94", lw=1.4, label="Observed")
    ax.set_title(f"{h}-week-ahead forecast")
    ax.set_ylabel(LABEL["cases"]); ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3); apply_date_axis(ax)
fig.suptitle("Variational Bayesian SEIR-PINN: cases, 1-4 weeks ahead", fontsize=14)
plt.tight_layout(); plt.savefig("variational_pinn_cases_grid.png", dpi=150); plt.show()

### Plot 3: a thinned set of posterior draws - the Bayesian analogue of the
### deep ensemble's member trajectories
N_SHOW = 50
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, SERIES):
    d, mu, obs, *_, dr = horizon_arrays(s, 1)
    show = dr[:N_SHOW] if dr.shape[0] > N_SHOW else dr
    for i in range(show.shape[0]):
        ax.plot(d, show[i], color="#7397de", lw=0.7, alpha=0.30,
                label=f"posterior draws (n={show.shape[0]} of {dr.shape[0]})" if i == 0 else None)
    ax.plot(d, mu, color="#1f3d99", lw=2.0, label="Posterior mean")
    ax.plot(d, obs, color="#004F94", lw=1.3, ls=":", label="Observed")
    ax.set_ylabel(LABEL[s]); ax.legend(fontsize=9, loc="upper right"); apply_date_axis(ax)
axes[0].set_title("England: posterior draws from the variational SEIR-PINN, 1-week-ahead")
plt.tight_layout(); plt.savefig("variational_pinn_draws_h1.png", dpi=150); plt.show()

### Plot 4: calibration - nominal vs empirical coverage
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
nominal = [0.5, 0.9, 0.95]
x = np.arange(Forecast_horizon)
width = 0.8 / len(nominal)
for ax, s in zip(axes, SERIES):
    sub = metrics[metrics["series"] == s].sort_values("horizon")
    for i, (lvl, col) in enumerate(zip(nominal, ["coverage_50", "coverage_90", "coverage_95"])):
        ax.bar(x + i * width - 0.4 + width / 2, sub[col].values, width,
               label=f"{int(lvl * 100)}% CrI")
        ax.axhline(lvl, color="grey", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"{h}wk" for h in range(1, Forecast_horizon + 1)])
    ax.set_title(LABEL[s]); ax.set_xlabel("Horizon"); ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
axes[0].set_ylabel("Empirical coverage")
axes[0].legend(fontsize=9, loc="lower left")
fig.suptitle("Variational Bayesian PINN interval calibration (dashed = nominal level)", fontsize=14)
plt.tight_layout(); plt.savefig("variational_pinn_coverage.png", dpi=150); plt.show()

### Plot 5: variance decomposition - posterior weight uncertainty vs observation noise
fig, ax = plt.subplots(figsize=(12, 5))
sub = metrics[metrics["series"] == "cases"].sort_values("horizon")
ax.bar(sub["horizon"] - 0.2, sub["mean_sd_epistemic"], 0.4, color="#1f3d99", label="Epistemic (posterior over weights)")
ax.bar(sub["horizon"] + 0.2, sub["mean_sd_aleatoric"], 0.4, color="#f1c40f", label="Aleatoric (observation noise)")
ax.set_xticks(list(range(1, Forecast_horizon + 1)))
ax.set_xlabel("Horizon (weeks ahead)"); ax.set_ylabel("Mean predictive sd (cases per week)")
ax.set_title("Variational Bayesian PINN uncertainty decomposition, cases")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("variational_pinn_variance_split.png", dpi=150); plt.show()

### Plot 6: R(t) with posterior credible band
fig, ax = plt.subplots(figsize=(12, 5))
h1 = sorted([k for k in keys if k[1] - k[0] == 1], key=lambda k: k[1])
rt_d = [study_dates[k[1]] for k in h1]
rt_draws = np.array([beta_draws_store[k] for k in h1]).T / GAMMA     ### (N_MC, n_times)
rt_mean = rt_draws.mean(axis=0)
rt_lo = np.quantile(rt_draws, 0.025, axis=0)
rt_hi = np.quantile(rt_draws, 0.975, axis=0)
ax.fill_between(rt_d, rt_lo, rt_hi, color="#7397de", alpha=0.35, label="95% credible interval")
ax.plot(rt_d, rt_mean, color="#1f3d99", lw=1.5, label="R(t)=beta/gamma (posterior mean)")
ax.axhline(1.0, color="gray", lw=1, ls="--", label="R=1")
ax.set_ylabel("R(t)"); ax.legend(); apply_date_axis(ax)
ax.set_title("England: variational Bayesian SEIR-PINN effective reproduction number")
plt.tight_layout(); plt.savefig("variational_pinn_Rt.png", dpi=150); plt.show()

### Save
### all_pred/all_obs/all_naive keep the PINN.pkl layout so compare_forecasts.py can
### read this file directly; the extra keys carry the posterior information
with open(DUMP_PATH, "wb") as f:
    pickle.dump({
        "all_pred": all_pred,          ### posterior predictive mean point forecast
        "all_obs": all_obs,
        "all_naive": all_naive,
        "f_draws": f_draws,            ### key -> (N_MC,) draws of the mean function
        "sigma_obs": sigma_obs,        ### key -> observation-noise sd, data units
        "quantiles": quantile_store,   ### key -> {quantile level: value}
        "sd_total": ens_sd,
        "sd_epistemic": ens_sd_epi,
        "sd_aleatoric": ens_sd_alea,
        "all_beta": all_beta,
        "beta_draws": beta_draws_store,
        "metrics": metrics,
        "config": {"n_iter": N_ITER, "n_mc": N_MC, "kl_max": KL_MAX,
                   "kl_ramp_iters": KL_RAMP_ITERS,
                   "sigma_bounds": (SIGMA_MIN, SIGMA_MAX),
                   "first_train_weeks": First_train_weeks,
                   "forecast_horizon": Forecast_horizon, "quantiles": QUANTILES},
    }, f)
print(f"\nsaved variational Bayesian PINN to {DUMP_PATH}")
