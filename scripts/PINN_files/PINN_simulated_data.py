import argparse
import os
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

### SEIR-PINN for synthetic data - infers beta(t) from observed I(t), sigma and gamma provided
### (PINN_all_learnable_parameters.py learns sigma and gamma too)

### Data loss on log scale -> early outbreak is not ignored
### ODE residual measured relative to each compartment's size -> physics enforced across the whole epidemic
### Initial conditions and S + E + I + R = 1 built into the outputs 
### ADAM + L-BFGS 

N_VAL = 100001
N = tf.constant(float(N_VAL), dtype=tf.float32)
SIGMA, GAMMA = 0.25, 0.25                 ### used when sigma and gamma are provided, not learned
T_DAYS = 100.0                             ### t is normalised to [0, 1] over 100 days
N_COL = 1000                               ### collocation points on [0, 1]
MAX_POINTS = 101
I0_FIXED = 1.0 / N_VAL                     ### one infected person at t = 0
LOG_EPS = 0.1 / N_VAL                      ### a tenth of a person, keeps log(I) finite

def create_pinn_model():
    """Outputs satisfy the initial conditions and conservation exactly:
        log I(t) = log I0 + t * a(t)                   I(0) = I0, and I is represented on a log scale
        E(t) = t * exp(b(t)),  R(t) = t * exp(c(t))     E(0) = R(0) = 0, both always positive
        S(t) = 1 - E - I - R                            S + E + I + R = 1

    The b and c output biases start at log(1/N) so E and R begin at the scale of a few people.
    beta(t) comes from a separate sub-network with a softplus output, so it stays positive."""

    t_input = Input(shape=(1,), name='time_input')

    seir = Dense(50, activation='tanh')(t_input)
    seir = Dense(50, activation='tanh')(seir)
    seir = Dense(50, activation='tanh')(seir)

    a = Dense(1, name='log_growth_I')(seir)
    b = Dense(1, bias_initializer=tf.keras.initializers.Constant(np.log(1.0 / N_VAL)), name='log_E')(seir)
    c = Dense(1, bias_initializer=tf.keras.initializers.Constant(np.log(1.0 / N_VAL)), name='log_R')(seir)

    I = Lambda(lambda z: tf.exp(np.log(I0_FIXED) + z[0] * z[1]), name='I')([t_input, a])
    E = Lambda(lambda z: z[0] * tf.exp(z[1]), name='E')([t_input, b])
    R = Lambda(lambda z: z[0] * tf.exp(z[1]), name='R')([t_input, c])
    S = Lambda(lambda z: 1.0 - z[0] - z[1] - z[2], name='S')([E, I, R])

    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(t_input)
    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_h)
    beta_h = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(1e-5))(beta_h)

    beta = Dense(1, activation='softplus', name='beta')(beta_h)

    return Model(inputs=t_input, outputs=[S, E, I, R, beta])


def loss_function(t_col, t_data, I_data, net, sigma=SIGMA, gamma=GAMMA, beta_const=None):
    """data: MSE on log(I + eps)
    ODE:  each compartment's residual divided by T * (compartment count + 1 person), i.e. a relative
          rate error, summed over S, E, I, R
    beta_const: if given, one learnable constant replaces the beta network"""
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t_col)
        S, E, I, R, beta = net(t_col)
    if beta_const is not None:
        beta = beta_const * tf.ones_like(beta)
    dS_dt = tape.gradient(S, t_col); dE_dt = tape.gradient(E, t_col)
    dI_dt = tape.gradient(I, t_col); dR_dt = tape.gradient(R, t_col)
    del tape

    S_abs, E_abs, I_abs, R_abs = S * N, E * N, I * N, R * N

    dS_phys = T_DAYS * (-beta * S_abs * I_abs / N)
    dE_phys = T_DAYS * (beta * S_abs * I_abs / N - sigma * E_abs)
    dI_phys = T_DAYS * (sigma * E_abs - gamma * I_abs)
    dR_phys = T_DAYS * (gamma * I_abs)

    rel = lambda d, phys, X_abs: tf.square((d * N - phys) / (T_DAYS * (X_abs + 1.0)))

    ode_loss = (tf.reduce_mean(rel(dS_dt, dS_phys, S_abs)) + tf.reduce_mean(rel(dE_dt, dE_phys, E_abs)) +
                tf.reduce_mean(rel(dI_dt, dI_phys, I_abs)) + tf.reduce_mean(rel(dR_dt, dR_phys, R_abs)))

    _, _, I_pred, _, _ = net(t_data)
    data_loss = tf.reduce_mean(tf.square(tf.math.log(I_pred + LOG_EPS) - tf.math.log(I_data + LOG_EPS)))

    total = data_loss + ode_loss
    return total, {"data_loss": data_loss, "ODE_loss": ode_loss}


class Fitter:
    def __init__(self, lr=1e-3, learn_sg=False, sg_init=0.5, beta_const=False, beta_init=1.0):
        """learn_sg: sigma and gamma become two learnable constants (on a log scale, so they stay
        positive), started at sg_init; otherwise they are fixed at SIGMA, GAMMA.
        beta_const: beta is one learnable constant (started at beta_init) instead of a network of time."""
        self.learn_sg, self.sg_init = learn_sg, sg_init
        self.beta_const, self.beta_init = beta_const, beta_init
        self.log_sg = tf.Variable(np.log([sg_init, sg_init]).astype(np.float32), name="log_sigma_gamma")
        self.log_beta = tf.Variable(np.float32(np.log(beta_init)), name="log_beta_const")
        self.model = create_pinn_model()
        self.t = tf.Variable(tf.zeros((MAX_POINTS, 1)))
        self.y = tf.Variable(tf.zeros((MAX_POINTS, 1)))
        self.n = tf.Variable(1, dtype=tf.int32)
        self.opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        self.opt._create_all_weights(self.train_vars())
        self.t_col = tf.constant(np.linspace(0, 1, N_COL).reshape(-1, 1), dtype=tf.float32)

    ### task and parameters 
    def set_task(self, t, I):
        """Train on the observations given (e.g. all 101 days, or days up to a forecast origin)"""
        n = len(t); pad = MAX_POINTS - n
        self.t.assign(np.pad(np.asarray(t, np.float32).reshape(-1, 1), ((0, pad), (0, 0))))
        self.y.assign(np.pad(np.asarray(I, np.float32).reshape(-1, 1), ((0, pad), (0, 0))))
        self.n.assign(n)

    def train_vars(self):
        return (self.model.trainable_variables + ([self.log_sg] if self.learn_sg else []) +
                ([self.log_beta] if self.beta_const else []))

    def sg(self):
        if self.learn_sg:
            return tf.exp(self.log_sg[0]), tf.exp(self.log_sg[1])
        return tf.constant(SIGMA), tf.constant(GAMMA)

    def bconst(self):
        return tf.exp(self.log_beta) if self.beta_const else None

    def get_state(self):
        """Everything that is learned: network weights and the scalar parameters."""
        return (self.model.get_weights(), self.log_sg.numpy().copy(), self.log_beta.numpy().copy())

    def set_state(self, state):
        self.model.set_weights(state[0]); self.log_sg.assign(state[1]); self.log_beta.assign(state[2])

    def reset(self, weights):
        """Start a new fit from network weights `weights`, scalar parameters at their initial guesses
        and a fresh optimiser."""
        self.model.set_weights(weights)
        self.log_sg.assign(np.log([self.sg_init, self.sg_init]).astype(np.float32))
        self.log_beta.assign(np.float32(np.log(self.beta_init)))
        self.reset_optimiser()

    def reset_optimiser(self):
        for v in self.opt.variables():
            v.assign(tf.zeros_like(v))

    ### training 
    @tf.function
    def step(self):
        n = self.n
        with tf.GradientTape() as tape:
            sigma, gamma = self.sg()
            loss, ld = loss_function(self.t_col, self.t[:n], self.y[:n], self.model, sigma, gamma, self.bconst())
        grads = tape.gradient(loss, self.train_vars())
        grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, self.train_vars())]
        self.opt.apply_gradients(zip(grads, self.train_vars()))
        return loss, ld

    def run(self, n_steps):
        for _ in range(n_steps):
            loss, ld = self.step()
        return float(loss)

    def run_keep_best(self, n_steps, best=None, check_every=1000):
        """Adam steps, checking the loss every check_every steps and keeping the best state seen.
        best = (loss, state) or None."""
        done = 0
        while done < n_steps:
            k = min(check_every, n_steps - done)
            loss = self.run(k); done += k
            if best is None or loss < best[0]:
                best = (loss, self.get_state())
        return best

    def lbfgs(self, maxiter=5000):
        """Polish the current state with L-BFGS (scipy). Reverts if it does not lower the loss.
        Returns (loss before, loss after)."""
        vs = self.train_vars()
        shapes = [v.shape for v in vs]; sizes = [int(np.prod(sh)) for sh in shapes]
        if not hasattr(self, "_vg"):
            @tf.function
            def vg(flat):
                i = 0
                for v, sh, k in zip(vs, shapes, sizes):
                    v.assign(tf.reshape(flat[i:i + k], sh)); i += k
                n = self.n
                with tf.GradientTape() as tape:
                    sigma, gamma = self.sg()
                    loss, _ = loss_function(self.t_col, self.t[:n], self.y[:n], self.model,
                                            sigma, gamma, self.bconst())
                g = tape.gradient(loss, vs)
                g = [x if x is not None else tf.zeros_like(v) for x, v in zip(g, vs)]
                return loss, tf.concat([tf.reshape(x, [-1]) for x in g], 0)
            self._vg = vg

        def f(x):
            loss, g = self._vg(tf.constant(x, tf.float32))
            return float(loss), g.numpy().astype(np.float64)

        before_state = self.get_state()
        x0 = np.concatenate([v.numpy().ravel() for v in vs]).astype(np.float64)
        before = f(x0)[0]
        res = scipy.optimize.minimize(f, x0, jac=True, method="L-BFGS-B",
                                      options={"maxiter": maxiter, "maxfun": 2 * maxiter,
                                               "ftol": 1e-15, "gtol": 1e-12})
        after = f(res.x)[0]                       ### also leaves the variables at res.x
        if not np.isfinite(after) or after > before:
            self.set_state(before_state); after = before
        return before, after

    def fit(self, adam_steps=50_000, lbfgs_iters=5000):
        """Full fit from the current state: Adam keeping the best state, then L-BFGS. Returns the loss."""
        best = self.run_keep_best(adam_steps)
        self.set_state(best[1])
        if lbfgs_iters:
            _, loss = self.lbfgs(lbfgs_iters)
            return loss
        return best[0]

    ### outputs
    def predict(self, t):
        S, E, I, R, beta = self.model(tf.constant(np.asarray(t, np.float32).reshape(-1, 1)))
        sigma, gamma = self.sg()
        if self.beta_const:
            beta = float(tf.exp(self.log_beta)) * tf.ones_like(beta)
        return {"S": S.numpy().ravel(), "E": E.numpy().ravel(), "I": I.numpy().ravel(),
                "R": R.numpy().ravel(), "beta": beta.numpy().ravel(),
                "sigma": float(sigma), "gamma": float(gamma)}


### running the synthetic scenarios
DEFAULT_SCENARIOS = (["beta_0.75", "beta_0.5", "beta_0.4"] +
                     ["beta_spline", "beta_spline_slow_rise", "beta_spline_gradual_decline",
                      "beta_spline_single_pulse", "beta_spline_two_waves", "beta_spline_three_waves",
                      "beta_spline_rapid", "beta_spline_very_rapid", "beta_spline_escalating",
                      "beta_spline_damped"] +
                     ["ood_piecewise", "ood_exp_decay", "noise_5", "noise_10", "noise_15", "noise_20"])

COLOURS = {"S": "#2ca02c", "E": "#ff7f0e", "I": "#d62728", "R": "#1f33b4"}


def report_label(label):
    """Scenario names as used in panel_figures.py: noise scenarios are Gaussian_noise_{p}percent."""
    return f"Gaussian_noise_{label.split('_')[1]}percent" if label.startswith("noise_") else label


def split_suffix(split, learn_sg):
    """File suffix used in panel_figures.py: 80_20 for an 80/20 split; all 100 days is 100_10 with sigma
    and gamma provided and 100_0 with them learned."""
    if abs(split - 0.8) < 1e-9:
        return "80_20"
    if abs(split - 1.0) < 1e-9:
        return "100_0" if learn_sg else "100_10"
    return f"{round(100 * split)}_{round(100 * (1 - split))}"


def run_scenarios(scenarios, split=1.0, seeds=1, adam_steps=50_000, lbfgs_iters=5000,
                  learn_sg=False, beta_const=False, prefix="PINN", pred_prefix="PINN",
                  data_folder="../../data", output_dir="../../png_files"):
    """Fit each scenario, save I(t), beta(t) and S/E/R plots, predictions and a metrics table."""
    import synthetic_scenarios as sc
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12, 'axes.labelsize': 13,
                         'axes.titlesize': 14, 'legend.fontsize': 11, 'lines.linewidth': 2,
                         'axes.grid': True, 'grid.alpha': 0.3})
    os.makedirs(output_dir, exist_ok=True)
    fit = Fitter(learn_sg=learn_sg, beta_const=beta_const)
    suffix = split_suffix(split, learn_sg)
    metrics = []
    for label in scenarios:
        task = sc.load_task(label, data_folder)
        t_days = task["t"] * T_DAYS
        n_train = int(round(split * len(task["t"])))
        for seed in range(seeds):
            tag = report_label(label) if seeds == 1 else f"{report_label(label)}_seed{seed}"
            print(f"\n{'=' * 50}\nRunning PINN for scenario: {tag}\n{'=' * 50}", flush=True)
            fit.set_task(task["t"][:n_train], task["I"][:n_train])
            tf.random.set_seed(seed)
            fit.reset(create_pinn_model().get_weights())
            loss = fit.fit(adam_steps, lbfgs_iters)
            out = fit.predict(task["t"])

            ape = np.abs(out["beta"] - task["beta"]) / task["beta"]
            row = {"scenario": tag, "loss": loss, "beta_MAPE_pct": 100 * ape.mean(),
                   "sigma": out["sigma"], "gamma": out["gamma"]}
            for k in ("S", "E", "I", "R"):
                if k in task:
                    row[f"{k}_MAE_pct"] = 100 * np.mean(np.abs(out[k] - task[k])) / np.mean(task[k])
            if n_train < len(task["t"]):
                obs = task["I"][n_train:]
                row["I_forecast_MAE_pct"] = 100 * np.mean(np.abs(out["I"][n_train:] - obs)) / np.mean(obs)
            metrics.append(row)
            print(f"loss {loss:.2e} | beta MAPE {row['beta_MAPE_pct']:.2f}% | sigma {out['sigma']:.3f} "
                  f"gamma {out['gamma']:.3f}", flush=True)

            plt.figure(figsize=(14, 6))
            plt.plot(t_days, out["I"] * N_VAL, color="#444444", label="Infected - PINN")
            plt.scatter(t_days, task["I"] * N_VAL, color=COLOURS["I"], s=10, alpha=0.5, zorder=3, label="Infected - data")
            if n_train < len(task["t"]):
                plt.axvline(t_days[n_train - 1], color="gray", ls="--", label="Train/Test Split")
            plt.xlabel("Days"); plt.ylabel("Number of infected individuals"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_I_{tag}_{suffix}.png")); plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(t_days, out["beta"], color="#444444",
                     label="β(t) estimated" + (f" (σ={out['sigma']:.3f}, γ={out['gamma']:.3f})" if learn_sg else ""))
            plt.scatter(t_days, task["beta"], color="#1f33b4", s=30, alpha=0.5, zorder=3, label="β(t) true")
            plt.xlabel("Days"); plt.ylabel("β(t)"); plt.ylim(0, max(1.6, 1.3 * task["beta"].max()))
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_parameter_est_beta_{tag}_{suffix}.png")); plt.close()

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, k, name in zip(axes, ("S", "E", "R"), ("Susceptible", "Exposed", "Recovered")):
                ax.plot(t_days, out[k] * N_VAL, color="#444444", label=f"{name} (PINN)")
                if k in task:
                    ax.scatter(t_days, task[k] * N_VAL, color=COLOURS[k], s=10, alpha=0.5, zorder=3,
                               label=f"{name} (ground truth)")
                ax.set_xlabel("Days"); ax.set_ylabel("Number of individuals"); ax.legend()
            plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{prefix}_SER_{tag}_{suffix}.png")); plt.close()

            pd.DataFrame({"t": task["t"], **{f"{k}_pred": out[k] for k in ("S", "E", "I", "R", "beta")}}).to_csv(
                os.path.join(output_dir, f"{pred_prefix}_predictions_{tag}_{suffix}.csv"), index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, f"{prefix}_error_metrics_{suffix}.csv"), index=False)
    print("\n" + metrics_df.round(3).to_string(index=False))
    return metrics_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    ap.add_argument("--split", type=float, default=1.0, help="fraction of days used for training")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--adam-steps", type=int, default=50_000)
    ap.add_argument("--lbfgs", type=int, default=5000)
    args = ap.parse_args()
    run_scenarios(args.scenarios, args.split, args.seeds, args.adam_steps, args.lbfgs)
