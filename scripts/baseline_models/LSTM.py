import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

### LSTM model for 1, 2, 3 and 4 week predictions
### forecasting cases, deaths and hospitalisations
### conventional baseline: one model trained per origin, multi-step via recursive inference
### https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-networks-using-pytorch/
### https://gist.github.com/Lexie88rus/8ab37c8ea8c9f92b0efbca3c584bf063
### https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html


### set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### plot style — fixed across all scripts
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

colours = {
    "S": "#2ca02c",
    "E": "#ff7f0e",
    "I": "#d62728",
    "R": "#1f33b4",
}

### load in data
n_val = 56_000_000
i_data = np.load("../../data/I_data_study.npy").reshape(-1).astype("float64")
h_data = np.load("../../data/H_data_study.npy").reshape(-1).astype("float64")
d_data = np.load("../../data/D_data_study.npy").reshape(-1).astype("float64")
study_dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")
n_total_points = len(i_data)
study_dates = study_dates[:n_total_points]

series = ["cases", "deaths", "hosp"]
data = {"cases": i_data, "deaths": d_data, "hosp": h_data}
label = {"cases": "daily cases", "hosp": "daily admissions", "deaths": "daily deaths"}

### configure lstm
first_train_weeks = 17
forecast_horizon = 4
window_size = 5
hidden, layers = 50, 2
epochs = 10000
patience = 500
learning_rate = 1e-3
leak_like_paper = False

print(f"study grid: {n_total_points} weeks ({study_dates[0]} -> {study_dates[-1]}) | device={device}")


class rnn_predictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=hidden, num_layers=layers, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def fit_scalers(hist3, leak_full=None):
    src = leak_full if (leak_like_paper and leak_full is not None) else hist3
    scalers = []
    for c in range(3):
        s = StandardScaler()
        s.fit(np.clip(src[:, c], 0, None).reshape(-1, 1))
        scalers.append(s)
    return scalers


def standardise(hist3, scalers):
    return np.column_stack([
        scalers[c].transform(np.clip(hist3[:, c], 0, None).reshape(-1, 1)).ravel()
        for c in range(3)
    ])


def train_one_step(scaled):
    x_seq, y_seq = [], []
    for i in range(len(scaled) - window_size):
        x_seq.append(scaled[i:i + window_size, :])
        y_seq.append(scaled[i + window_size, :])
    x_seq = torch.tensor(np.asarray(x_seq), dtype=torch.float32, device=device)
    y_seq = torch.tensor(np.asarray(y_seq), dtype=torch.float32, device=device)

    model = rnn_predictor().to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    crit = nn.MSELoss()
    best, best_loss, since = None, np.inf, 0
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = crit(model(x_seq), y_seq)
        loss.backward()
        opt.step()
        lv = float(loss)
        if patience is not None:
            if lv < best_loss - 1e-9:
                best_loss, since, best = lv, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= patience:
                    break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return model


def predict_step(model, window):
    with torch.no_grad():
        x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        return model(x).cpu().numpy().ravel()


def inverse(pred_scaled, scalers):
    raw = np.array([scalers[c].inverse_transform([[pred_scaled[c]]])[0, 0] for c in range(3)])
    return np.clip(raw, 1e-6, None)


### rolling forecast (train once per origin, recursive inference for horizons 1-4)
all_pred  = {s: {} for s in series}
all_naive = {s: {} for s in series}
all_obs   = {s: {} for s in series}

full3 = np.column_stack([data["cases"], data["deaths"], data["hosp"]])

for train_end in range(first_train_weeks, n_total_points - forecast_horizon + 1):
    origin = train_end - 1
    hist = full3[:train_end].copy()
    scalers = fit_scalers(hist, leak_full=full3)
    scaled = standardise(hist, scalers)
    model = train_one_step(scaled)

    window = scaled[-window_size:].copy()
    for h in range(1, forecast_horizon + 1):
        fidx = origin + h
        if fidx >= n_total_points:
            break
        pred_scaled = predict_step(model, window)
        pred = inverse(pred_scaled, scalers)
        for ci, s in enumerate(series):
            all_pred[s][(origin, fidx)]  = float(pred[ci])
            all_naive[s][(origin, fidx)] = float(full3[origin, ci])
            all_obs[s][fidx]             = float(full3[fidx, ci])
        window = np.vstack([window[1:], pred_scaled])
    print(f"origin {origin:3d} (train weeks 1-{train_end}) done | "
          f"1wk cases={all_pred['cases'][(origin, origin+1)]*n_val:,.0f}")

### collect / mase / plots
def collect(s_name):
    hres = {h: {"idx": [], "pred": [], "obs": [], "naive": []} for h in range(1, 5)}
    for (te, fidx), pv in all_pred[s_name].items():
        h = fidx - te
        hres[h]["idx"].append(fidx); hres[h]["pred"].append(pv)
        hres[h]["obs"].append(all_obs[s_name][fidx]); hres[h]["naive"].append(all_naive[s_name][(te, fidx)])
    return hres
 
def apply_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right')
 
print("\nlstm mase by horizon (model mae / naive mae; <1 beats naive):")
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for ax, s in zip(axes, series):
    hres = collect(s)
    ax.plot(study_dates, data[s] * n_val, color="#004F94", lw=1.5, ls=":", label="observed")
    for h in range(1, 5):
        idx = np.array(hres[h]["idx"], dtype=int)
        order = np.argsort(idx)
        ax.plot(study_dates[idx[order]], np.array(hres[h]["pred"])[order] * n_val,
                lw=1.2, alpha=0.85, label=f"{h} wk ahead")
    ax.set_ylabel(label[s]); ax.legend(fontsize=8, loc="upper right"); apply_date_axis(ax)
    line = f"  {s:8s}"
    for h in range(1, 5):
        p = np.array(hres[h]["pred"]); o = np.array(hres[h]["obs"]); n = np.array(hres[h]["naive"])
        mase = np.mean(np.abs(p - o)) / (np.mean(np.abs(n - o)) + 1e-12)
        line += f"  {h}wk={mase:.3f}"
    print(line)
axes[0].set_title("england: lstm per-horizon vs observed")
plt.tight_layout(); plt.savefig("lstm_perhorizon.png", dpi=150); plt.show()

### per-horizon 2x2 grid for cases
plot_series, label_str = "cases", "new cases per week"
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
for h, ax in zip(range(1, 5), axes.flatten()):
    hd, hp, ho, hn = [], [], [], []
    for (te, fidx), pv in all_pred[plot_series].items():
        if fidx - te == h:
            hd.append(study_dates[fidx]); hp.append(pv)
            ho.append(all_obs[plot_series][fidx]); hn.append(all_naive[plot_series][(te, fidx)])
    if hd:
        order = np.argsort(np.array(hd, dtype='datetime64[D]'))
        hd = np.array(hd, dtype='datetime64[D]')[order]
        hp = np.array(hp)[order] * n_val; ho = np.array(ho)[order] * n_val; hn = np.array(hn)[order] * n_val
        ax.plot(hd, ho, color="#004F94", lw=1.5, label="observed")
        ax.plot(hd, hp, color="#23e623", lw=1.5, label=f"lstm {h}-week")
        ax.plot(hd, hn, color="orange", lw=1.0, ls="--", label="naive baseline")
    ax.set_title(f"{h}-week-ahead forecast"); ax.set_ylabel(label_str)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3); apply_date_axis(ax)
fig.suptitle("lstm: cases (1-4 weeks ahead)", fontsize=14)
plt.tight_layout(); plt.savefig("lstm_cases_grid.png", dpi=150); plt.show()

### Save
import os, pickle
os.makedirs("../../forecasts", exist_ok=True)
with open("../../forecasts/preds_lstm.pkl", "wb") as f:
    pickle.dump({"all_pred": all_pred, "all_obs": all_obs, "all_naive": all_naive}, f)