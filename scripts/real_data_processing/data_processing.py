import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Load data
data = pd.read_csv("../../Data/Cases/nation_newCasesBySpecimenDate.csv")

### Filter for England
data = data[data['area_name'] == 'England']

### Keep only relevant columns
data = data[['date', 'value']].rename(columns={'value': 'newCases'})

### Convert dates
data['date'] = pd.to_datetime(data['date'])

### Sort chronologically
data = data.sort_values('date').reset_index(drop=True)

### 7-day centred smoothing (ONLY smoothing step kept)
data['newCases_smooth'] = (
    data['newCases']
    .rolling(7, center=True, min_periods=1)
    .mean()
)

### Convert directly to population fraction (NO rolling sum)
N = 56_000_000
infectious_days = 4  # 1/gamma = 1/0.25
data['I_obs'] = (data['newCases_smooth'] / N)

### Define study period
mask = (
    (data['date'] >= '2020-08-01') &  
    (data['date'] <= '2022-04-30')
)

data_study = data[mask].copy().reset_index(drop=True)

### Weekly aggregation
data_weekly = (
    data_study
    .set_index('date')
    .resample('W-SAT', label='left', closed='left')  # or 'W-MON'
    .mean(numeric_only=True)
    .reset_index()
    .dropna(subset=['I_obs'])
)

### Normalised time over study period
t_days_study = (data_weekly['date'] - data_weekly['date'].min()).dt.days

t_norm_study = (
    (t_days_study - t_days_study.min()) /
    (t_days_study.max() - t_days_study.min())
)

### Save study arrays (for PINN)
np.save(
    "../../data/t_data_study.npy",
    t_norm_study.values.reshape(-1, 1)
)

np.save(
    "../../data/I_data_study.npy",
    data_weekly['I_obs'].values.reshape(-1, 1)
)

np.save("../../data/dates_study.npy",
        data_weekly['date'].values.astype('datetime64[D]'))

### Collocation points
np.save(
    "../../data/t_col_study.npy",
    np.random.uniform(0, 1, (2000, 1))
)

### Print diagnostics
print(f"\nStudy period (weekly): {len(data_weekly)} points "
      f"({data_weekly['date'].min().date()} → {data_weekly['date'].max().date()})")

print(f"t_data_study range : {t_norm_study.min():.4f} → {t_norm_study.max():.4f}")
print(f"I_data_study range : {data_weekly['I_obs'].min():.6f} → {data_weekly['I_obs'].max():.6f}")

### Yearly datasets (FIXED)
years = {y: df for y, df in data.groupby(data['date'].dt.year)}

print(f"\nCreated {len(years)} yearly datasets.")

for y, df in years.items():
    y_str = str(y)

    df_weekly = (
        df.set_index('date')
        .resample('W')
        .sum(numeric_only=True)
        .reset_index()
        .dropna(subset=['I_obs'])
    )

    t_week = (df_weekly['date'] - df_weekly['date'].min()).dt.days

    t_norm = (
        (t_week - t_week.min()) /
        (t_week.max() - t_week.min())
    )

    np.save(f"../../data/t_data_{y_str}.npy", t_norm.values.reshape(-1, 1))
    np.save(f"../../data/I_data_{y_str}.npy", df_weekly['I_obs'].values.reshape(-1, 1))

    np.save(f"../../data/t_data_raw_{y_str}.npy", t_week.values)
    np.save(f"../../data/I_data_raw_{y_str}.npy", df_weekly['I_obs'].values)

    print(f"  {y_str}: {len(df_weekly)} weekly points saved")

### Verification
t_study = np.load("../../data/t_data_study.npy")
I_study = np.load("../../data/I_data_study.npy")

print(f"\nt_data_study : {len(t_study)} points, "
      f"range {t_study.min():.4f} → {t_study.max():.4f}")

print(f"I_data_study : min {I_study.min():.6f}, max {I_study.max():.6f}")

if len(t_study) < 85 or len(t_study) > 100:
    print("WARNING: unexpected number of weekly points.")
else:
    print(f"OK: {len(t_study)} weekly points (expected ~93).")

### Plot
plt.figure(figsize=(14, 5))

plt.plot(data['date'], data['newCases'],
         color='lightgray', lw=1, alpha=0.4, label="Raw cases")

plt.plot(data['date'], data['newCases_smooth'],
         color='blue', lw=2, label="7-day smoothed")

plt.plot(data_weekly['date'], data_weekly['I_obs'] * N,
         color='red', lw=2, marker='o', label="Weekly incidence")

plt.axvspan(pd.Timestamp('2020-07-01'),
            pd.Timestamp('2022-04-30'),
            alpha=0.08, color='green', label="Study period")

plt.title("COVID-19 cleaned incidence data (lag-reduced)")
plt.ylabel("Cases")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("../../data/covid_cleaned_incidence.png")
plt.show()

t = np.load("../../data/t_data_study.npy").reshape(-1)[:93]
I = np.load("../../data/I_data_study.npy").reshape(-1)[:93]

plt.figure(figsize=(12, 5))
plt.plot(t, I * 56_000_000, color="#004F94", lw=1.5, marker='o', markersize=3)
plt.title("Observed weekly infections (normalised time)", fontsize=13)
plt.xlabel("Normalised time")
plt.ylabel("Infected (count)")
plt.grid(True)
plt.tight_layout()
plt.savefig("../../data/observed_normalised.png", dpi=150)
plt.show()