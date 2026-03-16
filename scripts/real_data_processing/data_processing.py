import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
### Import data - COVID-19 cases from COVID dashboard
### (Zixuan email)
### File name = nation_newCasesBySpecimenDate.csv
data = pd.read_csv("../../Data/Cases/nation_newCasesBySpecimenDate.csv")
 
### Filter for England only
print(data.columns)
print(data.head())
data = data[data['area_name'] == 'England']
 
### Selecting only the date and the number of positive cases from the dataset
data = data[['date', 'value']].rename(columns={'value': 'newCases'})
### Convert date column to date time format
data['date'] = pd.to_datetime(data['date'])
 
### See when the data starts and ends
print("Start date:", data['date'].min())
print("End date:",   data['date'].max())
 
### Sort ascending (oldest first) - required for rolling operations
data = data.sort_values('date').reset_index(drop=True)
 
### Smooth 7-day average - consistent with published literature e.g. Li et al. 2024
data['newCases_smooth'] = data['newCases'].rolling(7, center=True, min_periods=1).mean()
 
### Normalised time (days since start) - used for raw/yearly saves
data['t'] = (data['date'] - data['date'].min()).dt.days
 
### Convert daily cases to infectious prevalence
### Raw data measures new infections but number of currently infected people
### is needed for the SEIR model.
### Assume each individual remains infected for 5 days.
infectious_period = 5
data['I_prev'] = data['newCases_smooth'].rolling(
    infectious_period, min_periods=1
).sum()
 
### Convert infection prevalence into a fraction of the UK population
N = 56_000_000
data['I_obs'] = data['I_prev'] / N

### Study period: July 2020 – April 2022  (~93 weekly points, matches Qian et al. 2025)
mask = (data['date'] >= '2020-07-01') & (data['date'] <= '2022-04-30')
data_study = data[mask].copy().reset_index(drop=True)
 
### Resample to weekly data (use mean of daily values within each week, aligned to week end)
data_weekly = (
    data_study
    .set_index('date')
    .resample('W')
    .mean(numeric_only=True)
    .reset_index()
    .dropna(subset=['I_obs'])
)
 
### Normalise time to [0, 1] over the full study period
t_days_study = (data_weekly['date'] - data_weekly['date'].min()).dt.days
t_norm_study = (t_days_study - t_days_study.min()) / (t_days_study.max() - t_days_study.min())
 
### Save combined study-period arrays 
np.save("../../data/t_data_study.npy", t_norm_study.values.reshape(-1, 1))
np.save("../../data/I_data_study.npy", data_weekly['I_obs'].values.reshape(-1, 1))
 
print(f"\nStudy period (weekly): {len(data_weekly)} points "
      f"({data_weekly['date'].min().date()} → {data_weekly['date'].max().date()})")
print(f"  t_data_study range : {t_norm_study.min():.4f} → {t_norm_study.max():.4f}")
print(f"  I_data_study range : {data_weekly['I_obs'].min():.6f} → {data_weekly['I_obs'].max():.6f}")
 
### Also save collocation points for the study period
np.save("../../data/t_col_study.npy", np.random.uniform(0, 1, (2000, 1)))
 
### Per-year weekly saves 
years = {y: df for y, df in data.groupby(data['date'].dt.year)}
print(f"\nCreated {len(years)} yearly datasets.")
 
for y, df in years.items():
    y_str = str(y)
 
    ### Resample this year's daily data to weekly
    df_weekly = (
        df.set_index('date')
        .resample('W')
        .mean(numeric_only=True)
        .reset_index()
        .dropna(subset=['I_obs'])
    )
 
    ### Normalise time to [0, 1] within the year
    t_week = (df_weekly['date'] - df_weekly['date'].min()).dt.days
    t_norm = (t_week - t_week.min()) / (t_week.max() - t_week.min())
 
    np.save(f"../../data/t_data_{y_str}.npy", t_norm.values.reshape(-1, 1))
    np.save(f"../../data/I_data_{y_str}.npy", df_weekly['I_obs'].values.reshape(-1, 1))
 
    ### Also save raw (unnormalised) time in days for reference
    np.save(f"../../data/t_data_raw_{y_str}.npy", t_week.values)
    np.save(f"../../data/I_data_raw_{y_str}.npy", df_weekly['I_obs'].values)
 
    print(f"  {y_str}: {len(df_weekly)} weekly points saved")
 
### Verification prints
t_study = np.load("../../data/t_data_study.npy")
I_study = np.load("../../data/I_data_study.npy")
print(f"\nt_data_study : {len(t_study)} points, "
      f"range {t_study.min():.4f} → {t_study.max():.4f}")
print(f"I_data_study : min {I_study.min():.6f}, max {I_study.max():.6f}")
 
if len(t_study) < 85 or len(t_study) > 100:
    print("WARNING: expected ~93 weekly points for July 2020–April 2022. "
          f"Got {len(t_study)} — check the date filter.")
else:
    print(f"OK: {len(t_study)} weekly points (expected ~93).")
 
# Visualisation of UK COVID-19 case data
plt.figure(figsize=(14, 5))
plt.plot(data['date'], data['I_obs'] * N,
         color='blue', lw=1, alpha=0.4, label="Daily estimated prevalence")
plt.plot(data_weekly['date'], data_weekly['I_obs'] * N,
         color='blue', lw=2, label="Weekly (study period)")
plt.scatter(data['date'], data['newCases'],
            color='pink', alpha=0.3, s=4, label="Daily new cases (raw)")
plt.axvspan(pd.Timestamp('2020-07-01'), pd.Timestamp('2022-04-30'),
            alpha=0.08, color='green', label="Study period")
plt.title("COVID-19: Estimated I(t) for SEIR PINN")
plt.ylabel("People")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("../../data/covid_all_time.png")
plt.show()