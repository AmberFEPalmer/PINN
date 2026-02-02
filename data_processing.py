import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Import data - COVID-19 cases from COVID dashboard
### (Zixuan email)
### File name = nation_newCasesBySpecimenDate.csv
data = pd.read_csv("Data/Cases/nation_newCasesBySpecimenDate.csv")

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
print("End date:", data['date'].max())

### Smooth 7-day average - consistent with published literature e.g. Li et al. 2024
data['newCases_smooth'] = data['newCases'].rolling(7, center=True, min_periods=1).mean()

### Normalized time to [0,1]
### (make time points between 0 and 1 - PINN works better with values between 0 and 1)
data['t'] = (data['date'] - data['date'].min()).dt.days
data['t_norm'] = (data['t'] - data['t'].min()) / (data['t'].max() - data['t'].min())

### Convert daily cases to infectious prevalence
### raw data measures new infections but number of currently infected people is needed for the SEIR model
### assume each individual remains infected for 5 days
infectious_period = 5
data['I_prev'] = data['newCases_smooth'].rolling(
    infectious_period, min_periods=1
).sum()

### Convert infection prevelance into a fraction of the population of the UK
N = 56_000_000
data['I_obs'] = data['I_prev'] / N

### Split into months
months ={ym: df for ym, df in data.groupby(data['date'].dt.to_period('M'))}
print(f"Created {len(months)} monthly datasets.")

### Save each month separately for PINN - as arrays (neural networks need arrays rather than dataframes)
for ym, df in months.items():
    ym_str = str(ym)  # e.g. "2020-03"
    np.save(f"t_data_{ym_str}.npy", df['t_norm'].values.reshape(-1, 1))
    np.save(f"I_data_{ym_str}.npy", df['I_obs'].values.reshape(-1, 1))

### Split into years
years ={y: df for y, df in data.groupby(data['date'].dt.year)}
print(f"Created {len(years)} yearly datasets.")

# Save each year separately with no normalisation to use in SEIR model
for y, df in years.items():
    y_str = str(y)

    # Raw time values for this year
    t_year = df['t'].values
    I_year = df['I_obs'].values

    np.save(f"t_data_raw_{y_str}.npy", t_year)
    np.save(f"I_data_raw_{y_str}.npy", I_year)

### Save each year separately for PINN as arrays 
for y, df in years.items():
    y_str = str(y)
    # Normalize t within this year
    t_year = (df['t'] - df['t'].min()) / (df['t'].max() - df['t'].min())
    # Save the per-year normalized t
    np.save(f"t_data_{y_str}.npy", t_year.values.reshape(-1, 1))
    np.save(f"I_data_{y_str}.npy", df['I_obs'].values.reshape(-1, 1))

t_data_2020 = np.load("t_data_2020.npy")
print("t_data_2020 range:", t_data_2020.min(), "to", t_data_2020.max())

t_data_2021 = np.load("t_data_2021.npy")
print("t_data_2021 range:", t_data_2021.min(), "to", t_data_2021.max())

### Save collocation points 
np.save("t__months_col.npy", np.random.uniform(0, 1, (2000, 1)))

## Visualization
plt.figure(figsize=(12, 5))
plt.plot(data['date'], data['I_obs'] * N, label="Estimated infectious prevalence", color='blue')
plt.scatter(data['date'], data['newCases'], label="Daily new cases", color='pink', alpha=0.5)
plt.title("COVID-19: Estimated I(t) for SEIR PINN")
plt.ylabel("People")
plt.grid(True)
plt.legend()
plt.savefig("covid_all_time.png")

#### plt.axvline(x=t_train_np[-1],color='gray', linestyle='--', label='Train/Test Split')

### TODO - add to graph
### 23rd March - first UK lockdown
### 1st June - phased re-opening of schools
### 15th June - shops reopen
### 4th July - first local lockdown
### 14th Aug - lockdown eased
### 14th Sept - rule of 6
### 31st Oct - 2nd lockdown
### 2nd Dec - 2nd lockdown lifted
### 6th Jan - 3rd lockdown
