import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Import data - COVID-19 cases from COVID dashboard
### (Zixuan email)
### File name = nation_newCasesBySpecimenDate.csv
data = pd.read_csv("/Users/oc25003/Desktop/Data/Cases/nation_newCasesBySpecimenDate.csv")
### Selecting only the date and the number of positive cases from the dataset
data = data[['date', 'value']].rename(columns={'value': 'newCases'})
### Convert date column to date time format
data['date'] = pd.to_datetime(data['date'])

### See when the data starts and ends
print("Start date:", data['date'].min())
print("End date:", data['date'].max())

### Check if any dates are missing - (there are not)
all_days = pd.date_range(start=data['date'].min(), end=data['date'].max())
missing = all_days.difference(data['date'])
print("Missing dates:")
print(missing)

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
N = 69_000_000
data['I_obs'] = data['I_prev'] / N

### Split into months 
months = {ym: df for ym, df in data.groupby(data['date'].dt.to_period('M'))}
print(f"\nCreated {len(months)} monthly datasets.")

### Process and save each month
for ym, df in months.items():
    ym_str = str(ym)
    
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Normalize time within THIS month only [0, 1]
    t_month = (df_copy['date'] - df_copy['date'].min()).dt.days
    t_norm_month = t_month / t_month.max() if t_month.max() > 0 else t_month
    
    t_array = t_norm_month.values.reshape(-1, 1)
    I_array = df_copy['I_obs'].values.reshape(-1, 1)
    
    # Save data
    np.save(f"t_data_{ym_str}.npy", t_array)
    np.save(f"I_data_{ym_str}.npy", I_array)
    
    # Collocation points span [0, 1]
    t_col = np.random.uniform(0, 1, (2000, 1))
    np.save(f"t_col_{ym_str}.npy", t_col)
    
    # Print diagnostics for this month
    print(f"{ym_str}: t_norm=[{t_array.min():.4f}, {t_array.max():.4f}], I_obs=[{I_array.min():.6f}, {I_array.max():.6f}], n={len(t_array)}")

### Visualization - use original dataset's normalization for display
data['t'] = (data['date'] - data['date'].min()).dt.days
data['t_norm'] = (data['t'] - data['t'].min()) / (data['t'].max() - data['t'].min())

plt.figure(figsize=(12, 5))
plt.plot(data['date'], data['I_obs'] * N, label="Estimated infectious prevalence", color='blue')
plt.scatter(data['date'], data['newCases'], label="Daily new cases", color='pink', alpha=0.5)
plt.title("COVID-19: Estimated I(t) for SEIR PINN")
plt.ylabel("People")
plt.xlabel("Date")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

### Show one example month
example_month = '2020-03'
if example_month in [str(ym) for ym in months.keys()]:
    t_example = np.load(f"t_data_{example_month}.npy")
    I_example = np.load(f"I_data_{example_month}.npy")
    t_col_example = np.load(f"t_col_{example_month}.npy")
    
    plt.figure(figsize=(12, 5))
    plt.scatter(t_example, I_example, label=f'I_obs ({example_month})', color='blue', s=50, zorder=5)
    plt.scatter(t_col_example, np.zeros_like(t_col_example), color='red', alpha=0.1, s=1, label='Collocation points')
    plt.xlabel('Normalized Time [0, 1]')
    plt.ylabel('I_obs (fraction)')
    plt.title(f'Example Month: {example_month}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()