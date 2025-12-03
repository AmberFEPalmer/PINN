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

### Smooth 7-day average - consistent with published literature e.g. Li et al. 2024
data['newCases_smooth'] = data['newCases'].rolling(7, center=True, min_periods=1).mean()

### Filter time window
### These dates show increasing cases therefore easy start for PINN
### (mask = filter)
start_date = '2021-12-15'
end_date   = '2021-12-31'
mask = (data['date'] >= start_date) & (data['date'] <= end_date)
data = data[mask]

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

### Save arrays for PINN (neural networks need arrays rather than dataframes)
np.save("t_data.npy",  data['t_norm'].values.reshape(-1, 1))
np.save("I_data.npy",  data['I_obs'].values.reshape(-1, 1))
np.save("t_col.npy", np.random.uniform(0, 1, (2000, 1)))

## Visualization
plt.figure(figsize=(12, 5))
plt.plot(data['date'], data['I_obs'] * N, label="Estimated infectious prevalence", color='blue')
plt.scatter(data['date'], data['newCases'], label="Daily new cases", color='pink', alpha=0.5)
plt.title("COVID-19: Estimated I(t) for SEIR PINN")
plt.ylabel("People")
plt.grid(True)
plt.legend()
plt.show()