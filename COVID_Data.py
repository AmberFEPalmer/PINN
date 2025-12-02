import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Import data - COVID-19 cases from COVID dashboard
### (Zixuan email)
### File name = nation_newCasesBySpecimenDate.csv
data = pd.read_csv("/Users/oc25003/Desktop/Data/Cases/nation_newCasesBySpecimenDate.csv")
### Selecting only the date and the number of positive cases from the dataset
data = data[['date', 'value']].rename(columns={'value': 'newCases'})
### Convert date column to date time
data['date'] = pd.to_datetime(data['date'])

### 7 day average of cases - consistent with Li et al. 2024 Journal of medical bioinformatics
data['newCases_smooth'] = data['newCases'].rolling(window=7, center=True, min_periods=1).mean()

### Convert date to time variable
data['t'] = (data['date'] - data['date'].min()).dt.days
t_min, t_max = data['t'].min(), data['t'].max()
data['t_norm'] = (data['t'] - t_min) / (t_max - t_min)

### N = Population of UK
N = 69_000_000
### Convert absolute daily new cases into the proportion of the population
data['I_obs'] = data['newCases_smooth'] / N

### Save data
### NumPy arrays
np.save("t_data.npy", data['t_norm'].values.reshape(-1, 1))
np.save("I_data.npy", data['I_obs'].values.reshape(-1, 1))

### Generate collocation points for PINN physics loss
### (Collocation points constitute the training dataset for PINN)
n_col = 2000
t_col = np.random.uniform(0, 1, (n_col, 1))
np.save("t_col.npy", t_col)

### Data visualisation

### Dates of COVID-19 government restrictions (UK)
lockdowns = [
    ("Lockdown 1", "2020-03-23"),
    ("Lockdown 2", "2020-12-02"),
    ("Lockdown 3", "2021-01-04")
]

plt.figure(figsize=(12, 5))
plt.plot(data['date'], data['I_obs'] * N, label="Observed infected (proxy)", color='pink')
plt.title("COVID-19: Observed Infected Proxy")
plt.ylabel("People")
plt.xlabel("Date")
plt.grid(True)

for label, date in lockdowns:
    plt.axvline(pd.to_datetime(date), color='grey', linestyle='--', linewidth=1)
    plt.text(pd.to_datetime(date), plt.ylim()[1], label,
             rotation=90, verticalalignment='top', color='grey')

plt.legend()
plt.show()


