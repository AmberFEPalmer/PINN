import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Import data - COVID-19 cases from COVID dashboard
### (Zixuan email)
### File name = "nation_newCasesBySpecimenDate.csv"
data = pd.read_csv("/Users/oc25003/Desktop/Data/Cases/nation_newCasesBySpecimenDate.csv")

### Inspect first 5 rows
print(data.head())

### Create new dataframe with only date and number of new cases
data = data[['date', 'value']]

### Rename value column to newCasesBySpecimenData
data = data.rename(columns={'value': 'newCasesBySpecimenDate'})

print(data.head())

### Convert dates to a numeric format
data['date'] = pd.to_datetime(data['date'])

data['t'] = (data['date'] - data['date'].min()).dt.days

### Population of the UK = 69_000_000
N = 69_000_000

data['I_fraction'] = data['newCasesBySpecimenDate'] / N

print(data.head())

### Select columns for PINN
t_data = data['t'].values.reshape(-1, 1)          # input = time
I_data = data['I_fraction'].values.reshape(-1, 1) # output = infected fraction

### Inspect pre-processed data for PINN
print("Processed data for PINN:")
print(data.head())

### Save array for PINN
np.save("t_data.npy", t_data)
np.save("I_data.npy", I_data)

### Visualising the data

### Plot new daily cases
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['newCasesBySpecimenDate'], color='pink', label='Daily New Cases')
plt.title('Daily COVID-19 Cases in the UK')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.grid(True)
plt.legend()
plt.show()

### Plot infected fraction of the population
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['I_fraction'], color='pink', label='Fraction of Population Infected')
plt.title('Daily Infected Fraction of the UK Population')
plt.xlabel('Date')
plt.ylabel('Infected Fraction')
plt.grid(True)
plt.legend()
plt.show()
