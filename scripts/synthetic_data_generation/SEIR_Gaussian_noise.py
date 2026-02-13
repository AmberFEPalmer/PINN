import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

### Initial conditions
S0, E0, I0, R0 = 100000, 0, 1, 0
N = S0 + E0 + I0 + R0
days = 100

### Parameters
beta, sigma, gamma = 0.8, 0.1, 0.1
dt = 0.1  # time step

t_arr = np.arange(0, days + dt, dt)

### Initialise arrays
S, E, I, R = S0, E0, I0, R0
S_list, E_list, I_list, R_list = [S], [E], [I], [R]

### Model
for t in t_arr[1:]:
    dS = -beta * S * I / N * dt
    dE = (beta * S * I / N - sigma * E) * dt
    dI = (sigma * E - gamma * I) * dt + 0.02 * I * np.random.normal(0, np.sqrt(dt))
    dR = gamma * I * dt

    S += dS
    E += dE
    I += dI
    R += dR

    S_list.append(S)
    E_list.append(E)
    I_list.append(I)
    R_list.append(R)

### Convert to arrays
S_array = np.array(S_list)
E_array = np.array(E_list)
I_array = np.array(I_list)
R_array = np.array(R_list)

### Visualisation
plt.figure(figsize=(10,6))
plt.plot(t_arr, S_array, label='Susceptible')
plt.plot(t_arr, E_array, label='Exposed')
plt.plot(t_arr, I_array, label='Infected')
plt.plot(t_arr, R_array, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SEIR Model with Noisy Beta')
plt.legend()
plt.show()

### Normalize entire time series
S_norm = S_array / N
E_norm = E_array / N
I_norm = I_array / N
R_norm = R_array / N

### Normalize time for PINN
t_norm = t_arr / t_arr.max()

### Export SEIR results to a csv file
data_folder = os.path.join("..", "..", "data")

SEIR_noise_data = pd.DataFrame({"time": t_norm,"I": I_norm,})
print(SEIR_noise_data)

csv_path = os.path.join(data_folder, "SEIR_noise_data.csv")
SEIR_noise_data.to_csv(csv_path, index=False)