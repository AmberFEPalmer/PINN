import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

output_dir = "../../png_files"
data_folder = os.path.join("..", "..", "data")

### Initial conditions
S0, E0, I0, R0 = 100000, 0, 1, 0
N = S0 + E0 + I0 + R0
days = 100

### Parameters
beta, sigma, gamma = 0.75, 0.25, 0.25
dt = 1.0
t_arr = np.arange(0, days + dt, dt)

### SEIR simulation
S, E, I, R = S0, E0, I0, R0
S_list, E_list, I_list, R_list = [S], [E], [I], [R]
for t in t_arr[1:]:
    dS = -beta * S * I / N * dt
    dE = (beta * S * I / N - sigma * E) * dt
    dI = (sigma * E - gamma * I) * dt
    dR = gamma * I * dt
    S += dS; E += dE; I += dI; R += dR
    S_list.append(S); E_list.append(E)
    I_list.append(I); R_list.append(R)

S_array = np.array(S_list)
E_array = np.array(E_list)
I_array = np.array(I_list)
R_array = np.array(R_list)

os.makedirs(data_folder, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for noise_percent in range(1, 21):
    noise_level = noise_percent / 100
    I_noisy = I_array + noise_level * I_array * np.random.normal(0, 1, len(I_array))

    ### Normalise to match other CSVs: time 0→1, compartments as fractions
    df = pd.DataFrame({
        "time": t_arr / days,        # 0→1
        "S":    S_array / N,         # fractions
        "E":    E_array / N,
        "I":    I_noisy / N,
        "R":    R_array / N,
    })

    csv_path = os.path.join(data_folder, f"SEIR_Gaussian_noise_{noise_percent}percent.csv")
    df.to_csv(csv_path, index=False)

    ### Plot in original scale (raw counts) for visual inspection
    plt.figure(figsize=(10, 6))
    plt.plot(t_arr, S_array,  label='Susceptible')
    plt.plot(t_arr, E_array,  label='Exposed')
    plt.plot(t_arr, I_noisy,  label='Infected (noisy)')
    plt.plot(t_arr, R_array,  label='Recovered')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title(f'SEIR Model with {noise_percent}% Gaussian Noise in Infected')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"SEIR_Gaussian_noise_{noise_percent}percent.png"))
    plt.close()