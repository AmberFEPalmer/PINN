import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

### https://adventuresinpython.blogspot.com/2012/08/fitting-differential-equation-system-to.html

### Population of England
N = 56000000

### Initial conditions
E0 = 0 
I0 = 1000 
R0 = 0
S0 = N - I0 - R0 - E0 
days = 337

### Parameters
sigma = 1/5.2
gamma = 1/2.9
beta = 0.4

### SEIR equations
def ode_model(t, y, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def ode_solver(t, initial_conditions, parameters, N):
    beta, sigma, gamma = parameters

    sol = solve_ivp(
        ode_model,
        (t[0], t[-1]),
        [S0, E0, I0, R0],
        args=(N, beta, sigma, gamma),
        t_eval=t
    )
    return sol

### Run the model
def run_seir(days, S0, E0, I0, R0, beta, sigma, gamma):
    tspan = np.arange(0, days)
    sol = ode_solver(tspan, [S0, E0, I0, R0], [beta, sigma, gamma], N)
    
    S, E, I, R = sol.y
    return tspan, S, E, I, R  

t, S, E, I, R = run_seir(days, S0, E0, I0, R0, beta, sigma, gamma)

### Plot results from SEIR model
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.plot(t, E)
plt.plot(t, I)
plt.plot(t, R)
plt.legend(["S", "E", "I", "R"])
plt.title("SEIR model")
plt.xlabel("Days")
plt.ylabel("Number of people in each compartment")
plt.show()

### Import the COVID-19 data
t_data = np.load("t_data_raw_2020.npy")       ### time points 
I_data = np.load("I_data_raw_2020.npy") 

plt.figure()
plt.plot(t_data, I_data)
plt.show()

### Plot results from SEIR model with the data
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.plot(t, E)
plt.plot(t, I)
plt.plot(t, R)
plt.plot(t, t_data)
plt.legend(["S", "E", "I", "R", "data"])
plt.title("SEIR model with data")
plt.xlabel("Days")
plt.ylabel("Number of people in each compartment")
plt.show()