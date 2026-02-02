import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

### Based on https://wrap.warwick.ac.uk/id/eprint/3301/

### Deterministic SIR metapopulation model
### Infectiousness in multiple stages

### Define ODE metapopulation model
### Variables written as a list rather than S, I1, I2, I3 etc to make the code shorter
def metapopulation_model(t, y, lam, gamma):
    """
    t = time
    y = state variables
    Lambda = force of infection
    gamma = recovery
    """
    
    S = y[0] ### first element
    I = y[1:-1] ### all elements from 1 to second to last
    R = y[-1] ### last element

    ### number of infectious stages
    m = len(I)
    
    ### Make an array to store the derivatives
    dydt = np.zeros_like(y)

    ### Susceptible equation
    dydt[0] = -lam * S

    ### Infected 1 equation
    dydt[1] = lam * S - gamma[0] * I[0]

    ### Other infected stages equation
    for n in range(1, m):
        dydt[n+1] = gamma[n-1] * I[n-1] - gamma[n] * I[n]

    ### Recovered equation
    dydt[-1] = gamma[-1] * I[-1]

    return dydt

### Define parameters
lam = 0.1
gamma = [0.2, 0.2, 0.2]

### Define state variables
S0 = 999
I0 = [1, 0, 0]
R0 = 0 

y = [S0] + I0 + [R0]

t_eval = np.linspace(0, 50, 500)

sol = solve_ivp(
    metapopulation_model,
    t_span = (0, 50),
    y0 = y,
    args=(lam, gamma),
    t_eval=t_eval
)

### Visualisation
plt.plot(sol.t, sol.y[0], label='S')
plt.plot(sol.t, sol.y[1], label='I1')
plt.plot(sol.t, sol.y[2], label='I2')
plt.plot(sol.t, sol.y[3], label='I3')
plt.plot(sol.t, sol.y[4], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

### Normalise time for PINN 
t_norm = sol.t / sol.t.max()

### Export metapopulation results to a csv file
I_total = sol.y[1:-1, :].sum(axis=0) 

### Normalise I for PINN
I_norm = I_total / I_total.max()   # scale to 0-1

metapopulation_data = pd.DataFrame({"time": t_norm,"I": I_norm,})

print(metapopulation_data)
metapopulation_data.to_csv("metapopulation_results.csv", index=False)
