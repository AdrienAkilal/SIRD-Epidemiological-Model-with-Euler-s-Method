import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d

#PARTIE 01:
def methode_euler(beta, gamma, mu, S0, I0, R0, D0, dt, t_end):
    n_steps = int(t_end / dt)
    t = np.linspace(0, t_end, n_steps + 1)
    S = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    D = np.zeros(n_steps + 1)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    for i in range(n_steps):
        dSdt = -beta * S[i] * I[i]
        dIdt = beta * S[i] * I[i] - gamma * I[i] - mu * I[i]
        dRdt = gamma * I[i]
        dDdt = mu * I[i]
        S[i+1] = S[i] + dSdt * dt
        I[i+1] = I[i] + dIdt * dt
        R[i+1] = R[i] + dRdt * dt
        D[i+1] = D[i] + dDdt * dt
    return t, S, I, R, D

#PARTIE 02:
# Parametres
beta = 0.5
gamma = 0.15
mu = 0.015
S0 = 0.99
I0 = 0.01
R0 = 0.0
D0 = 0.0
dt = 0.01
t_end = 365
t, S, I, R, D = methode_euler(beta, gamma, mu, S0, I0, R0, D0, dt, t_end)

# Visualisation
plt.figure(figsize=(12, 8))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Population (%)')
plt.title('SIRD Model Simulation')
plt.legend()
plt.grid(True)
plt.show()

#PARTIE 03:
data = pd.read_csv('sird_dataset (1).csv')
t_data = data['Jour'].values
S_data = data['Susceptibles'].values
I_data = data['Infectés'].values
R_data = data['Rétablis'].values
D_data = data['Décès'].values

def cost_function(params, t_data, S_data, I_data, R_data, D_data, dt=0.01):
    beta, gamma, mu = params
    t_sim, S, I, R, D = methode_euler(beta, gamma, mu, S0=S_data[0], I0=I_data[0], R0=R_data[0], D0=D_data[0], dt=dt, t_end=t_data[-1])
    S_interp = interp1d(t_sim, S)(t_data)
    I_interp = interp1d(t_sim, I)(t_data)
    R_interp = interp1d(t_sim, R)(t_data)
    D_interp = interp1d(t_sim, D)(t_data)
    mse_S = mean_squared_error(S_data, S_interp)
    mse_I = mean_squared_error(I_data, I_interp)
    mse_R = mean_squared_error(R_data, R_interp)
    mse_D = mean_squared_error(D_data, D_interp)
    return mse_S + mse_I + mse_R + mse_D

# Grid Search
beta_values = np.linspace(0.25, 0.5, 10)
gamma_values = np.linspace(0.08, 0.15, 10)
mu_values = np.linspace(0.005, 0.015, 10)
best_params = None
min_cost = float('inf')
for beta in beta_values:
  for gamma in gamma_values:
    for mu in mu_values:
      cost = cost_function((beta, gamma, mu), t_data, S_data, I_data, R_data, D_data)
      if cost < min_cost:
          min_cost = cost
          best_params = (beta, gamma, mu)

print(f"Les parametres: beta={best_params[0]:.4f}, gamma={best_params[1]:.4f}, mu={best_params[2]:.4f}")
t_opt, S_opt, I_opt, R_opt, D_opt = methode_euler(*best_params, S0=S_data[0], I0=I_data[0], R0=R_data[0], D0=D_data[0], dt=0.01, t_end=t_data[-1])
plt.figure(figsize=(12, 8))
plt.plot(t_data, S_data, label='Susceptible (Data)', linestyle='--', marker='o')
plt.plot(t_opt, S_opt, label='Susceptible (Model)')
plt.plot(t_data, I_data, label='Infected (Data)', linestyle='--', marker='o')
plt.plot(t_opt, I_opt, label='Infected (Model)')
plt.plot(t_data, R_data, label='Recovered (Data)', linestyle='--', marker='o')
plt.plot(t_opt, R_opt, label='Recovered (Model)')
plt.plot(t_data, D_data, label='Deceased (Data)', linestyle='--', marker='o')
plt.plot(t_opt, D_opt, label='Deceased (Model)')
plt.xlabel('Time (days)')
plt.ylabel('Population (%)')
plt.title('SIRD Model Simulation ')
plt.legend()
plt.grid(True)
plt.show()

# Intervention
beta_intervention = 0.2
intervention_start_day = 50


def methode_intervention(beta, gamma, mu, S0, I0, R0, D0, dt, t_end, beta_intervention, intervention_start_day):
    n_steps = int(t_end / dt)
    t = np.linspace(0, t_end, n_steps + 1)
    S = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    D = np.zeros(n_steps + 1)

    S[0] = S0
    I[0] = I0
    R[0] = R0
    D[0] = D0

    for i in range(n_steps):
        current_beta = beta_intervention if t[i] >= intervention_start_day else beta
        dSdt = -current_beta * S[i] * I[i]
        dIdt = current_beta * S[i] * I[i] - gamma * I[i] - mu * I[i]
        dRdt = gamma * I[i]
        dDdt = mu * I[i]

        S[i+1] = S[i] + dSdt * dt
        I[i+1] = I[i] + dIdt * dt
        R[i+1] = R[i] + dRdt * dt
        D[i+1] = D[i] + dDdt * dt

    return t, S, I, R, D


t_intervention, S_intervention, I_intervention, R_intervention, D_intervention = methode_intervention(
    beta, gamma, mu, S0, I0, R0, D0, dt, t_end, beta_intervention, intervention_start_day
)

# 3. Comparaison des scénarios

plt.figure(figsize=(12, 8))
plt.plot(t, I, label='Infected (Sans Intervention)')
plt.plot(t_intervention, I_intervention, label='Infected (Avec Intervention)')
plt.axvline(x=intervention_start_day, color='red', linestyle='--', label='Intervention Start')


plt.xlabel('Time (days)')
plt.ylabel('Population (%)')
plt.legend()
plt.grid(True)
plt.show()
