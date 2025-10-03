#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:17:16 2024

@author: carlosti
"""

import cvxpy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from bs4 import BeautifulSoup
from cvxopt import matrix,solvers
import requests

url="https://en.wikipedia.org/wiki/Nasdaq-100"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
response.raise_for_status()  # controlla eventuali errori HTTP




soup= BeautifulSoup(response.text,"html.parser")

tables = soup.find_all("table", {"id": "constituents"})

for table in tables:
     headers=[header.text.strip() for header in table.find_all("th") if header.text.strip()]
     rows= []
     for row in table.find_all("tr")[1:]: ## Gira la singola riga
         cells=[cell.text.strip() for cell in row.find_all("td") if cell.text.strip() ]
         rows.append(cells)

         


Table= pd.DataFrame(rows,columns=headers)

tickers=Table["Ticker"].to_list()


# In alternativa carichiamo csv con i prezzi prices = pd.read_csv('prices.csv', index_col=0)
 

prices= yf.download(tickers,start="2025-01-01",auto_adjust= True)
#%%
prices=prices.dropna(axis=1)["Close"]
#%%


returns=prices.pct_change()[1:]

n_obs, n_instruments = returns.shape

NN = 40

def cvar_optimization(returns: pd.DataFrame, confidence_level: float, eta: float | None = None) -> dict[str, float | pd.Series]:
    weights_variable = cp.Variable(n_instruments, nonneg=True)
    zeta = cp.Variable(1)
    d = cp.Variable(n_obs, nonneg=True)
    objective = cp.Minimize(zeta + 1 / (n_obs * confidence_level) * cp.sum(d))
    constraints = [
        cp.sum(weights_variable) == 1,
        -returns.values @ weights_variable - zeta - d <= 0
    ]
    mu_vector = returns.mean().values
    if eta is not None:
        constraints.append(mu_vector @ weights_variable == eta)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status != cp.OPTIMAL:
        raise Exception('Optimization not optimal')
    weights_series = pd.Series(weights_variable.value, index=returns.columns)
    return {
        "weights": weights_series,
        "objective_value": problem.value,
        "expected_return": mu_vector @ weights_series
    }


confidence_level = 0.05
ptf_dict = cvar_optimization(returns, confidence_level)


w = ptf_dict["weights"]
#w.plot.bar(figsize=(12, 8))


# compute cvar
ptf_rets = returns @ w
sorted_rets = np.sort(ptf_rets, axis=0)
cvar = - np.mean(sorted_rets[:int(confidence_level * sorted_rets.shape[0])])




mu_vector = returns.mean().values
min_ptf_mu = mu_vector @ w
max_ptf_mu = mu_vector.max()
eta_vector = np.linspace(min_ptf_mu, max_ptf_mu, NN)


ptf_list = [cvar_optimization(returns, confidence_level, eta) for eta in eta_vector]

Portfolio_Cvar= np.array([s["objective_value"] for s in ptf_list])
Portfolio_ExpReturns=[s["expected_return"] for s in ptf_list]

plt.figure()
plt.plot(Portfolio_Cvar,Portfolio_ExpReturns)
plt.xlabel("CVar")
plt.ylabel("Mu")
plt.title("Frontiera Efficiente")
plt.show()
