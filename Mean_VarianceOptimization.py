#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 20:25:09 2024

@author: carlosti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import yfinance as yf
from bs4 import BeautifulSoup
from cvxopt import matrix,solvers
import requests
#%%

# Scarico i tickers presenti nel Nasdaq-100

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

def MarkowitzOptimization(returns: pd.DataFrame, eta: float | None=None) -> dict[str, float | pd.Series]:
    [T,n]=returns.shape
    mu=returns.mean().values
    P= (returns.cov().values)
    q=(np.zeros((n,1)))
    G=(-np.eye(n))
    h=(np.zeros((n,1)))
    A=(np.ones((1,n)))
    b=([1.0])
    if eta != None:
        A=np.vstack((A,mu))
        b=np.vstack((b,eta))
    sol= solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    if sol["status"] != "optimal":
        raise Exception("Warning- Failed Optimization")
    X=np.array(sol["x"]).flatten()
    X[X<1e-5]=0
    Portfolio_Weights=pd.Series(X,index=returns.columns)
    Portfolio_Variance=np.array(sol["primal objective"])
    Portfolio_ExpRet= Portfolio_Weights @ mu
    return {"X":Portfolio_Weights,
            "Variance": Portfolio_Variance,
            "Exp_Ret": Portfolio_ExpRet}


#  Portafoglio a minimo rischio
sol= (MarkowitzOptimization(returns))


# Nested for loop solution
mu=returns.mean().values
eta_min=sol["Exp_Ret"]
#Portafoglio a massimo rendimento = titolo con rendimento maggiore
eta_max=np.max(mu)

NN=50

eta=np.linspace(eta_min, eta_max,NN)

# %% Ottimizzazione for loops
sol= [MarkowitzOptimization(returns,et) for et in eta]
#%%


Portfolio_Variance= np.array([s["Variance"] for s in sol])
Portfolio_ExpReturns=[s["Exp_Ret"] for s in sol]

plt.figure()
plt.plot(Portfolio_Variance,Portfolio_ExpReturns)
plt.xlabel("Sigma")
plt.ylabel("Mu")
plt.title("Frontiera Efficiente")
plt.show()

# Estrarre pesi portafoglio k 
#pesi=sol[k]["X"]

