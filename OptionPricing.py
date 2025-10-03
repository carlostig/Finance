#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:40:23 2025

@author: carlosti
"""

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
# Seleziona Ticker 
ticker= "BMPS.MI"
period_= "1y"

df=yf.download(ticker,period=period_,auto_adjust= True)
# Prendiamo i prezzi di chiusura (aggiustati) 
S=df["Close"]
#Ultimo prezzo sarà il prezzo di partenza per l'evoluzione dei prezzi futuri 
S_0=S.iloc[-1].values

log_r= (np.log(S.iloc[1::]/S.shift(1))).dropna()
#lin_r= S.pct_change().dropna() #alternativa ai log ret, per var.rendimenti bassi i due risultati coincidono
mu= log_r.mean().values # Rendimenti Giornalieri 
sigma= log_r.std().values # Volatilità Giornaliera 

# GBM dSt= muStdt + sigmaStdwt --> Ito --> ST= S_0 * e ** ((mu- 1/2*sigma**2)*t_k + sigma*W_T)  -> W_T aprox sqrt(T)*std.norm

n_sim= 60000 # N.Simulazioni
dt=1 # Passo Giornaliero  --> Se si vuole cambiare passo è necessario aggiustare coerentemente mu e sigma
trading_days=32 #Orizzonte Temporale / Time to Maturity

T=np.arange(0,trading_days + 1,dt)
time_mat=np.tile(T,[n_sim,1])
#
dW= np.random.standard_normal([n_sim,len(T)-1])* np.sqrt(dt)
W_tk= np.hstack([np.zeros([n_sim,1]),np.cumsum(dW,1)]) #Facciamo partire il moto browniano in T=0 da zero 

#%% #plot of the ABM

# plt.figure()
# plt.plot(time_mat.T,W_tk.T)
# plt.show()
#

S_tk= S_0* np.exp((mu- 0.5*np.power(sigma,2))*time_mat + sigma*W_tk)
#%% plot GBM
# plt.figure()
# plt.plot(time_mat.T,S_tk.T)
# plt.xlabel("Trading Days")
# plt.ylabel("Prezzo")
# plt.title(f"Evoluzione del titolo {ticker}")
# plt.show()


# Informazioni sui prezzi finali
S_T = S_tk[:, -1]
print("")
print("---")
print(f" Media prezzo a {trading_days} giorni: {S_T.mean():.2f}")
print(f" Mediana: {np.median(S_T):.2f}")
print(f" VaR 5%: {np.percentile(S_T,5):.3f}")
print(f" CVaR 5%: {np.mean(np.sort(S_T)[:int(len(S_T)*0.05)]):.3f}")
print(f" 95° percentile: {np.percentile(S_T,95):.2f}")
print("---")

#%% Option Pricing MonteCarlo - B&S
#per azioni ita con maturity < 1yrs estrapolo rf da quotazioni bot -- possibile estensione con selenium/bs + tassi interpolati
#url = "https://www.borsaitaliana.it/borsa/obbligazioni/mot/bot/lista.html"

r_f_mensile= (100/99.67) - 1 # Tasso mensile o setta 0 
rf_a= ((1+r_f_mensile)**12 - 1)  # Tasso annuale
r_f_giornaliero = (1 + rf_a)**(1/252) - 1 # Tasso giornaliero --> Uso Tassi giornalieri per migliore adattabilità a opzioni con scadenza minore di un anno

K= 7.80  # Strike Price
n_shares= 1000 # Aggiusta se opzione da diritto all'acquisto di un lotto diverso di azioni
#%% Call
S_T_RN = S_0 * np.exp((r_f_giornaliero - 0.5*sigma**2)*time_mat + sigma*W_tk) # Evoluzione sottostante sotto aspettative Risk Neutral 
S_T_RN_final= S_T_RN[:,-1] # Essendo una Call Europea mi interessa solo del prezzo a scadenza per fare pricing, evoluzione del prezzo del sottostante utile per possibile pricing di opzioni asiatiche

C_0_MC= n_shares * np.exp(-r_f_giornaliero*trading_days)*np.mean(np.maximum(S_T_RN_final-K,0))

d1 = (np.log(S_0/K) + (r_f_giornaliero + 0.5 * sigma**2) * trading_days) / (sigma * np.sqrt(trading_days))
d2 = d1 - sigma * np.sqrt(trading_days)
C_0_BS = n_shares * (S_0 * norm.cdf(d1) - K * np.exp(-r_f_giornaliero * trading_days) * norm.cdf(d2))

#Calcolo Greche 

Delta_C= norm.cdf(d1)
Gamma_C= norm.pdf(d1) / (S_0 * sigma * np.sqrt(trading_days))
Vega_C = S_0 * norm.pdf(d1) * np.sqrt(trading_days) 
Theta_C= (-S_0 * norm.pdf(d1) * sigma / (2 * np.sqrt(trading_days))  - r_f_giornaliero * K * np.exp(-r_f_giornaliero * trading_days) * norm.cdf(d2))
print("---")
print(f" Delta Call {ticker}: {Delta_C[0]:.3f}")
print(f" Gamma Call {ticker} : {Gamma_C[0]:.3f}")
print(f" Vega Call {ticker}: {Vega_C[0]:.3f}")
print(f" Theta Call {ticker} : {Theta_C[0]:.3f}")
print(f"Per fare delta hedging è necessario vendere {round(Delta_C[0],3)*n_shares} azioni {ticker}")
prob_ITM_C = 100 * Delta_C[0]
print(f"La probabilità che la call scada ITM è {prob_ITM_C:.3f}%")
print("---")
#Put 

P_0_MC= n_shares * np.exp(-r_f_giornaliero*trading_days)*np.mean(np.maximum(K-S_T_RN_final,0))
P_0_BS= n_shares* (K * np.exp(-r_f_giornaliero * trading_days) * norm.cdf(-d2)- S_0 * norm.cdf(-d1))
Delta_P = norm.cdf(d1) - 1 
Gamma_P= norm.pdf(d1) / (S_0 * sigma * np.sqrt(trading_days))
Vega_P = S_0 * norm.pdf(d1) * np.sqrt(trading_days) 
Theta_P = (-S_0 * norm.pdf(d1) * sigma / (2 * np.sqrt(trading_days)) + r_f_giornaliero * K * np.exp(-r_f_giornaliero * trading_days) * norm.cdf(-d2))
print("---")
print(f" Delta Put {ticker}: {Delta_P[0]:.3f}")
print(f" Gamma Call {ticker} : {Gamma_P[0]:.3f}")
print(f" Vega Call {ticker}: {Vega_P[0]:.3f}")
print(f" Theta Call {ticker} : {Theta_P[0]:.3f}")
print(f"Per fare delta hedging è necessario acquistare {abs(round(Delta_P[0],3)*n_shares)} azioni {ticker}")
prob_ITM_P = 100 * norm.cdf(-d1)
print(f"La probabilità che la put scada ITM è {prob_ITM_P[0]:.3f}%")
print("---")
