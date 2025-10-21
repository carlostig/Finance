#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:05:46 2025

@author: carlosti
"""

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
from datetime import datetime
#%%
class OptionPricing:
    def __init__(self,ticker: str,maturity_date: str, strike_price: float | int, n_shares: float| int | None= None, rf_rate: float|int |None= None, period: str | None= None):
        """
        ticker = ticker da valutare; e.g: "BMPS.MI"
        maturity_date= expiration date dell'opzione; e.g : "2025-11-28"
        strike_price = prezzo strike 
        n_shares= numero di sottostanti incorporati nel contratto, default 1 titolo
        rf_rate = Tasso Risk-free annuale, default 0
        period = finestra temporale dei rendimenti da estrarre, default 1yr
        """
        if period is None:
            self.period = "1y"
        else:
            self.period = period
        if n_shares is None:
            self.n_shares= 1
        else:
            self.n_shares= n_shares
        if rf_rate is None:
            self.rf_rate= 0
        else:
            self.rf_rate= (1+rf_rate)**(1/252) - 1
            
        self.ticker= ticker
        self.df=yf.download(self.ticker,period=self.period,auto_adjust= True)
        self.maturity_date= maturity_date
        self.strike_price= strike_price
        
    def asset(self):
        self.S=self.df["Close"]
        self.log_r= (np.log(self.S.iloc[1::]/self.S.shift(1))).dropna()
        self.mu= self.log_r.mean().values  
        self.sigma= self.log_r.std().values 
        self.S_0=self.S.iloc[-1].item()
        return self 
    
    def calcola_ttm(self, maturity_date):
        if isinstance(maturity_date, str):
            self.maturity = pd.to_datetime(maturity_date)
        else:
            self.maturity = maturity_date
        
        today = pd.Timestamp.now().normalize()
        
        self.trading_days = pd.bdate_range(start=today, end=self.maturity)
        return len(self.trading_days) - 1
    
    
    def GBM(self,n_sim):
        dt= 1
        self.asset()
        self.T=np.arange(0,self.calcola_ttm(self.maturity_date) + 1,dt)
        self.time_mat=np.tile(self.T,[n_sim,1])
        self.dW= np.random.standard_normal([n_sim,len(self.T)-1])* np.sqrt(dt)
        self.W_tk= np.hstack([np.zeros([n_sim,1]),np.cumsum(self.dW,1)]) 
        #self.S_0=self.S.iloc[-1].item()
        S_tk= self.S_0* np.exp((self.mu- 0.5*np.power(self.sigma,2))*self.time_mat + self.sigma*self.W_tk)
        self.S_T = S_tk[:, -1]
        self.ttm = self.calcola_ttm(self.maturity_date) 
        
        plt.plot(self.time_mat.T,S_tk.T)
        plt.xlabel("Trading Days")
        plt.ylabel("Prezzo")
        plt.title(f"Evoluzione del titolo {self.ticker}")
        plt.show()
 
        print("")
        print("---")
        print(f" Media prezzo a {self.ttm} giorni: {self.S_T.mean():.2f}")
        print(f" Mediana: {np.median(self.S_T):.2f}")
        print(f" VaR 5%: {np.percentile(self.S_T,5):.3f}")
        print(f" CVaR 5%: {np.mean(np.sort(self.S_T)[:int(len(self.S_T)*0.05)]):.3f}")
        print(f" 95° percentile: {np.percentile(self.S_T,95):.2f}")
        print("---")
        return self
    
        
    def MonteCarlo(self,flag : int | None = None):
        """
        
        Flag = 0; Call
        Flag = 1; Put
        Flag = None; Call & Put
        """
        if not hasattr(self, 'W_tk'):
            raise ValueError(" Chiama GBM() prima di MonteCarlo()")
    
        if not hasattr(self, 'S_0'):
            self.asset()
            self.S_0 = self.S.iloc[-1]
            
        S_T_RN = self.S_0 * np.exp((self.rf_rate - 0.5*self.sigma**2)*self.time_mat + self.sigma*self.W_tk) 
        S_T_RN_final= S_T_RN[:,-1] 
        self.C_0_MC= self.n_shares * np.exp(-self.rf_rate*self.ttm)*np.mean(np.maximum(S_T_RN_final-self.strike_price,0))
        self.P_0_MC= self.n_shares * np.exp(-self.rf_rate*self.ttm)*np.mean(np.maximum(self.strike_price-S_T_RN_final,0))
        if flag == 0:
            return self.C_0_MC
        elif flag == 1:
            return self.P_0_MC
        else:
            return self.C_0_MC, self.P_0_MC 

    
    def BlackScholes(self,flag : int | None = None):
        """
        Flag = 0; Call
        Flag = 1; Put
        Flag = None; Call & Put
        """
        if not hasattr(self, 'S_0'):
            self.asset()
            self.S_0 = self.S.iloc[-1].item()
        
        T = self.calcola_ttm(self.maturity_date)
        
        self.d1 = (np.log(self.S_0 / self.strike_price) + 
                   (self.rf_rate + 0.5 * self.sigma**2) * T) / \
                  (self.sigma * np.sqrt(T))
        self.d2 = self.d1 - self.sigma * np.sqrt(T)
        
        self.C_0_BS = self.n_shares * (
            self.S_0 * norm.cdf(self.d1) - 
            self.strike_price * np.exp(-self.rf_rate * T) * norm.cdf(self.d2)
        )
        
        self.P_0_BS = self.n_shares * (
            self.strike_price * np.exp(-self.rf_rate * T) * norm.cdf(-self.d2) - 
            self.S_0 * norm.cdf(-self.d1)
        )
        
        if flag == 0:
            return self.C_0_BS
        elif flag == 1:
            return self.P_0_BS
        else:
            return self.C_0_BS, self.P_0_BS
        
    def greeks(self,flag):
        if not hasattr(self,'d1'):
            self.BlackScholes(flag)
            
        T = self.calcola_ttm(self.maturity_date)
        Delta_C= norm.cdf(self.d1)
        Gamma_C= norm.pdf(self.d1) / (self.S_0 * self.sigma * np.sqrt(T))
        Vega_C = self.S_0 * norm.pdf(self.d1) * np.sqrt(T) 
        Theta_C= (-self.S_0 * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(T))  - self.rf_rate * self.strike_price * np.exp(-self.rf_rate * T) * norm.cdf(self.d2))
        
        Delta_P = norm.cdf(self.d1) - 1 
        Gamma_P= norm.pdf(self.d1) / (self.S_0 * self.sigma * np.sqrt(T))
        Vega_P = self.S_0 * norm.pdf(self.d1) * np.sqrt(T) 
        Theta_P = (-self.S_0 * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(T)) + self.rf_rate * self.strike_price * np.exp(-self.rf_rate * T) * norm.cdf(-self.d2))
        if flag == 0:
            print("---")
            print(f" V_0 Call {self.ticker}: {self.C_0_BS[0]:.3f}")
            print(f" Delta Call {self.ticker}: {Delta_C[0]:.3f}")
            print(f" Gamma Call {self.ticker} : {Gamma_C[0]:.3f}")
            print(f" Vega Call {self.ticker}: {Vega_C[0]:.3f}")
            print(f" Theta Call {self.ticker} : {Theta_C[0]:.3f}")
            print(f"Per fare delta hedging è necessario vendere {round(Delta_C[0],3)*self.n_shares} azioni {self.ticker}")
            prob_ITM_C = 100 * Delta_C[0]
            print(f"La probabilità che la call scada ITM è {prob_ITM_C:.3f}%")
            print("---")
            return ({'Delta': Delta_C, 'Gamma': Gamma_C, 'Vega': Vega_C, 'Theta': Theta_C})
        elif flag == 1:
            print("---")
            print(f" V_0 Put {self.ticker}: {self.P_0_BS[0]:.3f}")
            print(f" Delta Put {self.ticker}: {Delta_P[0]:.3f}")
            print(f" Gamma Put {self.ticker} : {Gamma_P[0]:.3f}")
            print(f" Vega Put {self.ticker}: {Vega_P[0]:.3f}")
            print(f" Theta Put {self.ticker} : {Theta_P[0]:.3f}")
            print(f"Per fare delta hedging è necessario acquistare {abs(round(Delta_P[0],3)*self.n_shares)} azioni {self.ticker}")
            prob_ITM_P = 100 * norm.cdf(-self.d1)
            print(f"La probabilità che la put scada ITM è {prob_ITM_P[0]:.3f}%")
            print("---")
            return ({'Delta': Delta_P, 'Gamma': Gamma_P, 'Vega': Vega_P, 'Theta': Theta_P})
        else: 
            return ({
            'Call': {'Delta': Delta_C, 'Gamma': Gamma_C, 'Vega': Vega_C, 'Theta': Theta_C},
            'Put': {'Delta': Delta_P, 'Gamma': Gamma_P, 'Vega': Vega_P, 'Theta': Theta_P}
        })
#%% 
# ticker= "BMPS.MI"
# period_= "1y"
# tmt= '2025-11-28'
# strike= 7.80
#%%
# ops= OptionPricing(ticker,period_,tmt,strike,n_shares=1000)

# call_bs= ops.BlackScholes(0)
# put_bs=ops.BlackScholes(1)

# ops.GBM(10000)
# call_mc= ops.MonteCarlo(0)
# put_mc= ops.MonteCarlo(1)
# delta= ops.greeks(1)

#%%
tickers= ["BMPS.MI","LDO.MI"]
periods = ["1y","1y"]
tmt= ["2025-11-28",'2025-12-25']
strikes= [7.80,54]
n_shares= [1000,500]

results= {}
for k in range(len(tickers)):
    
    ops= OptionPricing(tickers[k],tmt[k],strikes[k],n_shares[k], period = periods[k])
    call_bs= ops.BlackScholes(0)
    
    df = {
        "Prezzo_Sottostante": [ops.S_0],
        "Strike_Price":strikes[k],
        "Days_To_Maturity": ops.calcola_ttm(tmt[k]),
        "N_Underlying": n_shares[k],
        "Prezzo_Call": call_bs
    }
    delta= ops.greeks(0)
    comb= df | delta
    results[f"{tickers[k]}"]= pd.DataFrame(comb)
