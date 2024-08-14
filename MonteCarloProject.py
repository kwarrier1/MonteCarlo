# Monte Carlo Simulation for European Call Option
# Author: Keshav Warrier
# Inspiration from QuantPy YouTube channel
import numpy as np
import pandas as pd
from scipy.stats import norm
import datetime

S = 232.98      # stock price
K = 240         # strike
vol = 0.2219    # implied vol
r = 0.043       # rfr
N = 10          # number of time steps
M = 1000        # number of simulations
T = ((datetime.date(2024,7,26)-datetime.date(2024,7,5)).days+1) / 365           # Expires July 26th

def user_info():
    print("European Call Option Price Calculator")
    flag = input("Use predetermined values? (Y/N) ")
    if (flag == "N"):
        global S, K, vol, M, T
        S = int(input("Enter stock price ($): "))
        K = int(input("Enter strike price ($): "))
        vol = float(input("Enter implied volatility (% in decimal form): "))
        M = int(input("Number of simulations: "))
        expStr = input("Expiration date (year-month-date): ")
        year, month, date = map(int, expStr.split("-"))
        exp = datetime.date(year, month, date)
        T = ((exp-datetime.date.today()).days + 1) / 365
    elif (flag == "Y"):
        print("Starting simulation...")
    else:
        print("Invalid flag, restarting program")
        user_info()
print(datetime.datetime.today())
user_info()

# Finds present value of an European call option according to the Black Scholes formula    
def get_bs_price(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + vol**2/2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(S/K) + (r - vol**2/2) * T) / (vol * np.sqrt(T))
    bs_price = S * norm.cdf(d1, 0, 1) - np.exp(-r*T)*K*norm.cdf(d2, 0, 1)
    return bs_price

# Monte Carlo
def get_MC_sim(S, K, T, r, vol):
    dt = T/N                                                                                  # change in time
    nudt = (r - 0.5*vol**2)*dt                                                                # drift term calculation
    volsdt = vol * np.sqrt(dt)                                                                # volatility term (vol * change in brownian variable)
    lnS = np.log(S)                                                                           # Ito calculus uses ln of stock price

    X = np.random.normal( size = (N, M))                                                      # matrix of size time steps x number of simulations
    dlnSt = nudt + volsdt*X                                                                   # new matrix for change in lnSt
    lnSt = lnS + np.cumsum(dlnSt, axis = 0)                                                   # sum the changes along each path, and add the natural log of the original price
    endPrices = np.exp(lnSt)                                                                  # reverting end price
    endValues = np.maximum(0, endPrices - K)                                                  # value of call is equal to the max of 0 and the end price - strike
    callFV = np.sum(endValues[-1]) / M                                                        # average all simulations to find the future call value
    callPV = callFV * np.exp(-r*T)                                                            # present value of call

    sx = np.sqrt(np.sum((endValues[-1] - callFV) ** 2) / ( M - 1))                            # standard deviation of x, which is the call value
    SE = sx / np.sqrt(M)                                                                      # standard error calculation
    return callPV, SE

callPV, SE = get_MC_sim(S, K, T, r, vol)
print("Monte Carlo evaluation of call option: ${0} with SE {1}".format(np.round(callPV, 2), np.round(SE, 2)))
print("Black Scholes evaluation of call option: ${0}".format(np.round(get_bs_price(S, K, T, r, vol), 2)))
