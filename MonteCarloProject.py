import numpy as np
from scipy.stats import norm

# Black-Scholes formula for European call and put options
def get_bs_price(S, K, T, r, vol, option_type="call"):
    """
    Calculate the price of a European option using the Black-Scholes formula.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate
        vol (float): Volatility (standard deviation of stock returns)
        option_type (str): "call" or "put"

    Returns:
        float: Option price
    """
    d1 = (np.log(S / K) + (r + vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)
    elif option_type == "put":
        return np.exp(-r * T) * K * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

# Monte Carlo simulation for European call and put options
def get_MC_sim(S, K, T, r, vol, N, M, option_type="call"):
    """
    Calculate the price of a European option using Monte Carlo simulation.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free interest rate
        vol (float): Volatility (standard deviation of stock returns)
        N (int): Number of time steps
        M (int): Number of simulations
        option_type (str): "call" or "put"

    Returns:
        tuple: (Option price, Standard error)
    """
    dt = T / N
    nudt = (r - 0.5 * vol**2) * dt
    volsdt = vol * np.sqrt(dt)
    lnS = np.log(S)

    X = np.random.normal(size=(N, M))
    dlnSt = nudt + volsdt * X
    lnSt = lnS + np.cumsum(dlnSt, axis=0)
    endPrices = np.exp(lnSt)

    if option_type == "call":
        endValues = np.maximum(0, endPrices[-1] - K)
    elif option_type == "put":
        endValues = np.maximum(0, K - endPrices[-1])
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    optionFV = np.mean(endValues)
    optionPV = optionFV * np.exp(-r * T)

    sx = np.sqrt(np.sum((endValues - optionFV) ** 2) / (M - 1))
    SE = sx / np.sqrt(M)
    return optionPV, SE

# Example usage
S = 232.98  # Stock price
K = 240     # Strike price
T = 0.1     # Time to expiration (in years)
r = 0.043   # Risk-free rate
vol = 0.2219  # Volatility
N = 100     # Number of time steps
M = 1000    # Number of simulations

callPV, callSE = get_MC_sim(S, K, T, r, vol, N, M, option_type="call")
putPV, putSE = get_MC_sim(S, K, T, r, vol, N, M, option_type="put")

print(f"Monte Carlo Call Option Price: ${callPV:.2f} with SE {callSE:.2f}")
print(f"Monte Carlo Put Option Price: ${putPV:.2f} with SE {putSE:.2f}")
print(f"Black-Scholes Call Option Price: ${get_bs_price(S, K, T, r, vol, option_type='call'):.2f}")
print(f"Black-Scholes Put Option Price: ${get_bs_price(S, K, T, r, vol, option_type='put'):.2f}")