import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm
from MonteCarloProject import *
import time
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Option Pricing Tool", layout="wide")

st.title("Option Pricing Calculator")
st.write("Compare Black-Scholes and Monte Carlo pricing methods for European options")

# Sidebar Inputs
st.sidebar.header("Option Parameters")

option_type = st.sidebar.radio("Option Type", ["call", "put"])

col1, col2 = st.sidebar.columns(2)
with col1:
    S = st.slider("Stock Price ($)", 50.0, 500.0, 232.98, 0.01)
    K = st.slider("Strike Price ($)", 50.0, 500.0, 240.0, 0.01)
    T = st.slider("Time to Expiration (years)", 0.01, 2.0, 0.1, 0.01)

with col2:
    r = st.slider("Risk-free Rate (%)", 0.0, 10.0, 4.3, 0.1) / 100
    vol = st.slider("Volatility (%)", 1.0, 100.0, 22.19, 0.01) / 100

st.sidebar.header("Monte Carlo Parameters")
N = st.sidebar.slider("Number of Time Steps", 10, 500, 100, 10)
M = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, 100)

st.write("### Selected Parameters")
params_df = pd.DataFrame({
    'Parameter': ['Stock Price', 'Strike Price', 'Time to Expiration', 'Risk-free Rate', 'Volatility'],
    'Value': [f"${S:.2f}", f"${K:.2f}", f"{T:.2f} years", f"{r*100:.2f}%", f"{vol*100:.2f}%"]
})
st.table(params_df)

bs_price = get_bs_price(S, K, T, r, vol, option_type)
mc_price, mc_se, price_paths = get_MC_sim(S, K, T, r, vol, N, M, option_type, return_paths=True)

st.write("### Pricing Results")
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Black-Scholes Price", value=f"${bs_price:.4f}")

with col2:
    st.metric(label="Monte Carlo Price", value=f"${mc_price:.4f}", delta=f"SE: {mc_se:.4f}")

# Monte Carlo Convergence
if st.button("Run New Monte Carlo Simulation"):
    progress_bar = st.progress(0)
    sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    if M > 10000:
        sample_sizes.append(M)

    results = []

    for i, size in enumerate(sample_sizes):
        if size > M:
            break
        mc_p, mc_s = get_MC_sim(S, K, T, r, vol, N, size, option_type)
        results.append([size, mc_p, mc_s])
        progress = (i + 1) / len(sample_sizes)
        progress_bar.progress(progress)
        time.sleep(0.1)

    st.write("### Monte Carlo Convergence")
    results_df = pd.DataFrame(results, columns=["Simulations", "Price", "Std Error"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df["Simulations"], results_df["Price"], marker='o', linewidth=2)
    ax.axhline(y=bs_price, color='r', linestyle='--', label='Black-Scholes Price')
    ax.fill_between(
        results_df["Simulations"],
        results_df["Price"] - 1.96 * results_df["Std Error"],
        results_df["Price"] + 1.96 * results_df["Std Error"],
        alpha=0.2
    )
    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Option Price ($)')
    ax.set_title('Monte Carlo Price Convergence')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Monte Carlo Simulation Paths
st.write("### Monte Carlo Simulation Visualization")

fig, ax = plt.subplots(figsize=(10, 6))
path_cmap = LinearSegmentedColormap.from_list(
    "path_cm", ["#e0e0e0", "#2171b5"] if option_type == "call" else ["#e0e0e0", "#cb181d"]
)
itm_color = "#41ab5d"

num_paths_to_plot = min(100, M)
time_points = np.linspace(0, T, N)

for i in range(num_paths_to_plot):
    end_price = price_paths[-1, i]
    intensity = min(1.0, abs(end_price - K) / (K * 0.3))
    color = path_cmap(intensity)
    ax.plot(time_points, price_paths[:, i], color=color, linewidth=0.5, alpha=0.4)

itm_indices = np.where(price_paths[-1, :num_paths_to_plot] > K)[0] if option_type == "call" else np.where(price_paths[-1, :num_paths_to_plot] < K)[0]
for idx in itm_indices:
    ax.plot(time_points, price_paths[:, idx], color=itm_color, linewidth=0.8, alpha=0.7)

ax.axhline(y=K, color='#d95f0e', linestyle='--', linewidth=1.5, label=f'Strike (${K:.2f})')
ax.axhline(y=S, color='#756bb1', linestyle='-', linewidth=1.5, label=f'Current (${S:.2f})')

if option_type == "call":
    ax.axhspan(K, ax.get_ylim()[1], alpha=0.1, color='green')
else:
    ax.axhspan(ax.get_ylim()[0], K, alpha=0.1, color='green')

ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price ($)')
ax.set_title(f'Monte Carlo Stock Price Paths for {option_type.capitalize()} Option')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Final Stock Price Distribution
st.write("### Final Stock Price Distribution")

end_prices = price_paths[-1, :]
if option_type == "call":
    profit_condition = end_prices > K
else:
    profit_condition = end_prices < K

profit_prices = end_prices[profit_condition]
loss_prices = end_prices[~profit_condition]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(profit_prices, bins=50, alpha=0.7, color="#41ab5d", label="Profit")
ax.hist(loss_prices, bins=50, alpha=0.7, color="#cb181d", label="Loss")
ax.axvline(x=K, color='#d95f0e', linestyle='--', linewidth=2, label=f'Strike (${K:.2f})')
ax.axvline(x=S, color='#756bb1', linestyle='-', linewidth=2, label=f'Current (${S:.2f})')

ax.set_xlabel('Stock Price at Expiration ($)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Final Stock Prices')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Log-Normal Distribution Analysis
st.write("### Log-Normal Distribution Analysis")

mu = np.log(S) + (r - 0.5 * vol**2) * T
sigma = vol * np.sqrt(T)
log_normal_samples = np.exp(np.random.normal(mu, sigma, 10000))

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(end_prices, bins=50, alpha=0.5, color="#2171b5", label='Monte Carlo', density=True)
ax.hist(log_normal_samples, bins=50, alpha=0.5, color="#fc8d59", label='Theoretical', density=True)
ax.axvline(x=K, color='#d95f0e', linestyle='--')
ax.axvline(x=S, color='#756bb1', linestyle='-')
ax.set_xlabel('Stock Price at Expiration ($)')
ax.set_ylabel('Probability Density')
ax.set_title('Comparison with Theoretical Log-Normal Distribution')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Greeks Radar Chart
st.write("### Option Greeks Visualization")

def calculate_greeks(S, K, T, r, vol, option_type):
    d1 = (np.log(S/K) + (r + vol**2/2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    theta = (-S*norm.pdf(d1)*vol/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) if option_type == "call" else (-S*norm.pdf(d1)*vol/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))
    vega = S*np.sqrt(T)*norm.pdf(d1)
    rho = (K*T*np.exp(-r*T)*norm.cdf(d2)) if option_type == "call" else (-K*T*np.exp(-r*T)*norm.cdf(-d2))
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta/365,
        'Vega': vega/100,
        'Rho': rho/100
    }

greeks = calculate_greeks(S, K, T, r, vol, option_type)

# Radar chart
labels = list(greeks.keys())
stats = list(greeks.values())

# Prepare angles
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# Close the stats circle
stats += stats[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, stats, color='b', linewidth=2)
ax.fill(angles, stats, color='b', alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title('Option Greeks Radar Chart')
st.pyplot(fig)