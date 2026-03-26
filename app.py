# ============================================================
# INSTITUTIONAL OPTIONS STRATEGY DASHBOARD
# (LEAPS + PUT HEDGE + CREDIT SPREAD ENGINE)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import math

st.set_page_config(page_title="Institutional Options Fund", layout="wide")

# ============================================================
# BLACK-SCHOLES (STANDARDIZED)
# ============================================================

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0), 1.0 if S > K else 0.0

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    delta = norm.cdf(d1)
    return price, delta


def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0), -1.0 if K > S else 0.0

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    delta = norm.cdf(d1) - 1
    return price, delta


# ============================================================
# DELTA TARGET STRIKE FINDER (FIXED + STABLE)
# ============================================================

def find_strike_for_delta(S, T, r, sigma, target_delta, option="call"):
    low, high = S * 0.5, S * 1.5

    for _ in range(100):
        mid = (low + high) / 2

        if option == "call":
            _, d = bs_call(S, mid, T, r, sigma)
        else:
            _, d = bs_put(S, mid, T, r, sigma)
            d = abs(d)

        if d > target_delta:
            low = mid
        else:
            high = mid

    return (low + high) / 2


# ============================================================
# MARKET SIMULATION (GBM)
# ============================================================

def simulate_market(years, mu, sigma, start=100):
    steps = years * 252
    dt = 1/252

    prices = [start]

    for _ in range(steps):
        shock = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt))
        prices.append(prices[-1] * np.exp(shock))

    return np.array(prices)


# ============================================================
# STRATEGY ENGINE
# ============================================================

def run_strategy(prices, sigma, r, initial_capital):

    cash = initial_capital
    leaps_val = 0
    puts_val = 0
    cs_pnl = 0

    nav_history = []

    peak = prices[0]

    for i in range(1, len(prices)):
        S = prices[i]
        peak = max(peak, S)

        spy_dd = (peak - S) / peak

        # -----------------------------
        # LEAPS ENGINE
        # -----------------------------
        T = 1.5
        strike = find_strike_for_delta(S, T, r, sigma, 0.75)
        price, delta = bs_call(S, strike, T, r, sigma)

        leaps_val = price * 100

        # FIXED NOTIONAL CALCULATION
        leaps_notional = leaps_val / max(delta, 0.01)

        # -----------------------------
        # PUT HEDGE ENGINE
        # -----------------------------
        put_strike = S * 0.8
        put_price, _ = bs_put(S, put_strike, 0.25, r, sigma)

        # FIXED MULTIPLIER
        multiplier = min(15, 5 + (spy_dd / 0.20) * 10)

        puts_val = put_price * multiplier * 100

        # -----------------------------
        # CREDIT SPREAD ENGINE
        # -----------------------------
        coverage = 0.3

        contract_size = 100
        n_spreads = int((leaps_notional * coverage) / contract_size)

        credit = 2.0  # simplified premium
        cs_pnl += n_spreads * credit

        # -----------------------------
        # NAV CALCULATION (FIXED)
        # -----------------------------
        nav = leaps_val + puts_val + cash + cs_pnl

        nav_history.append(nav)

    return nav_history


# ============================================================
# UI CONTROLS
# ============================================================

st.title("Institutional Options Strategy Dashboard")

initial = st.sidebar.number_input("Initial Capital", 10000, 1000000, 50000)
years = st.sidebar.slider("Years", 1, 20, 10)
mu = st.sidebar.slider("Market Return %", 5.0, 15.0, 10.0) / 100
sigma = st.sidebar.slider("Volatility %", 10.0, 40.0, 20.0) / 100
r = 0.04

# ============================================================
# RUN SIMULATION
# ============================================================

prices = simulate_market(years, mu, sigma)
nav = run_strategy(prices, sigma, r, initial)

# ============================================================
# PLOT
# ============================================================

fig = go.Figure()
fig.add_trace(go.Scatter(y=nav, name="Strategy NAV"))
fig.add_trace(go.Scatter(y=prices * (initial / prices[0]), name="SPY Benchmark"))

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# METRICS
# ============================================================

returns = pd.Series(nav).pct_change().dropna()

cagr = (nav[-1] / nav[0])**(1/years) - 1
sharpe = returns.mean() / returns.std() * np.sqrt(252)
max_dd = (pd.Series(nav).cummax() - nav).max() / max(nav)

st.subheader("Performance Metrics")

st.write("CAGR:", round(cagr * 100, 2), "%")
st.write("Sharpe Ratio:", round(sharpe, 2))
st.write("Max Drawdown:", round(max_dd * 100, 2), "%")
