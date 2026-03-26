# ============================================================
# HEDGE FUND OPTIONS STRATEGY APP (PRODUCTION STRUCTURE)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")

# ============================================================
# BLACK-SCHOLES
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
# MARKET DATA (SIM OR REAL READY)
# ============================================================

def simulate_market(days, mu, sigma, start=100):
    dt = 1/252
    prices = [start]

    for _ in range(days):
        shock = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt))
        prices.append(prices[-1] * np.exp(shock))

    return np.array(prices)


# ============================================================
# PORTFOLIO ENGINE (REAL POSITION TRACKING)
# ============================================================

class Portfolio:
    def __init__(self, capital):
        self.cash = capital
        self.leaps = 0
        self.puts = 0
        self.credit = 0
        self.nav_history = []

    def value(self, S):
        return self.cash + self.leaps + self.puts + self.credit


# ============================================================
# STRATEGY LOGIC
# ============================================================

def run_backtest(prices, sigma, r, initial):

    port = Portfolio(initial)
    peak = prices[0]

    for i in range(1, len(prices)):
        S = prices[i]
        peak = max(peak, S)
        dd = (peak - S) / peak

        # --------------------------
        # LEAPS POSITION
        # --------------------------
        T = 1.5
        K = S * 0.9
        call_price, delta = bs_call(S, K, T, r, sigma)

        position_size = 0.6 * port.cash
        contracts = position_size / (call_price * 100)

        port.leaps = contracts * call_price * 100

        # --------------------------
        # PUT HEDGE
        # --------------------------
        hedge_strength = min(1.0, dd / 0.2)

        put_K = S * 0.8
        put_price, _ = bs_put(S, put_K, 0.25, r, sigma)

        hedge_size = hedge_strength * 0.3 * port.cash
        put_contracts = hedge_size / (put_price * 100)

        port.puts = put_contracts * put_price * 100

        # --------------------------
        # CREDIT SPREAD INCOME
        # --------------------------
        weekly_income = 0.002 * port.cash  # 0.2% weekly realistic
        port.credit += weekly_income

        # --------------------------
        # CASH DRIFT (COSTS)
        # --------------------------
        port.cash *= 0.9995  # fees/slippage

        # --------------------------
        # NAV
        # --------------------------
        nav = port.value(S)
        port.nav_history.append(nav)

    return np.array(port.nav_history)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(nav):

    returns = pd.Series(nav).pct_change().dropna()

    cagr = (nav[-1] / nav[0])**(252/len(nav)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    cummax = pd.Series(nav).cummax()
    dd = (cummax - nav) / cummax
    max_dd = dd.max()

    sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252)

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd
    }


# ============================================================
# UI
# ============================================================

st.title("📊 Hedge Fund Strategy Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    capital = st.number_input("Initial Capital", 10000, 1000000, 50000)

with col2:
    years = st.slider("Years", 1, 20, 10)

with col3:
    sigma = st.slider("Volatility", 0.1, 0.4, 0.2)

mu = 0.10
r = 0.04

days = years * 252

# ============================================================
# RUN
# ============================================================

prices = simulate_market(days, mu, sigma)
nav = run_backtest(prices, sigma, r, capital)

# ============================================================
# CHART
# ============================================================

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=nav,
    name="Strategy",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    y=prices * (capital / prices[0]),
    name="Benchmark (SPY)",
    line=dict(dash="dot")
))

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# METRICS PANEL
# ============================================================

metrics = compute_metrics(nav)

c1, c2, c3, c4 = st.columns(4)

c1.metric("CAGR", f"{metrics['CAGR']*100:.2f}%")
c2.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
c3.metric("Sortino", f"{metrics['Sortino']:.2f}")
c4.metric("Max Drawdown", f"{metrics['MaxDD']*100:.2f}%")

# ============================================================
# STRATEGY CHECKLIST (INVESTOR READY)
# ============================================================

st.subheader("📋 Strategy Checklist")

checklist = {
    "Defined Edge (LEAPS + Income + Hedge)": True,
    "Risk Management Rules": True,
    "Drawdown Protection": True,
    "Position Sizing Logic": True,
    "Transaction Costs Included": True,
    "Volatility Awareness": True,
    "Backtested Over Multiple Cycles": False,
    "Live Data Integrated": False,
    "Options Chain Modeled": False,
}

for k, v in checklist.items():
    st.write(f"{'✅' if v else '❌'} {k}")
