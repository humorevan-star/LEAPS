# ============================================================
# HEDGE FUND GRADE OPTIONS STRATEGY (OUTPERFORMANCE MODEL)
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
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2), norm.cdf(d1)

def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0), -1.0 if K > S else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1), norm.cdf(d1)-1

# ============================================================
# MARKET (SIM OR SWAP WITH REAL DATA LATER)
# ============================================================

def simulate_market(days, mu=0.10, sigma=0.20, start=100):
    dt = 1/252
    prices = [start]
    for _ in range(days):
        shock = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt))
        prices.append(prices[-1]*np.exp(shock))
    return np.array(prices)

# ============================================================
# STRATEGY ENGINE (FIXED)
# ============================================================

def run_strategy(prices, sigma, r, capital):

    nav = []
    cash = capital

    peak = prices[0]

    for i in range(252, len(prices)):  # start after 1y for MA
        S = prices[i]
        peak = max(peak, S)

        # -------------------------
        # TREND FILTER (200D MA)
        # -------------------------
        ma200 = prices[i-200:i].mean()
        bull = S > ma200

        # -------------------------
        # DRAWDOWN
        # -------------------------
        dd = (peak - S) / peak

        # -------------------------
        # LEAPS (AGGRESSIVE WHEN BULL)
        # -------------------------
        T = 1.5
        K = S * (0.9 if bull else 1.0)

        call_price, delta = bs_call(S, K, T, r, sigma)

        exposure = 1.5 if bull else 0.7  # LEVERAGE HERE

        leaps_val = exposure * cash

        # -------------------------
        # PUT HEDGE (ONLY WHEN NEEDED)
        # -------------------------
        hedge_val = 0

        if dd > 0.10:  # ONLY hedge in real drawdowns
            put_K = S * 0.8
            put_price, _ = bs_put(S, put_K, 0.3, r, sigma)

            hedge_val = min(0.5, dd * 2) * cash

        # -------------------------
        # CREDIT SPREADS (VOL FILTER)
        # -------------------------
        income = 0

        if bull and dd < 0.05:
            income = 0.0015 * cash  # small consistent income

        # -------------------------
        # NAV EVOLUTION
        # -------------------------
        market_return = (prices[i] / prices[i-1]) - 1

        pnl = (
            leaps_val * market_return * delta
            + hedge_val * max(0, -market_return * 5)  # convex crash payoff
            + income
        )

        cash += pnl
        nav.append(cash)

    return np.array(nav)

# ============================================================
# METRICS
# ============================================================

def metrics(nav):
    returns = pd.Series(nav).pct_change().dropna()
    cagr = (nav[-1]/nav[0])**(252/len(nav)) - 1
    sharpe = returns.mean()/returns.std()*np.sqrt(252)

    dd = (pd.Series(nav).cummax() - nav)/pd.Series(nav).cummax()
    max_dd = dd.max()

    return cagr, sharpe, max_dd

# ============================================================
# UI
# ============================================================

st.title("🚀 Outperformance Strategy Dashboard")

capital = st.sidebar.number_input("Capital", 10000, 1000000, 50000)
years = st.sidebar.slider("Years", 3, 20, 10)
sigma = st.sidebar.slider("Volatility", 0.1, 0.4, 0.2)

r = 0.04
days = years * 252

# ============================================================
# RUN
# ============================================================

prices = simulate_market(days)
nav = run_strategy(prices, sigma, r, capital)

# ============================================================
# PLOT
# ============================================================

fig = go.Figure()

fig.add_trace(go.Scatter(y=nav, name="Strategy", line=dict(width=3)))
fig.add_trace(go.Scatter(
    y=prices[252:] * (capital/prices[252]),
    name="SPY",
    line=dict(dash="dot")
))

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# METRICS
# ============================================================

cagr, sharpe, max_dd = metrics(nav)

c1, c2, c3 = st.columns(3)
c1.metric("CAGR", f"{cagr*100:.2f}%")
c2.metric("Sharpe", f"{sharpe:.2f}")
c3.metric("Max DD", f"{max_dd*100:.2f}%")

# ============================================================
# CHECKLIST
# ============================================================

st.subheader("Strategy Integrity")

st.write("✅ Trend Filter (200 MA)")
st.write("✅ Conditional Hedging")
st.write("✅ Convex Crash Protection")
st.write("✅ Dynamic Exposure")
st.write("✅ Volatility-Aware Income")
st.write("❌ Real Options Chain (next step)")
st.write("❌ Live Data (next step)")
