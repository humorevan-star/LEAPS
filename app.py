# ==============================
# SPY LEAPS vs SPXL vs VOO APP
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

st.set_page_config(page_title="SPY LEAPS vs SPXL vs VOO", layout="wide")

# =============================================================================
# MATH CORE
# =============================================================================

def leaps_annual_return(ann_r, itm_pct, theta_ann=0.04, roll_cost=0.005):
    old_prem = itm_pct * 1.20
    new_intr = max(ann_r + itm_pct, 0.0)
    new_prem = new_intr * 1.20
    return (new_prem / old_prem) - 1.0 - theta_ann - roll_cost


def vol_drag(lev, sigma):
    return lev * (lev - 1) / 2 * sigma**2


def simulate(years, monthly, ann_r, ann_s, initial, itm_pct, theta_ann=0.04):
    leaps_r = leaps_annual_return(ann_r, itm_pct, theta_ann)
    drag3 = vol_drag(3.0, ann_s)
    spxl_r = 3.0 * ann_r - drag3 - 0.0091
    voo_r = ann_r - 0.0003

    leaps, voo, spxl = initial, initial, initial
    invested = initial

    la, va, sa, ia = [initial], [initial], [initial], [initial]

    for _ in range(years):
        dca = monthly * 12
        leaps = max(leaps * (1 + leaps_r), 0) + dca
        voo = voo * (1 + voo_r) + dca
        spxl = max(spxl * (1 + spxl_r), 0) + dca
        invested += dca

        la.append(round(leaps))
        va.append(round(voo))
        sa.append(round(spxl))
        ia.append(round(invested))

    return {"leaps": la, "voo": va, "spxl": sa, "invested": ia, "leaps_r": leaps_r}


# =============================================================================
# BLACK-SCHOLES
# =============================================================================

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    delta = norm_cdf(d1)
    return max(price, 0.0), delta


def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0), (-1.0 if K > S else 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    delta = norm_cdf(d1) - 1
    return max(price, 0.0), delta


# ✅ FIXED FUNCTION
def find_strike_for_delta(S, T, r, sigma, target_delta,
                         option_type="call",
                         lo_pct=0.5, hi_pct=1.5, steps=200):
    lo, hi = S * lo_pct, S * hi_pct
    for _ in range(steps):
        mid = (lo + hi) / 2
        if option_type == "call":
            _, d = black_scholes_call(S, mid, T, r, sigma)
        else:
            _, d = black_scholes_put(S, mid, T, r, sigma)
            d = abs(d)

        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# =============================================================================
# UI
# =============================================================================

st.title("SPY LEAPS vs SPXL vs VOO")

initial = st.sidebar.number_input("Initial Investment", 1000, 1000000, 20000)
monthly = st.sidebar.number_input("Monthly Contribution", 0, 10000, 500)
years = st.sidebar.slider("Years", 1, 30, 20)

ann_r = st.sidebar.slider("SPY Return %", 5.0, 18.0, 10.0) / 100
ann_vol = st.sidebar.slider("Volatility %", 8.0, 35.0, 18.0) / 100
itm = st.sidebar.slider("ITM %", 10, 40, 25) / 100

results = simulate(years, monthly, ann_r, ann_vol, initial, itm)

# =============================================================================
# CHART
# =============================================================================

x = list(range(years + 1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=results["leaps"], name="LEAPS"))
fig.add_trace(go.Scatter(x=x, y=results["spxl"], name="SPXL"))
fig.add_trace(go.Scatter(x=x, y=results["voo"], name="VOO"))

st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# IRON HARVEST (LIGHT VERSION)
# =============================================================================

st.subheader("Iron Harvest (Simplified Test)")

S = 100
T = 0.5
r = 0.04
sigma = ann_vol

strike = find_strike_for_delta(S, T, r, sigma, 0.25)
price, delta = black_scholes_call(S, strike, T, r, sigma)

st.write("Target 0.25 delta strike:", round(strike, 2))
st.write("Option price:", round(price, 2))
st.write("Delta:", round(delta, 3))
