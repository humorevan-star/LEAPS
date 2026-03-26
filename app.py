import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="SPY LEAPS vs SPXL vs VOO",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# MATH CORE
# =============================================================================

def leaps_annual_return(ann_r: float, itm_pct: float,
                         theta_ann: float = 0.04, roll_cost: float = 0.005) -> float:
    """
    Annual return on capital for deep-ITM LEAPS.
    Premium = itm_pct * 1.20 (intrinsic + 20% extrinsic)
    New premium after ann_r move = max(ann_r + itm_pct, 0) * 1.20
    Return = new_premium / old_premium - 1 - theta - roll_cost
    """
    old_prem = itm_pct * 1.20
    new_intr  = max(ann_r + itm_pct, 0.0)
    new_prem  = new_intr * 1.20
    return (new_prem / old_prem) - 1.0 - theta_ann - roll_cost


def vol_drag(lev: float, sigma: float) -> float:
    return lev * (lev - 1) / 2 * sigma ** 2


def simulate(years: int, monthly: float, ann_r: float, ann_s: float,
             initial: float, itm_pct: float,
             theta_ann: float = 0.04, roll_cost: float = 0.005) -> dict:
    """
    Simulate all three strategies year-by-year.
    Returns dict of lists for each strategy + invested total.
    """
    leaps_r = leaps_annual_return(ann_r, itm_pct, theta_ann, roll_cost)
    drag3   = vol_drag(3.0, ann_s)
    spxl_r  = 3.0 * ann_r - drag3 - 0.0091
    voo_r   = ann_r - 0.0003

    leaps = initial
    voo   = initial
    spxl  = initial
    invested = initial

    la, va, sa, ia = [initial], [initial], [initial], [initial]

    for _ in range(years):
        dca = monthly * 12

        leaps = max(leaps * (1 + leaps_r), 0) + dca
        voo   = voo   * (1 + voo_r)           + dca
        spxl  = max(spxl * (1 + spxl_r), 0)   + dca
        invested += dca

        la.append(round(leaps))
        va.append(round(voo))
        sa.append(round(spxl))
        ia.append(round(invested))

    return {"leaps": la, "voo": va, "spxl": sa, "invested": ia, "leaps_r": leaps_r}


def annual_returns_breakdown(ann_r: float, ann_s: float, itm_pct: float) -> dict:
    old_prem  = itm_pct * 1.20
    new_intr  = max(ann_r + itm_pct, 0.0)
    gross_r   = (new_intr * 1.20 / old_prem - 1) * 100
    theta_r   = 4.0   # %
    roll_r    = 0.5   # %
    net_r     = gross_r - theta_r - roll_r
    drag3     = vol_drag(3.0, ann_s)
    spxl_r    = (3.0 * ann_r - drag3 - 0.0091) * 100
    voo_r     = (ann_r - 0.0003) * 100
    return {
        "gross_r": gross_r, "theta_r": theta_r, "roll_r": roll_r,
        "net_r": net_r, "spxl_r": spxl_r, "voo_r": voo_r,
        "old_prem_pct": old_prem * 100,
    }


def leaps_drawdown(spy_crash: float, itm_pct: float) -> float:
    """
    Real option math drawdown.
    If crash pushes below strike → OTM, tiny time value only.
    Capped at -100%.
    """
    old_prem = itm_pct * 1.20
    new_intr = spy_crash + itm_pct
    if new_intr > 0:
        new_prem = new_intr * 1.20
    else:
        new_prem = max(0.02 * np.exp(spy_crash * 3), 0.0)
    return max(round((new_prem / old_prem - 1) * 100), -100)


def fmt_currency(v: float) -> str:
    a = abs(v)
    if a >= 1e9:  return f"${v/1e9:.2f}B"
    if a >= 1e6:  return f"${v/1e6:.2f}M"
    if a >= 1e3:  return f"${v:,.0f}"
    return f"${v:.0f}"


def kpi(col, label: str, value: str, sub: str = "", color: str = "#e8eaf0"):
    sub_html = f'<p style="margin:2px 0 0;font-size:11px;color:#4a5568;font-family:monospace;">{sub}</p>' if sub else ""
    col.markdown(
        f'<div style="background:#111318;padding:14px 16px;border-radius:6px;'
        f'border:1px solid rgba(255,255,255,0.07);text-align:center;">'
        f'<p style="margin:0 0 3px;font-size:9px;color:#4a5568;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:0.1em;">{label}</p>'
        f'<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;'
        f'color:{color};">{value}</p>{sub_html}</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## Strategy Controls")

initial_inv = st.sidebar.number_input(
    "Initial Investment ($)", value=20_000, min_value=0, step=1_000,
)
monthly_inv = st.sidebar.number_input(
    "Monthly DCA ($)", value=500, min_value=0, step=50,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Projection")
proj_years = st.sidebar.slider("Years", 3, 30, 20)

st.sidebar.markdown("---")
st.sidebar.markdown("### Market Assumptions")
ann_ret = st.sidebar.slider("SPY Annual Return (%)", 5.0, 18.0, 10.0, 0.5) / 100
ann_vol = st.sidebar.slider("Annual Volatility (%)", 8.0, 35.0, 18.0, 0.5) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### LEAPS Parameters")
itm_pct = st.sidebar.slider(
    "Strike Depth — ITM (%)", 10, 40, 25, 5,
    help="Strike = spot × (1 − ITM%). Deeper = safer but less leverage.",
) / 100
theta_ann = st.sidebar.slider(
    "Theta drag (%/yr on premium)", 2.0, 8.0, 4.0, 0.5,
    help="Annual theta cost as % of option premium. Deep ITM is lower.",
) / 100


# =============================================================================
# MAIN
# =============================================================================
st.title("SPY LEAPS vs SPXL vs VOO")
st.caption(
    f"${initial_inv:,.0f} initial · ${monthly_inv}/mo DCA · {proj_years}yr · "
    f"{ann_ret*100:.1f}% SPY return · {ann_vol*100:.1f}% vol · {itm_pct*100:.0f}% ITM"
)

# Ticker info row
c1, c2, c3 = st.columns(3)
c1.info("**SPY LEAPS** — Deep-ITM calls · Jan 2027/2028 · Strike = spot × (1−ITM%) · δ≈0.90 · Roll every 9mo")
c2.info("**SPXL** — Direxion Daily S&P 500 Bull 3× ETF · Daily reset · 0.91% expense")
c3.info("**VOO** — Vanguard S&P 500 ETF · 1× · 0.03% expense · benchmark")

st.markdown("")

# Run simulation
results  = simulate(proj_years, monthly_inv, ann_ret, ann_vol, initial_inv, itm_pct, theta_ann)
rets     = annual_returns_breakdown(ann_ret, ann_vol, itm_pct)
invested = results["invested"][-1]
lf, vf, sf = results["leaps"][-1], results["voo"][-1], results["spxl"][-1]

def cagr(v):
    if invested > 0 and proj_years > 0:
        return ((v / invested) ** (1 / proj_years) - 1) * 100
    return 0.0

# KPI strip
k = st.columns(4)
kpi(k[0], "SPY LEAPS final",  fmt_currency(lf), f"CAGR {cagr(lf):.1f}% · {results['leaps_r']*100:.1f}%/yr", "#EF9F27")
kpi(k[1], "SPXL final",       fmt_currency(sf), f"CAGR {cagr(sf):.1f}%",                                   "#378ADD")
kpi(k[2], "VOO final",        fmt_currency(vf), f"CAGR {cagr(vf):.1f}%",                                   "#1D9E75")
kpi(k[3], "LEAPS vs SPXL",    f"{lf/max(sf,1):.2f}x", f"vs VOO: {lf/max(vf,1):.2f}x",                     "#e8eaf0")

st.markdown("")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Growth",
    "📊 Annual Returns",
    "📉 Crash Drawdowns",
    "⚡ Capital Efficiency",
    "🏆 Iron Harvest Elite",
])

# ── TAB 1: GROWTH CHART ──────────────────────────────────────────────────────
with tab1:
    x = list(range(proj_years + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=results["leaps"], name="SPY LEAPS",
                             line=dict(color="#EF9F27", width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=results["spxl"],  name="SPXL",
                             line=dict(color="#378ADD", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=results["voo"],   name="VOO",
                             line=dict(color="#1D9E75", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=results["invested"], name="Total Invested",
                             line=dict(color="rgba(136,135,128,.5)", width=1, dash="dot")))
    fig.update_layout(
        template="plotly_dark", height=420,
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Year",
        yaxis=dict(tickprefix="$"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"LEAPS at {itm_pct*100:.0f}% ITM earns ~{results['leaps_r']*100:.1f}%/yr on capital. "
        f"SPXL earns {(3*ann_ret - vol_drag(3,ann_vol) - 0.0091)*100:.1f}%/yr after vol drag. "
        f"VOO earns {(ann_ret-0.0003)*100:.1f}%/yr."
    )

# ── TAB 2: ANNUAL RETURNS BREAKDOWN ──────────────────────────────────────────
with tab2:
    labels_r = ["LEAPS gross", "Theta cost", "LEAPS net", "SPXL net", "VOO net"]
    values_r = [
        rets["gross_r"],
        -rets["theta_r"],
        rets["net_r"],
        rets["spxl_r"],
        rets["voo_r"],
    ]
    colors_r = ["#EF9F27", "#E24B4A", "#BA7517", "#378ADD", "#1D9E75"]

    fig2 = go.Figure(go.Bar(
        x=labels_r, y=values_r,
        marker_color=colors_r,
        text=[f"{v:.1f}%" for v in values_r],
        textposition="outside",
    ))
    fig2.update_layout(
        template="plotly_dark", height=380,
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(ticksuffix="%", title="% return on capital"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        f"At **{itm_pct*100:.0f}% ITM**, premium ≈ **{rets['old_prem_pct']:.0f}%** of SPY price. "
        f"Gross gain **{rets['gross_r']:.0f}%**, theta **−{rets['theta_r']:.0f}%**, "
        f"net **{rets['net_r']:.0f}%** per year on capital. "
        f"Deeper ITM = higher gross gain but more premium = more theta drag."
    )

# ── TAB 3: CRASH DRAWDOWNS ────────────────────────────────────────────────────
with tab3:
    crashes = [
        {"label": "2008 GFC (−57%)",   "spy": -0.57, "spxl": -97, "voo": -57},
        {"label": "2020 Covid (−34%)", "spy": -0.34, "spxl": -76, "voo": -34},
        {"label": "2022 Bear (−25%)",  "spy": -0.25, "spxl": -78, "voo": -25},
        {"label": "2018 Q4 (−19%)",    "spy": -0.19, "spxl": -40, "voo": -19},
    ]
    leaps_dds = [leaps_drawdown(c["spy"], itm_pct) for c in crashes]
    survived  = sum(1 for v in leaps_dds if v > -95)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="SPY LEAPS", x=[c["label"] for c in crashes], y=leaps_dds,
        marker_color="#EF9F27",
        text=[f"{v}%" for v in leaps_dds], textposition="outside",
    ))
    fig3.add_trace(go.Bar(
        name="SPXL", x=[c["label"] for c in crashes], y=[c["spxl"] for c in crashes],
        marker_color="#378ADD",
        text=[f"{c['spxl']}%" for c in crashes], textposition="outside",
    ))
    fig3.add_trace(go.Bar(
        name="VOO", x=[c["label"] for c in crashes], y=[c["voo"] for c in crashes],
        marker_color="#1D9E75",
        text=[f"{c['voo']}%" for c in crashes], textposition="outside",
    ))
    fig3.update_layout(
        template="plotly_dark", height=400, barmode="group",
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(ticksuffix="%", range=[-110, 15], title="Drawdown %"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig3, use_container_width=True)

    itm100 = round(itm_pct * 100)
    msg = (
        f"At **{itm100}% ITM**, your strike sits at **{100-itm100}%** of entry price. "
        f"Crashes deeper than **−{itm100}%** push the option out-of-the-money. "
        f"**{survived}/4** historical crashes keep intrinsic value at this ITM level. "
    )
    if survived < 4:
        msg += (
            f"The {4-survived} that go near zero still lose a **maximum of 100% of premium paid** — "
            "not more. Roll into a new contract at the lower strike to participate in the recovery. "
            "SPXL at −97% needs a 3,233% gain just to break even."
        )
    else:
        msg += "All 4 crashes survive with intrinsic value intact."
    st.info(msg)

# ── TAB 4: CAPITAL EFFICIENCY ─────────────────────────────────────────────────
with tab4:
    prem500  = itm_pct * 1.20 * 500
    notional = round(500 / prem500 * 500)
    drag3_   = vol_drag(3.0, ann_vol)
    leaps_g  = round(500 * rets["net_r"] / 100)
    spxl_g   = round(500 * max(rets["spxl_r"] / 100, 0))
    voo_g    = round(500 * rets["voo_r"] / 100)

    instruments = ["$500 → VOO", "$500 → SPXL", "$500 → SPY LEAPS"]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        name="Notional controlled",
        x=instruments, y=[500, 500, notional],
        marker_color="rgba(136,135,128,.25)",
        text=[f"${v:,}" for v in [500, 500, notional]],
        textposition="outside",
    ))
    fig4.add_trace(go.Bar(
        name="Net dollar gain (10% SPY yr)",
        x=instruments, y=[voo_g, spxl_g, leaps_g],
        marker_color=["#1D9E75", "#378ADD", "#EF9F27"],
        text=[f"${v}" for v in [voo_g, spxl_g, leaps_g]],
        textposition="outside",
    ))
    fig4.update_layout(
        template="plotly_dark", height=380, barmode="group",
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(tickprefix="$", title="Dollars"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.info(
        f"At **{itm100}% ITM**, $500 of premium controls **${notional:,}** notional. "
        f"On a 10% SPY year: VOO nets **${voo_g}**, SPXL nets **${spxl_g}**, "
        f"LEAPS nets **${leaps_g}** — after theta and roll costs."
    )




# =============================================================================
# IRON HARVEST ELITE ENGINE
# =============================================================================

from scipy.stats import norm as _norm

def black_scholes_call(S, K, T, r, sigma):
    """Standard Black-Scholes call price and delta."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * _norm.cdf(d1) - K * np.exp(-r * T) * _norm.cdf(d2)
    delta = float(_norm.cdf(d1))
    return max(price, 0.0), delta


def black_scholes_put(S, K, T, r, sigma):
    """Standard Black-Scholes put price and delta."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0), (_norm.cdf(-1.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * _norm.cdf(-d2) - S * _norm.cdf(-d1)
    delta = float(_norm.cdf(d1) - 1.0)
    return max(price, 0.0), delta


def find_strike_for_delta(S, T, r, sigma, target_delta, option_type="call",
                           lo_pct=0.5, hi_pct=1.5, steps=200):
    """Binary-search strike for a target delta using Black-Scholes."""
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


@st.cache_data(ttl=3600)
def run_iron_harvest_backtest(
    starting_capital: float,
    years: int,
    ann_ret: float,
    ann_vol: float,
    risk_free: float = 0.045,
    leaps_alloc: float = 0.75,
    cash_alloc: float  = 0.15,
    puts_alloc: float  = 0.05,
    commission: float  = 0.65,   # $ per contract
    slippage: float    = 0.002,  # 0.2% of option value
) -> dict:
    """
    Year-by-year Iron Harvest Elite simulation.
    Engines:
      A — LEAPS: deep-ITM calls, delta-targeted by VIX regime
      B — Call spreads: PMCC overlay, regime-dependent coverage
      C — Tail puts: 20-30% OTM, 6-9mo, monetise on -20%+ drop
    Cash deployed mechanically on SPY drawdowns from ATH.
    """
    np.random.seed(42)
    n_steps = years * 12   # monthly granularity

    # Simulate SPY path (GBM monthly)
    dt = 1 / 12
    monthly_r = ann_ret / 12
    monthly_s = ann_vol / np.sqrt(12)
    spy_returns = np.random.normal(monthly_r - 0.5 * monthly_s**2, monthly_s, n_steps)
    spy_path = np.cumprod(1 + spy_returns) * 100.0   # start at $100

    # Simulate VIX path (mean-reverting around 18, correlated to -SPY)
    vix_path = np.zeros(n_steps)
    vix = 18.0
    for i, r in enumerate(spy_returns):
        shock = -r * 200 + np.random.normal(0, 1.5)
        vix = max(min(vix * 0.92 + 18 * 0.08 + shock, 80), 9)
        vix_path[i] = vix

    # Portfolio state
    nav        = starting_capital
    leaps_val  = nav * leaps_alloc
    cash       = nav * cash_alloc
    puts_val   = nav * puts_alloc
    cs_pnl     = 0.0          # cumulative call spread P&L
    cs_wins    = 0
    cs_trades  = 0

    nav_series = [nav]
    leaps_series, cash_series, puts_series = [leaps_val], [cash], [puts_val]
    cs_pnl_series = [0.0]
    leverage_series = [leaps_val / nav]
    vix_series = list(vix_path)

    spy_ath = spy_path[0]
    drawdown_deployed = 0.0  # cumulative cash deployed into drawdown
    call_selling_halted = False

    for i in range(1, n_steps):
        S   = spy_path[i]
        S_p = spy_path[i - 1]
        vix = vix_path[i]
        spy_ret = spy_path[i] / spy_path[i - 1] - 1
        spy_ath = max(spy_ath, S)

        # ── ENGINE A: LEAPS growth (delta-targeted by VIX) ─────────────────
        # Effective LEAPS leverage scales with VIX regime
        if vix < 15:
            leaps_delta = 0.775   # midpoint 75-80
        elif vix < 25:
            leaps_delta = 0.825   # midpoint 80-85
        else:
            leaps_delta = 0.925   # midpoint 90-95

        # Effective LEAPS return = delta * SPY return - theta_daily * 30days
        theta_monthly = 0.003    # ~3.6%/yr on premium
        leaps_monthly_r = leaps_delta * spy_ret - theta_monthly
        leaps_val = max(leaps_val * (1 + leaps_monthly_r), 0.0)

        # ── ENGINE B: CALL SPREAD OVERLAY ───────────────────────────────────
        # Determine coverage fraction by regime
        spy_vs_50sma = S / np.mean(spy_path[max(0, i - 6):i]) - 1  # ~2mo proxy
        if vix > 25:
            coverage = 0.90
        elif abs(spy_vs_50sma) < 0.02:
            coverage = 0.75
        elif spy_vs_50sma > 0.02:
            coverage = 0.50
        else:
            coverage = 0.80

        if call_selling_halted:
            coverage = 0.0

        # Sell 0.25 delta call, buy $75 higher (midpoint $50-$100 width)
        T_cs  = 45 / 365            # ~45 DTE for spreads
        sell_strike = find_strike_for_delta(S, T_cs, risk_free / 12, ann_vol, 0.25)
        buy_strike  = sell_strike + 75
        sell_price, _ = black_scholes_call(S, sell_strike, T_cs, risk_free / 12, ann_vol)
        buy_price,  _ = black_scholes_call(S, buy_strike,  T_cs, risk_free / 12, ann_vol)
        spread_credit = sell_price - buy_price

        # Number of spreads based on coverage of LEAPS notional
        leaps_notional = leaps_val / (leaps_delta * 0.25)   # approx shares equivalent
        n_spreads = max(int(leaps_notional * coverage / 100), 0)

        # Assume spread closes at 50% profit each month (simplified)
        spread_monthly_pnl = n_spreads * spread_credit * 0.50
        spread_monthly_pnl -= n_spreads * commission * 2   # 2 legs
        spread_monthly_pnl -= abs(spread_monthly_pnl) * slippage
        cs_pnl += spread_monthly_pnl
        cs_trades += 1
        if spread_monthly_pnl > 0:
            cs_wins += 1

        # ── ENGINE C: TAIL PUTS ──────────────────────────────────────────────
        # Puts grow during market drops, monetise on -20%+ drop
        T_put = 7 / 12              # 7 months average
        put_strike = S * 0.75       # 25% OTM midpoint
        put_price, _ = black_scholes_put(S, put_strike, T_put, risk_free / 12, ann_vol)

        spy_dd_from_ath = (spy_ath - S) / spy_ath
        if spy_dd_from_ath >= 0.20 and puts_val > 0:
            # Monetise puts — they are worth multiples in a crash
            # Simplified: assume 10-15x on premium for a 20%+ move
            multiplier = 12 * spy_dd_from_ath / 0.20   # scales with depth
            puts_pnl   = puts_val * multiplier
            leaps_val += puts_pnl   # recycle into LEAPS
            puts_val   = nav * puts_alloc * 0.20   # rebuild with smaller budget

        # ── DRAWDOWN CASH DEPLOYMENT ─────────────────────────────────────────
        if spy_dd_from_ath >= 0.05 and spy_dd_from_ath < 0.10:
            deploy = min(cash * 0.25, cash)
        elif spy_dd_from_ath >= 0.10 and spy_dd_from_ath < 0.20:
            deploy = min(cash * 0.25, cash)
        elif spy_dd_from_ath >= 0.20 and spy_dd_from_ath < 0.30:
            deploy = min(cash * 0.25, cash)
            call_selling_halted = True
        elif spy_dd_from_ath >= 0.30:
            deploy = cash
            call_selling_halted = True
        else:
            deploy = 0.0
            if spy_dd_from_ath < 0.05:
                call_selling_halted = False

        if deploy > 0:
            leaps_val += deploy * (1 - slippage)
            cash      -= deploy

        # Cash earns T-Bill rate
        cash = cash * (1 + risk_free / 12)

        # ── NAV RECONCILIATION ────────────────────────────────────────────────
        nav = leaps_val + cash + puts_val + cs_pnl * 0.1   # cs_pnl as running credit

        nav_series.append(round(nav, 2))
        leaps_series.append(round(leaps_val, 2))
        cash_series.append(round(cash, 2))
        puts_series.append(round(puts_val, 2))
        cs_pnl_series.append(round(cs_pnl, 2))
        leverage_series.append(round(leaps_val / max(nav, 1), 4))

    # ── METRICS ───────────────────────────────────────────────────────────────
    nav_arr   = np.array(nav_series)
    peak      = np.maximum.accumulate(nav_arr)
    dd_series = (nav_arr - peak) / peak * 100
    max_dd    = float(dd_series.min())
    total_ret = (nav_arr[-1] / nav_arr[0]) - 1
    cagr_ihe  = (nav_arr[-1] / nav_arr[0]) ** (12 / n_steps) - 1
    monthly_rets = np.diff(nav_arr) / nav_arr[:-1]
    sharpe    = (monthly_rets.mean() / (monthly_rets.std() + 1e-9)) * np.sqrt(12)
    win_rate  = cs_wins / max(cs_trades, 1) * 100

    months = list(range(n_steps + 1))
    return {
        "nav":        nav_series,
        "leaps":      leaps_series,
        "cash":       cash_series,
        "puts":       puts_series,
        "cs_pnl":     cs_pnl_series,
        "leverage":   leverage_series,
        "drawdown":   dd_series.tolist(),
        "spy":        [100.0] + list(spy_path),
        "vix":        [18.0]  + list(vix_path),
        "months":     months,
        "cagr":       cagr_ihe * 100,
        "max_dd":     max_dd,
        "sharpe":     sharpe,
        "win_rate":   win_rate,
        "final_nav":  nav_arr[-1],
    }


# ── TAB 5: IRON HARVEST ELITE ────────────────────────────────────────────────
with tab5:
    st.markdown(
        "**Iron Harvest Elite** — systematic options portfolio combining LEAPS core exposure, "
        "call spread yield overlay, tail-hedge puts, and mechanical cash deployment on drawdowns."
    )

    # Strategy overview cards
    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("LEAPS (Engine A)", "70–75%", "Core beta, delta-targeted by VIX")
    ov2.metric("Cash / T-Bills",   "10–15%", "Deployed on SPY drawdowns")
    ov3.metric("Tail Puts (Engine C)", "3–5%", "Monetise on −20%+ drop")
    ov4.metric("Call Spreads (Engine B)", "Overlay", "PMCC yield + alpha")

    st.markdown("---")

    # IHE-specific controls
    ih1, ih2, ih3 = st.columns(3)
    with ih1:
        ihe_capital  = st.number_input("Starting Capital ($)", value=int(initial_inv) if initial_inv > 0 else 100_000,
                                        min_value=10_000, step=10_000, key="ihe_cap")
        ihe_years    = st.slider("Simulation years", 3, 20, min(proj_years, 15), key="ihe_yrs")
    with ih2:
        ihe_ret  = st.slider("SPY annual return (%)", 5.0, 18.0, ann_ret * 100, 0.5, key="ihe_ret") / 100
        ihe_vol  = st.slider("Annual volatility (%)", 8.0, 35.0, ann_vol * 100, 0.5, key="ihe_vol") / 100
    with ih3:
        ihe_rfr      = st.slider("Risk-free rate (%)", 1.0, 6.0, 4.5, 0.25, key="ihe_rfr") / 100
        ihe_slip     = st.slider("Slippage (%)", 0.0, 1.0, 0.2, 0.1, key="ihe_slip") / 100

    with st.spinner("Running Iron Harvest Elite simulation..."):
        ihe = run_iron_harvest_backtest(
            starting_capital = ihe_capital,
            years            = ihe_years,
            ann_ret          = ihe_ret,
            ann_vol          = ihe_vol,
            risk_free        = ihe_rfr,
            slippage         = ihe_slip,
        )

    # KPI strip
    kk = st.columns(5)
    kpi(kk[0], "Final NAV",      fmt_currency(ihe["final_nav"]),  f"from {fmt_currency(ihe_capital)}", "#EF9F27")
    kpi(kk[1], "CAGR",           f"{ihe['cagr']:.1f}%",           "annualised",                        "#00d4a0")
    kpi(kk[2], "Max Drawdown",   f"{ihe['max_dd']:.1f}%",         "peak-to-trough",                    "#f5a623")
    kpi(kk[3], "Sharpe Ratio",   f"{ihe['sharpe']:.2f}",          "monthly, ann.",                     "#00ffcc")
    kpi(kk[4], "CS Win Rate", f"{ihe['win_rate']:.0f}%", "Engine B win rate", "#aaaaaa")

    st.markdown("")

    ihe_tab1, ihe_tab2, ihe_tab3, ihe_tab4 = st.tabs([
        "Equity Curve", "Drawdown", "Portfolio Exposure", "Strategy Blueprint"
    ])

    months_labels = [f"Mo {m}" if m % 12 == 0 else "" for m in ihe["months"]]
    spy_norm = [s / ihe["spy"][0] * ihe_capital for s in ihe["spy"]]

    with ihe_tab1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=ihe["months"], y=ihe["nav"], name="Iron Harvest Elite",
            line=dict(color="#EF9F27", width=2.5),
        ))
        fig_e.add_trace(go.Scatter(
            x=ihe["months"], y=spy_norm, name="SPY (buy & hold)",
            line=dict(color="#1D9E75", width=1.5, dash="dash"),
        ))
        # Shade component areas
        fig_e.add_trace(go.Scatter(
            x=ihe["months"], y=ihe["leaps"], name="LEAPS value",
            line=dict(color="#EF9F27", width=0),
            fill="tozeroy", fillcolor="rgba(239,159,39,0.08)",
        ))
        fig_e.update_layout(
            template="plotly_dark", height=400,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Month", yaxis=dict(tickprefix="$"),
        )
        st.plotly_chart(fig_e, use_container_width=True)

    with ihe_tab2:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=ihe["months"], y=ihe["drawdown"], name="Drawdown",
            line=dict(color="#f5a623", width=1.5),
            fill="tozeroy", fillcolor="rgba(245,166,35,0.12)",
        ))
        # Drawdown deployment thresholds
        for lvl, col in [(-5, "rgba(255,255,255,.1)"), (-10, "rgba(255,200,0,.15)"),
                          (-20, "rgba(255,100,0,.2)"),  (-30, "rgba(255,50,50,.25)")]:
            fig_dd.add_hline(
                y=lvl, line=dict(color=col, width=1, dash="dot"),
                annotation_text=f"{lvl}% deploy" if lvl in [-5, -30] else f"{lvl}%",
                annotation_position="right",
            )
        fig_dd.update_layout(
            template="plotly_dark", height=340,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Month",
            yaxis=dict(ticksuffix="%", title="Drawdown from peak"),
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        st.caption("Dashed lines show cash deployment thresholds: −5%, −10%, −20% (halt call selling), −30% (deploy all remaining cash).")

    with ihe_tab3:
        fig_lev = go.Figure()
        fig_lev.add_trace(go.Scatter(
            x=ihe["months"], y=[l * 100 for l in ihe["leverage"]],
            name="LEAPS as % of NAV",
            line=dict(color="#EF9F27", width=2),
            fill="tozeroy", fillcolor="rgba(239,159,39,0.10)",
        ))
        fig_lev.add_hline(y=60, line=dict(color="rgba(255,255,255,.2)", width=1, dash="dot"),
                           annotation_text="Min target 60%", annotation_position="right")
        fig_lev.add_hline(y=75, line=dict(color="rgba(255,255,255,.3)", width=1, dash="dot"),
                           annotation_text="Target 75%", annotation_position="right")

        # VIX on secondary y
        fig_lev.add_trace(go.Scatter(
            x=ihe["months"], y=ihe["vix"][1:] + [ihe["vix"][-1]],
            name="VIX", yaxis="y2",
            line=dict(color="#378ADD", width=1, dash="dot"),
        ))
        fig_lev.update_layout(
            template="plotly_dark", height=340,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Month",
            yaxis=dict(ticksuffix="%", title="LEAPS / NAV"),
            yaxis2=dict(title="VIX", overlaying="y", side="right",
                        range=[0, 80], tickcolor="#378ADD", titlefont=dict(color="#378ADD")),
        )
        st.plotly_chart(fig_lev, use_container_width=True)
        st.caption("Orange = LEAPS exposure as % of NAV (target 70–75%). Blue dashed = simulated VIX. Higher VIX → higher delta target → LEAPS becomes more defensive.")

    with ihe_tab4:
        st.markdown("### System Architecture")

        bp1, bp2 = st.columns(2)
        with bp1:
            st.markdown("#### Engine A — LEAPS Core")
            st.markdown("""
- **Ticker:** SPY or SPLG
- **DTE:** 12–18 months to expiry
- **Strike:** 20–30% In-The-Money
- **Roll:** When DTE ≤ 250 days (9–10 months remaining)
- **Delta by VIX regime:**
  - VIX < 15 → δ 0.75–0.80 (bull, accept more theta)
  - VIX 15–25 → δ 0.80–0.85 (neutral)
  - VIX > 25 → δ 0.90–0.95 (fear, maximise intrinsic)
""")
            st.markdown("#### Engine C — Tail Puts")
            st.markdown("""
- **Structure:** 20–30% OTM puts, 6–9 months out
- **Budget:** Exactly 5% of portfolio NAV
- **Action on −20%+ SPY drop:** Monetise immediately
- **Recycle:** Roll profits into 90–95δ LEAPS for recovery
""")
        with bp2:
            st.markdown("#### Engine B — Call Spread Overlay (PMCC)")
            st.markdown("""
- **Sell leg:** 0.20–0.30 delta call
- **Buy leg:** $50–$100 higher (width strictly enforced)
- **Take profit:** Close at 40–60% of max profit
- **Coverage by regime:**
  - SPY > 50SMA & VIX < 15 → 50% coverage
  - Flat market (±2% / 30d) → 75%
  - VIX > 25 → 90%
  - Market in −20%+ drawdown → **halt all call selling**
""")
            st.markdown("#### Cash Deployment Protocol")
            dd_df = pd.DataFrame({
                "SPY Drawdown": ["−5%", "−10%", "−20%", "−30%"],
                "Cash Deployed": ["25% of buffer", "25% of buffer", "25% of buffer", "100% remaining"],
                "Action": ["Buy LEAPS", "Buy LEAPS", "Buy LEAPS + halt calls", "Max exposure"],
            })
            st.dataframe(dd_df, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Target Leverage Range")
        st.markdown("""
| Market State | LEAPS Delta | Coverage | Target Leverage |
|---|---|---|---|
| Strong bull (VIX < 15) | 0.75–0.80 | 50% | 0.6–0.8× |
| Neutral (VIX 15–25) | 0.80–0.85 | 75% | 0.8–1.2× |
| Fear (VIX > 25) | 0.90–0.95 | 90% | 1.2–1.6× |
| Crash (SPY −30%) | 0.90–0.95 | Halted | Max exposure |
""")


# =============================================================================
# FOOTER — DISCLAIMER
# =============================================================================
st.markdown("---")
st.caption(
    "Educational simulation only. Not financial advice. "
    "LEAPS options involve significant risk including total loss of premium. "
    "Past returns do not guarantee future results. "
    "Always consult a licensed financial advisor before trading options."
)
