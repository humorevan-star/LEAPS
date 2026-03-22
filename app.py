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

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Growth",
    "📊 Annual Returns",
    "📉 Crash Drawdowns",
    "⚡ Capital Efficiency",
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
# FOOTER — DISCLAIMER
# =============================================================================
st.markdown("---")
st.caption(
    "Educational simulation only. Not financial advice. "
    "LEAPS options involve significant risk including total loss of premium. "
    "Past returns do not guarantee future results. "
    "Always consult a licensed financial advisor before trading options."
)
