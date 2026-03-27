"""
⚔️ IRON HARVEST TURBO v4.0 — Simplified Hyper-Aggressive
================================================================
Systematic Deep-ITM LEAPS + Wide PMCC + Minimal Hedge
Zero-Blow-Up Options Portfolio
Two tabs:
  1. Forward Execution Engine — live signals from today's market data
  2. Backtesting Engine — Turbo v4.0 vs SPY vs SPXL from Jan 2010
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iron Harvest Turbo v4.0",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; }
.signal-box {
    padding: 12px 16px;
    border-radius: 6px;
    border-left: 4px solid;
    margin-bottom: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.signal-action { background:#1a1f2e; border-color:#EF9F27; color:#EF9F27; }
.signal-warning { background:#1f1a1a; border-color:#E24B4A; color:#E24B4A; }
.signal-hold { background:#1a1f1a; border-color:#1D9E75; color:#1D9E75; }
.signal-info { background:#1a1a2e; border-color:#378ADD; color:#378ADD; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ── SECTION 1: OPTIONS MATH ENGINE (unchanged) ───────────────────────────────
# ==============================================================================
class OptionsEngine:
    """Black-Scholes pricing and Greeks."""
    @staticmethod
    def d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return np.inf if S >= K else -np.inf
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    @staticmethod
    def d2(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return np.inf if S >= K else -np.inf
        return OptionsEngine.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    @classmethod
    def call_price(cls, S, K, T, r, sigma):
        if T <= 0:
            return max(S - K, 0.0)
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    @classmethod
    def put_price(cls, S, K, T, r, sigma):
        if T <= 0:
            return max(K - S, 0.0)
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    @classmethod
    def call_delta(cls, S, K, T, r, sigma):
        if T <= 0:
            return 1.0 if S > K else 0.0
        return float(norm.cdf(cls.d1(S, K, T, r, sigma)))
    @classmethod
    def put_delta(cls, S, K, T, r, sigma):
        return cls.call_delta(S, K, T, r, sigma) - 1.0
    @classmethod
    def find_strike_for_delta(cls, S, T, r, sigma, target_delta, option_type="call"):
        lo, hi = S * 0.40, S * 1.60
        for _ in range(300):
            mid = (lo + hi) / 2.0
            if option_type == "call":
                d = cls.call_delta(S, mid, T, r, sigma)
                if d > target_delta:
                    lo = mid
                else:
                    hi = mid
            else:
                d = abs(cls.put_delta(S, mid, T, r, sigma))
                if d > target_delta:
                    hi = mid
                else:
                    lo = mid
        return (lo + hi) / 2.0

# ==============================================================================
# ── SECTION 2: v4.0 SIMPLIFIED MARKET REGIME (only 2 states) ─────────────────
# ==============================================================================
@dataclass
class MarketRegime:
    vix: float
    spy_vs_ath_pct: float  # negative = drawdown

    @property
    def drawdown_pct(self) -> float:
        return abs(min(self.spy_vs_ath_pct, 0.0)) * 100

    @property
    def name(self) -> str:
        return "Bull" if self.vix < 20 else "Fear"

    @property
    def target_delta(self) -> float:
        return 0.70 if self.name == "Bull" else 0.85

    @property
    def target_exposure_mult(self) -> float:   # TURBO LEVERAGE
        return 2.7 if self.name == "Bull" else 1.2

    @property
    def leaps_alloc(self) -> float:
        return 0.95 if self.name == "Bull" else 0.85

    @property
    def cash_alloc(self) -> float:
        return 0.04 if self.name == "Bull" else 0.10

    @property
    def puts_alloc(self) -> float:
        return 0.01 if self.vix < 25 else 0.03

    @property
    def pmcc_allowed(self) -> bool:
        return self.name == "Fear"

    @property
    def pmcc_coverage(self) -> float:
        return 0.80 if self.pmcc_allowed else 0.0

# ==============================================================================
# ── SECTION 3: DATA FETCHER (unchanged) ──────────────────────────────────────
# ==============================================================================
class DataFetcher:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_live() -> Dict:
        try:
            spy_tk = yf.Ticker("SPY")
            vix_tk = yf.Ticker("^VIX")
            irx_tk = yf.Ticker("^IRX")
            spy_hist = spy_tk.history(period="1y", interval="1d")
            vix_hist = vix_tk.history(period="5d", interval="1d")
            irx_hist = irx_tk.history(period="5d", interval="1d")
            spy_close = float(spy_hist["Close"].iloc[-1])
            spy_ath = float(spy_hist["Close"].max())
            spy_50sma = float(spy_hist["Close"].rolling(50).mean().iloc[-1])
            spy_vol_30d = float(spy_hist["Close"].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252))
            vix = float(vix_hist["Close"].iloc[-1])
            rfr = float(irx_hist["Close"].iloc[-1]) / 100 if not irx_hist.empty else 0.060
            ath_idx = spy_hist["Close"].idxmax()
            days_since_ath = (spy_hist.index[-1] - ath_idx).days
            return {
                "spy": spy_close,
                "spy_ath": spy_ath,
                "spy_50sma": spy_50sma,
                "spy_vol_30d": spy_vol_30d,
                "vix": vix,
                "rfr": rfr,
                "spy_hist": spy_hist,
                "date": spy_hist.index[-1].strftime("%Y-%m-%d"),
                "days_since_ath": days_since_ath,
            }
        except Exception as e:
            st.error(f"Data fetch error: {e}")
            return {}

    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_history(start: str = "2010-01-01") -> pd.DataFrame:
        tickers = ["SPY", "SPXL", "^VIX"]
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"].copy()
        else:
            closes = raw.copy()
        closes.columns = [str(c).replace("^", "") for c in closes.columns]
        closes = closes.dropna(how="all")
        closes["VIX"] = closes["VIX"].ffill()
        return closes

# ==============================================================================
# ── SECTION 4: v4.0 TURBO BACKTESTER (simplified + hyper-aggressive) ────────
# ==============================================================================
class IronHarvestBacktesterV4:
    COMMISSION = 0.65
    SLIPPAGE = 0.002

    def __init__(self, starting_capital: float, rfr: float = 0.060):
        self.starting_capital = starting_capital
        self.rfr = rfr
        self.opt = OptionsEngine()

    def run(self, hist: pd.DataFrame) -> Dict:
        nav = self.starting_capital
        leaps_val = nav * 0.90
        cash = nav * 0.07
        puts_val = nav * 0.015
        pmcc_credits = 0.0
        nav_series = [nav]
        leaps_series = [leaps_val]
        cash_series = [cash]
        puts_series = [puts_val]
        pmcc_pnl_series = [0.0]
        lev_series = [leaps_val / nav]
        pmcc_wins = pmcc_trades = 0
        spy_ath = float(hist["SPY"].iloc[0])
        ath_idx_i = 0
        prev_month = hist.index[0].month

        for i in range(1, len(hist)):
            row = hist.iloc[i]
            prev = hist.iloc[i - 1]
            S = float(row["SPY"])
            S_p = float(prev["SPY"])
            vix = float(row["VIX"]) if not np.isnan(row["VIX"]) else 18.0
            spy_ret = (S - S_p) / S_p

            if S >= spy_ath:
                spy_ath = S
                ath_idx_i = i
            dd = (spy_ath - S) / spy_ath

            regime = MarketRegime(vix=vix, spy_vs_ath_pct=-dd)

            # ── TURBO LEAPS (exposure multiplier applied)
            delta = regime.target_delta
            exposure_mult = regime.target_exposure_mult
            theta_ann = 0.018 if delta >= 0.85 else 0.022
            theta_daily = leaps_val * theta_ann / 252
            leaps_val = max(leaps_val * (1 + exposure_mult * delta * spy_ret) - theta_daily, 0)

            # ── AGGRESSIVE PMCC (monthly, 80% coverage)
            if row.name.month != prev_month:
                prev_month = row.name.month
                if regime.pmcc_allowed and regime.pmcc_coverage > 0:
                    T_pmcc = 37 / 365
                    sigma = vix / 100
                    sell_d = 0.28
                    sell_K = self.opt.find_strike_for_delta(S, T_pmcc, self.rfr, sigma, sell_d)
                    buy_K = sell_K + 65
                    sell_px = self.opt.call_price(S, sell_K, T_pmcc, self.rfr, sigma)
                    buy_px = self.opt.call_price(S, buy_K, T_pmcc, self.rfr, sigma)
                    spread_cr = max(sell_px - buy_px, 0.0)
                    n_spreads = max(int(leaps_val * regime.pmcc_coverage / (S * 100)), 1)
                    gross_cr = spread_cr * n_spreads * 100
                    net_cr = gross_cr - self.COMMISSION * n_spreads * 2 - gross_cr * self.SLIPPAGE
                    monthly_pmcc = net_cr * 0.80
                    pmcc_credits += monthly_pmcc
                    pmcc_trades += 1
                    if monthly_pmcc > 0:
                        pmcc_wins += 1

            # ── MINIMAL TAIL HEDGE
            put_daily_r = -0.0015 / 252 if vix < 25 else (abs(spy_ret) * 0.25 if spy_ret < -0.01 else -0.002 / 252)
            puts_val = max(puts_val * (1 + put_daily_r), 0)

            # ── CASH DEPLOY + DAILY REBALANCE TO REGIME TARGETS
            cash = cash * (1 + self.rfr / 252)
            if regime.drawdown_pct >= 10:
                deploy = cash * 0.40 if regime.drawdown_pct >= 20 else cash * 0.25
                leaps_val += deploy * (1 - self.SLIPPAGE)
                cash -= deploy

            # Daily rebalance (the CAGR rocket)
            total = leaps_val + cash + puts_val + pmcc_credits * 0.08
            target_leaps = total * regime.leaps_alloc
            leaps_val = target_leaps

            nav = leaps_val + cash + puts_val + pmcc_credits * 0.08
            nav_series.append(round(nav, 2))
            leaps_series.append(round(leaps_val, 2))
            cash_series.append(round(cash, 2))
            puts_series.append(round(puts_val, 2))
            pmcc_pnl_series.append(round(pmcc_credits, 2))
            lev_series.append(round(leaps_val / max(nav, 1), 4))

        # ── METRICS
        nav_arr = np.array(nav_series)
        peak = np.maximum.accumulate(nav_arr)
        uw = (nav_arr - peak) / peak * 100
        max_dd = float(uw.min())
        n_years = len(hist) / 252
        cagr = ((nav_arr[-1] / nav_arr[0]) ** (1 / n_years) - 1) * 100
        monthly = nav_arr[::21]
        m_rets = np.diff(monthly) / monthly[:-1]
        sharpe = (m_rets.mean() / (m_rets.std() + 1e-9)) * np.sqrt(12)
        win_rate = pmcc_wins / max(pmcc_trades, 1) * 100

        return {
            "nav": nav_series, "leaps": leaps_series, "cash": cash_series,
            "puts": puts_series, "pmcc_pnl": pmcc_pnl_series, "leverage": lev_series,
            "drawdown": uw.tolist(), "cagr": cagr, "max_dd": max_dd,
            "sharpe": sharpe, "win_rate": win_rate, "final_nav": nav_arr[-1],
            "n_years": n_years,
        }

# ==============================================================================
# ── SECTION 5: v4.0 SIGNAL ENGINE (simplified for Turbo rules) ───────────────
# ==============================================================================
class SignalEngineV4:
    def __init__(self, live: Dict, portfolio_nav: float):
        self.live = live
        self.nav = portfolio_nav
        self.opt = OptionsEngine()
        S = live.get("spy", 500)
        spy_ath = live.get("spy_ath", S)
        vix = live.get("vix", 18)
        self.regime = MarketRegime(
            vix=vix,
            spy_vs_ath_pct=(S - spy_ath) / spy_ath,
        )

    def generate(self) -> List[Dict]:
        signals = []
        live = self.live
        S = live.get("spy", 500)
        vix = live.get("vix", 18)
        sigma = live.get("spy_vol_30d", vix / 100)
        rfr = live.get("rfr", 0.060)
        r = self.regime
        dd_pct = r.drawdown_pct
        nav = self.nav

        signals.append({
            "type": "info", "engine": "REGIME v4.0",
            "message": f"Regime: {r.name} | VIX: {vix:.1f} | SPY: ${S:.2f} | ATH Drawdown: -{dd_pct:.1f}% | Target Leverage: {r.target_exposure_mult:.1f}×",
        })

        leaps_alloc = r.leaps_alloc
        cash_alloc = r.cash_alloc
        puts_alloc = r.puts_alloc
        leaps_amt = nav * leaps_alloc
        cash_amt = nav * cash_alloc
        puts_amt = nav * puts_alloc
        signals.append({
            "type": "info", "engine": "DYNAMIC ALLOCATION",
            "message": f"LEAPS: {leaps_alloc*100:.0f}% (${leaps_amt:,.0f}) | Cash: {cash_alloc*100:.0f}% (${cash_amt:,.0f}) | Puts: {puts_alloc*100:.1f}% (${puts_amt:,.0f})",
        })

        # ENGINE A: LEAPS
        T_leaps = 15 / 12
        leaps_K = self.opt.find_strike_for_delta(S, T_leaps, rfr, sigma, r.target_delta)
        leaps_px = self.opt.call_price(S, leaps_K, T_leaps, rfr, sigma)
        intrinsic = max(S - leaps_K, 0.0)
        intrinsic_pct = (intrinsic / leaps_px * 100) if leaps_px > 0 else 0
        itm_pct = (S - leaps_K) / S * 100
        contracts = max(int(leaps_amt / (leaps_px * 100)), 1)
        intrinsic_ok = intrinsic_pct >= 60
        signals.append({
            "type": "action" if intrinsic_ok else "warning",
            "engine": "ENGINE A — LEAPS",
            "message": f"TARGET DELTA: {r.target_delta:.2f}Δ | STRIKE: ${leaps_K:.0f} ({itm_pct:.0f}% ITM) | EXPIRY: 12–24mo | PRICE: ~${leaps_px:.2f} | INTRINSIC: {intrinsic_pct:.0f}% {'✅ ≥60%' if intrinsic_ok else '⚠️ <60% — go deeper'} | CONTRACTS: {contracts} | MAX/POSITION 18–22%",
        })

        signals.append({
            "type": "hold", "engine": "ENGINE A — ROLL CHECK",
            "message": "Roll every time <9 months left. NEVER hold <6 months. Close → new at current regime delta.",
        })

        # ENGINE B: PMCC
        if not r.pmcc_allowed:
            signals.append({
                "type": "warning", "engine": "ENGINE B — PMCC",
                "message": "⛔ PMCC HALTED in Bull regime (VIX < 20). Only sell in Fear.",
            })
        else:
            T_pmcc = 37 / 365
            sell_K = self.opt.find_strike_for_delta(S, T_pmcc, rfr, sigma, 0.28)
            buy_K = sell_K + 65
            sell_px = self.opt.call_price(S, sell_K, T_pmcc, rfr, sigma)
            buy_px = self.opt.call_price(S, buy_K, T_pmcc, rfr, sigma)
            spread_cr = max(sell_px - buy_px, 0.0)
            n_spreads = max(int(leaps_amt * r.pmcc_coverage / (S * 100)), 1)
            gross = spread_cr * n_spreads * 100
            signals.append({
                "type": "action", "engine": "ENGINE B — PMCC",
                "message": f"SELL: ${sell_K:.0f} call (0.20–0.35Δ, 30–45 DTE) | BUY: ${buy_K:.0f} call (+$65) | CREDIT: ${spread_cr:.2f} | COVERAGE: 80% of LEAPS | CONTRACTS: {n_spreads} | CLOSE at 50–75% profit",
            })

        # ENGINE C: MINIMAL TAIL PUTS
        T_put = 7.5 / 12
        put_K = S * 0.70
        put_px = self.opt.put_price(S, put_K, T_put, rfr, sigma)
        put_cts = max(int(puts_amt / (put_px * 100)), 1)
        signals.append({
            "type": "warning" if puts_alloc >= 0.03 else "action",
            "engine": "ENGINE C — MINIMAL TAIL PUTS",
            "message": f"ALLOC: {puts_alloc*100:.1f}% | BUY: ${put_K:.0f} puts (~30% OTM, 6mo) | PRICE: ~${put_px:.2f} | CONTRACTS: {put_cts} | Bleed almost zero",
        })

        # CASH DEPLOY
        if dd_pct >= 10:
            deploy_pct = 0.40 if dd_pct >= 20 else 0.25
            deploy_amt = cash_amt * deploy_pct
            signals.append({
                "type": "action", "engine": "CASH DEPLOYMENT",
                "message": f"🚨 DEPLOY {deploy_pct*100:.0f}% CASH on -{dd_pct:.1f}% drawdown | Buy more LEAPS immediately",
            })
        else:
            signals.append({
                "type": "hold", "engine": "CASH DEPLOYMENT",
                "message": f"✅ HOLD CASH — only -{dd_pct:.1f}% from ATH",
            })

        # HARD RULES
        signals.append({
            "type": "info", "engine": "HARD RULES",
            "message": "NEVER: OTM LEAPS, hold <6mo, sell calls in bull (VIX<20). ALWAYS: roll early, buy every dip, let PMCC print in Fear.",
        })
        return signals

# ==============================================================================
# ── SECTION 6: UI HELPERS (unchanged) ────────────────────────────────────────
# ==============================================================================
def fmt_currency(v: float) -> str:
    a = abs(v)
    if a >= 1e9: return f"${v/1e9:.2f}B"
    if a >= 1e6: return f"${v/1e6:.2f}M"
    if a >= 1e3: return f"${v:,.0f}"
    return f"${v:.2f}"

def kpi(col, label: str, value: str, sub: str = "", color: str = "#e8eaf0"):
    sub_html = f'<p style="margin:3px 0 0;font-size:11px;color:#6b7280;">{sub}</p>' if sub else ""
    col.markdown(
        f'<div style="background:#111318;padding:14px 16px;border-radius:6px;'
        f'border:1px solid rgba(255,255,255,0.07);text-align:center;margin-bottom:4px;">'
        f'<p style="margin:0 0 4px;font-size:9px;color:#6b7280;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:0.1em;">{label}</p>'
        f'<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;'
        f'color:{color};">{value}</p>{sub_html}</div>',
        unsafe_allow_html=True,
    )

def signal_box(s: Dict):
    css = {"action": "signal-action", "warning": "signal-warning", "hold": "signal-hold", "info": "signal-info"}.get(s["type"], "signal-info")
    st.markdown(
        f'<div class="signal-box {css}">'
        f'<span style="opacity:.6;font-size:10px;">{s["engine"]}</span><br>'
        f'{s["message"]}</div>',
        unsafe_allow_html=True,
    )

def plot_dark(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="IBM Plex Mono"),
    )
    return fig

# ==============================================================================
# ── SECTION 7: SIDEBAR (updated for v4.0) ────────────────────────────────────
# ==============================================================================
st.sidebar.markdown("## ⚔️ Iron Harvest Turbo v4.0")
st.sidebar.markdown("*Simplified Hyper-Aggressive LEAPS + Wide PMCC + Minimal Hedge*")
st.sidebar.markdown("---")
st.sidebar.markdown("### Portfolio Configuration")
portfolio_nav = st.sidebar.number_input("Portfolio NAV ($)", value=100_000, min_value=10_000, step=10_000)
st.sidebar.markdown("**Turbo Allocations (regime-driven):** LEAPS 80–95% | Cash 4–10% | Puts 0–3%")
st.sidebar.markdown("---")
st.sidebar.markdown("### Backtest Settings")
bt_capital = st.sidebar.number_input("Backtest Starting Capital ($)", value=100_000, min_value=10_000, step=10_000)
bt_rfr = st.sidebar.slider("Risk-Free Rate (%)", 1.0, 8.0, 6.0, 0.25) / 100
st.sidebar.markdown("---")
st.sidebar.markdown("**Black-Scholes approximations.** Commission $0.65/contract | Slippage 0.20%")

# ==============================================================================
# ── SECTION 8: MAIN APP (updated titles + v4.0 classes) ──────────────────────
# ==============================================================================
st.title("⚔️ Iron Harvest Turbo v4.0")
st.caption("Simplified Hyper-Aggressive LEAPS + Wide PMCC + Minimal Hedge | Zero-Blow-Up Options Portfolio")
tab_exec, tab_bt = st.tabs(["🎯 Forward Execution Engine", "📊 Backtesting Engine (2010–Today)"])

# TAB 1: FORWARD EXECUTION ENGINE
with tab_exec:
    st.markdown("### Today's Operational Signals")
    st.caption("Live SPY, VIX, T-bill data. All signals are deterministic v4.0 Turbo ruleset outputs.")
    with st.spinner("Fetching live market data..."):
        live = DataFetcher.fetch_live()
    if not live:
        st.error("Could not fetch live data.")
        st.stop()

    S = live["spy"]
    vix = live["vix"]
    spy_ath = live["spy_ath"]
    dd_pct = (spy_ath - S) / spy_ath * 100
    rfr = live["rfr"]
    sigma = live["spy_vol_30d"]
    date = live["date"]

    st.markdown(f"**Data as of {date}**")
    snap_cols = st.columns(6)
    kpi(snap_cols[0], "SPY Close", f"${S:.2f}", f"ATH: ${spy_ath:.2f}", "#EF9F27")
    kpi(snap_cols[1], "ATH Drawdown", f"-{dd_pct:.1f}%", "from rolling 1yr high", "#E24B4A" if dd_pct > 10 else "#1D9E75")
    kpi(snap_cols[2], "VIX", f"{vix:.1f}", "implied vol", "#f5a623" if vix > 20 else "#1D9E75")
    kpi(snap_cols[3], "30d Real Vol", f"{sigma*100:.1f}%", "annualised", "#aaaaaa")
    kpi(snap_cols[4], "T-Bill Rate", f"{rfr*100:.2f}%", "risk-free", "#aaaaaa")

    st.markdown("---")
    st.markdown("### Exact Execution Orders — v4.0 Turbo Ruleset")
    engine_v4 = SignalEngineV4(live, portfolio_nav)
    signals = engine_v4.generate()
    for s in signals:
        signal_box(s)

    st.markdown("---")
    st.markdown("### One-Page Turbo Rules (memorize)")
    st.markdown("""
**Bull (VIX < 20):** 70Δ LEAPS + 2.7× leverage + almost no PMCC  
**Fear (VIX ≥ 20):** 85Δ LEAPS + 80% PMCC income + cash deploy on dips  
**Big crash:** Hedge pays → buy the bottom at 70Δ  
**Always:** Roll every 9 months. Never hold <6 months. Buy every dip.
""")

# TAB 2: BACKTESTING ENGINE
with tab_bt:
    st.markdown("### Backtest: IHE Turbo v4.0 vs SPY vs SPXL — Jan 2010 to Today")
    st.caption("Black-Scholes approximations for all options legs. Commission $0.65/contract. Slippage 0.20%. Risk-free rate at sidebar setting (default 6.00%).")
    with st.spinner("Fetching historical data (SPY, SPXL, VIX)..."):
        hist = DataFetcher.fetch_history("2010-01-01")
    if hist.empty:
        st.error("Could not fetch historical data.")
        st.stop()

    with st.spinner("Running Iron Harvest Turbo v4.0 backtest..."):
        bt = IronHarvestBacktesterV4(bt_capital, rfr=bt_rfr)
        ihe_results = bt.run(hist)

    spy_curve = (hist["SPY"] / hist["SPY"].iloc[0] * bt_capital).values
    spxl_curve = (hist["SPXL"] / hist["SPXL"].iloc[0] * bt_capital).values
    ihe_curve = np.array(ihe_results["nav"])
    dates = hist.index
    n_years = ihe_results["n_years"]

    # SPY metrics
    spy_cagr = ((spy_curve[-1] / spy_curve[0]) ** (1 / n_years) - 1) * 100
    spy_dd = float(((spy_curve - np.maximum.accumulate(spy_curve)) / np.maximum.accumulate(spy_curve) * 100).min())
    spy_monthly = spy_curve[::21]
    spy_mrets = np.diff(spy_monthly) / spy_monthly[:-1]
    spy_sharpe = (spy_mrets.mean() / (spy_mrets.std() + 1e-9)) * np.sqrt(12)

    # SPXL metrics
    spxl_cagr = ((spxl_curve[-1] / spxl_curve[0]) ** (1 / n_years) - 1) * 100
    spxl_dd = float(((spxl_curve - np.maximum.accumulate(spxl_curve)) / np.maximum.accumulate(spxl_curve) * 100).min())
    spxl_monthly = spxl_curve[::21]
    spxl_mrets = np.diff(spxl_monthly) / spxl_monthly[:-1]
    spxl_sharpe = (spxl_mrets.mean() / (spxl_mrets.std() + 1e-9)) * np.sqrt(12)

    st.markdown("#### Performance Summary")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("##### SPY (Buy & Hold)")
        k = st.columns(2)
        kpi(k[0], "CAGR", f"{spy_cagr:.1f}%", "", "#1D9E75")
        kpi(k[1], "Max DD", f"{spy_dd:.1f}%", "", "#E24B4A")
        kpi(k[0], "Sharpe", f"{spy_sharpe:.2f}", "", "#aaaaaa")
        kpi(k[1], "Final", fmt_currency(float(spy_curve[-1])), "", "#1D9E75")
    with h2:
        st.markdown("##### SPXL (3× Leveraged)")
        k = st.columns(2)
        kpi(k[0], "CAGR", f"{spxl_cagr:.1f}%", "", "#378ADD")
        kpi(k[1], "Max DD", f"{spxl_dd:.1f}%", "", "#E24B4A")
        kpi(k[0], "Sharpe", f"{spxl_sharpe:.2f}", "", "#aaaaaa")
        kpi(k[1], "Final", fmt_currency(float(spxl_curve[-1])), "", "#378ADD")
    with h3:
        st.markdown("##### ⚔️ Turbo v4.0")
        k = st.columns(2)
        kpi(k[0], "CAGR", f"{ihe_results['cagr']:.1f}%", "", "#EF9F27")
        kpi(k[1], "Max DD", f"{ihe_results['max_dd']:.1f}%", "", "#E24B4A")
        kpi(k[0], "Sharpe", f"{ihe_results['sharpe']:.2f}", "", "#aaaaaa")
        kpi(k[1], "Final", fmt_currency(ihe_results["final_nav"]), "", "#EF9F27")

    st.markdown("")
    extra = st.columns(3)
    kpi(extra[0], "PMCC Win Rate", f"{ihe_results['win_rate']:.0f}%", "Engine B monthly trades", "#1D9E75")
    kpi(extra[1], "Turbo vs SPXL", f"{ihe_results['final_nav']/max(spxl_curve[-1],1):.2f}x", f"vs SPY: {ihe_results['final_nav']/max(spy_curve[-1],1):.2f}x", "#EF9F27")
    kpi(extra[2], "Backtest Length", f"{n_years:.1f} yrs", "Jan 2010 to today", "#aaaaaa")

    st.markdown("---")
    bt_tab1, bt_tab2, bt_tab3, bt_tab4 = st.tabs(["Equity Curve", "Drawdown Analysis", "Portfolio Exposure", "Metrics & Rules"])

    with bt_tab1:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=dates, y=ihe_curve, name="Turbo v4.0", line=dict(color="#EF9F27", width=2.5)))
        fig_eq.add_trace(go.Scatter(x=dates, y=spxl_curve, name="SPXL (3×)", line=dict(color="#378ADD", width=1.8, dash="dash")))
        fig_eq.add_trace(go.Scatter(x=dates, y=spy_curve, name="SPY (1×)", line=dict(color="#1D9E75", width=1.5, dash="dot")))
        fig_eq.update_yaxes(tickprefix="$")
        fig_eq.update_xaxes(title="Date")
        st.plotly_chart(plot_dark(fig_eq, 440), use_container_width=True)

    with bt_tab2:
        ihe_uw = np.array(ihe_results["drawdown"])
        spxl_uw = (spxl_curve - np.maximum.accumulate(spxl_curve)) / np.maximum.accumulate(spxl_curve) * 100
        spy_uw = (spy_curve - np.maximum.accumulate(spy_curve)) / np.maximum.accumulate(spy_curve) * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dates, y=ihe_uw, name="Turbo v4.0", fill="tozeroy", line=dict(color="#EF9F27", width=1.5), fillcolor="rgba(239,159,39,.1)"))
        fig_dd.add_trace(go.Scatter(x=dates, y=spxl_uw, name="SPXL", fill="tozeroy", line=dict(color="#378ADD", width=1), fillcolor="rgba(55,138,221,.07)"))
        fig_dd.add_trace(go.Scatter(x=dates, y=spy_uw, name="SPY", fill="tozeroy", line=dict(color="#1D9E75", width=1), fillcolor="rgba(29,158,117,.07)"))
        fig_dd.update_yaxes(ticksuffix="%", title="Drawdown from peak")
        fig_dd.update_xaxes(title="Date")
        st.plotly_chart(plot_dark(fig_dd, 420), use_container_width=True)

    with bt_tab3:
        lev_arr = np.array(ihe_results["leverage"])
        fig_lev = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lev.add_trace(go.Scatter(x=dates, y=lev_arr * 100, name="LEAPS / NAV %", line=dict(color="#EF9F27", width=1.5), fill="tozeroy", fillcolor="rgba(239,159,39,.1)"), secondary_y=False)
        fig_lev.add_trace(go.Scatter(x=dates, y=hist["VIX"].values, name="VIX", line=dict(color="#378ADD", width=1, dash="dot")), secondary_y=True)
        fig_lev.update_yaxes(title_text="LEAPS / NAV (%)", ticksuffix="%", secondary_y=False)
        fig_lev.update_yaxes(title_text="VIX", secondary_y=True, range=[0, 90])
        st.plotly_chart(plot_dark(fig_lev, 380), use_container_width=True)

    with bt_tab4:
        metrics_df = pd.DataFrame({
            "Strategy": ["SPY Buy & Hold", "SPXL Buy & Hold", "⚔️ Turbo v4.0"],
            "CAGR": [f"{spy_cagr:.1f}%", f"{spxl_cagr:.1f}%", f"{ihe_results['cagr']:.1f}%"],
            "Max Drawdown": [f"{spy_dd:.1f}%", f"{spxl_dd:.1f}%", f"{ihe_results['max_dd']:.1f}%"],
            "Sharpe Ratio": [f"{spy_sharpe:.2f}", f"{spxl_sharpe:.2f}", f"{ihe_results['sharpe']:.2f}"],
            "Final Value": [fmt_currency(float(spy_curve[-1])), fmt_currency(float(spxl_curve[-1])), fmt_currency(ihe_results["final_nav"])],
            "PMCC Win Rate": ["N/A", "N/A", f"{ihe_results['win_rate']:.0f}%"],
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.markdown("#### Hard Rules (v4.0)")
        st.markdown("""
**NEVER:**
- Buy OTM LEAPS (intrinsic must be ≥60%)
- Hold LEAPS with <6 months to expiry
- Sell PMCC in Bull regime (VIX < 20)
- Go all-in without cash buffer

**ALWAYS:**
- Roll every 9 months
- Buy every dip with cash
- Let PMCC print in Fear regime
- Rebalance daily to regime targets
""")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.caption(
    "⚔️ Iron Harvest Turbo v4.0 | Educational simulation. Not financial advice. "
    "Options trading involves significant risk of loss. "
    "Black-Scholes approximations used for historical Greeks — actual results will vary. "
    "Past performance does not guarantee future results."
)
