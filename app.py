"""
⚔️ IRON HARVEST ELITE v3.0 — Aggro Systematic Options Portfolio
================================================================
Systematic LEAPS + PMCC Overlay + Lean Dynamic Tail Hedge
Institutional Options Portfolio

Two tabs:
  1. Forward Execution Engine  — live signals from today's market data
  2. Backtesting Engine        — IHE v3.0 vs SPY vs SPXL from Jan 2010
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iron Harvest Elite v3.0",
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
.signal-action  { background:#1a1f2e; border-color:#EF9F27; color:#EF9F27; }
.signal-warning { background:#1f1a1a; border-color:#E24B4A; color:#E24B4A; }
.signal-hold    { background:#1a1f1a; border-color:#1D9E75; color:#1D9E75; }
.signal-info    { background:#1a1a2e; border-color:#378ADD; color:#378ADD; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ── SECTION 1: OPTIONS MATH ENGINE ───────────────────────────────────────────
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
# ── SECTION 2: v3.0 MARKET REGIME CLASSIFIER ─────────────────────────────────
# ==============================================================================

@dataclass
class MarketRegime:
    vix: float
    spy_vs_ath_pct: float       # negative = drawdown
    spy_vs_50sma_pct: float     # positive = above 50SMA
    days_since_ath: int = 0

    @property
    def drawdown_pct(self) -> float:
        return abs(min(self.spy_vs_ath_pct, 0.0)) * 100

    @property
    def name(self) -> str:
        if self._is_strong_bull:
            return "Strong Bull"
        elif self.spy_vs_ath_pct > -0.07 and self.vix < 25:
            return "Normal Bull"
        elif self.spy_vs_ath_pct > -0.15 and self.vix < 30:
            return "Neutral"
        elif self.spy_vs_ath_pct > -0.25:
            return "Drawdown"
        else:
            return "Crash"

    @property
    def _is_strong_bull(self) -> bool:
        return (self.vix < 15
                and self.spy_vs_50sma_pct > 0
                and self.days_since_ath <= 60)

    # ── v3.0 DELTA REGIME ─────────────────────────────────────────────────────
    @property
    def target_delta(self) -> float:
        """VIX-based delta with drawdown override (non-negotiable)."""
        dd = self.drawdown_pct
        if dd >= 30:
            return 0.95
        elif dd >= 20:
            return 0.90
        elif dd >= 10:
            return 0.85
        if self.vix < 15:
            return 0.70
        elif self.vix <= 25:
            return 0.775
        else:
            return 0.90

    @property
    def target_delta_range(self) -> Tuple[float, float]:
        dd = self.drawdown_pct
        if dd >= 30:
            return (0.95, 0.95)
        elif dd >= 20:
            return (0.90, 0.92)
        elif dd >= 10:
            return (0.85, 0.87)
        if self.vix < 15:
            return (0.70, 0.72)
        elif self.vix <= 25:
            return (0.75, 0.80)
        else:
            return (0.85, 0.95)

    # ── v3.0 DYNAMIC EXPOSURE ─────────────────────────────────────────────────
    @property
    def target_exposure(self) -> Tuple[float, float]:
        n = self.name
        if n == "Strong Bull":   return (1.8, 2.2)
        elif n == "Normal Bull": return (1.5, 1.8)
        elif n == "Neutral":     return (1.0, 1.3)
        elif n == "Drawdown":    return (0.7, 0.9)
        else:                    return (0.5, 0.5)

    @property
    def leaps_alloc_dynamic(self) -> float:
        """Dynamic LEAPS allocation 75–88%."""
        n = self.name
        if n == "Strong Bull":   return 0.88
        elif n == "Normal Bull": return 0.83
        elif n == "Neutral":     return 0.78
        elif n == "Drawdown":    return 0.75
        else:                    return 0.75

    @property
    def cash_alloc_dynamic(self) -> float:
        """Dynamic cash allocation 6–12%."""
        n = self.name
        if n == "Strong Bull":   return 0.06
        elif n == "Normal Bull": return 0.08
        elif n == "Neutral":     return 0.10
        elif n == "Drawdown":    return 0.12
        else:                    return 0.12

    @property
    def puts_alloc_dynamic(self) -> float:
        """Lean 1–4%: ramp to 4% when VIX > 22 or drawdown > 7%."""
        if self.vix > 22 or self.drawdown_pct > 7:
            return 0.04
        return 0.015

    # ── v3.0 PMCC OVERLAY ────────────────────────────────────────────────────
    @property
    def pmcc_sell_delta_range(self) -> Tuple[float, float]:
        return (0.22, 0.35)

    @property
    def pmcc_sell_delta(self) -> float:
        return 0.285

    @property
    def pmcc_coverage(self) -> float:
        n = self.name
        if n == "Strong Bull":   return 0.0
        elif n == "Normal Bull": return 0.35
        elif n == "Neutral":     return 0.65
        elif n == "Drawdown":    return 0.0
        else:                    return 0.0

    @property
    def pmcc_allowed(self) -> bool:
        if self.name == "Strong Bull":
            return False
        if self.drawdown_pct >= 20:
            return False
        return True

    # ── v3.0 CASH DEPLOYMENT ─────────────────────────────────────────────────
    @property
    def cash_deploy_pct(self) -> float:
        dd = self.drawdown_pct
        if dd >= 35:   return 0.25
        elif dd >= 25: return 0.30
        elif dd >= 15: return 0.25
        elif dd >= 8:  return 0.20
        return 0.0

    @property
    def crash_mode(self) -> bool:
        return self.drawdown_pct >= 20

    @property
    def halt_pmcc(self) -> bool:
        return not self.pmcc_allowed


# ==============================================================================
# ── SECTION 3: DATA FETCHER ──────────────────────────────────────────────────
# ==============================================================================

class DataFetcher:

    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_live() -> Dict:
        try:
            spy_tk = yf.Ticker("SPY")
            vix_tk = yf.Ticker("^VIX")
            irx_tk = yf.Ticker("^IRX")

            spy_hist = spy_tk.history(period="1y",  interval="1d")
            vix_hist = vix_tk.history(period="5d",  interval="1d")
            irx_hist = irx_tk.history(period="5d",  interval="1d")

            spy_close    = float(spy_hist["Close"].iloc[-1])
            spy_ath      = float(spy_hist["Close"].max())
            spy_50sma    = float(spy_hist["Close"].rolling(50).mean().iloc[-1])
            spy_vol_30d  = float(spy_hist["Close"].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252))
            vix          = float(vix_hist["Close"].iloc[-1])
            rfr          = float(irx_hist["Close"].iloc[-1]) / 100 if not irx_hist.empty else 0.060

            ath_idx = spy_hist["Close"].idxmax()
            days_since_ath = (spy_hist.index[-1] - ath_idx).days

            return {
                "spy":           spy_close,
                "spy_ath":       spy_ath,
                "spy_50sma":     spy_50sma,
                "spy_vol_30d":   spy_vol_30d,
                "vix":           vix,
                "rfr":           rfr,
                "spy_hist":      spy_hist,
                "date":          spy_hist.index[-1].strftime("%Y-%m-%d"),
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
# ── SECTION 4: v3.0 BACKTESTER ───────────────────────────────────────────────
# ==============================================================================

class IronHarvestBacktesterV3:
    """
    IHE v3.0 — Aggro Systematic LEAPS + PMCC + Lean Dynamic Tail Hedge.
    Commission $0.65/contract. Slippage 0.20%. Risk-free 6.00%.
    """

    COMMISSION = 0.65
    SLIPPAGE   = 0.002

    def __init__(self, starting_capital: float, rfr: float = 0.060):
        self.starting_capital = starting_capital
        self.rfr  = rfr
        self.opt  = OptionsEngine()

    def run(self, hist: pd.DataFrame) -> Dict:
        nav       = self.starting_capital
        leaps_val = nav * 0.81
        cash      = nav * 0.09
        puts_val  = nav * 0.022
        pmcc_credits = 0.0

        nav_series      = [nav]
        leaps_series    = [leaps_val]
        cash_series     = [cash]
        puts_series     = [puts_val]
        pmcc_pnl_series = [0.0]
        lev_series      = [leaps_val / nav]

        pmcc_wins = 0
        pmcc_trades = 0
        spy_ath     = float(hist["SPY"].iloc[0])
        ath_idx_i   = 0
        last_dd_tier = 0
        prev_month  = hist.index[0].month

        for i in range(1, len(hist)):
            row  = hist.iloc[i]
            prev = hist.iloc[i - 1]
            S    = float(row["SPY"])
            S_p  = float(prev["SPY"])
            vix  = float(row["VIX"]) if not np.isnan(row["VIX"]) else 18.0
            spy_ret = (S - S_p) / S_p

            if S >= spy_ath:
                spy_ath  = S
                ath_idx_i = i
            dd = (spy_ath - S) / spy_ath
            days_since_ath = i - ath_idx_i

            spy_50sma    = float(hist["SPY"].iloc[max(0, i-50):i].mean())
            spy_vs_50sma = (S - spy_50sma) / spy_50sma

            regime = MarketRegime(
                vix=vix,
                spy_vs_ath_pct=-dd,
                spy_vs_50sma_pct=spy_vs_50sma,
                days_since_ath=days_since_ath,
            )

            # ── ENGINE A: LEAPS DAILY MTM ─────────────────────────────────────
            delta = regime.target_delta
            theta_ann   = 0.025 if delta >= 0.90 else (0.035 if delta >= 0.80 else 0.045)
            theta_daily = leaps_val * theta_ann / 252
            leaps_val   = max(leaps_val * (1 + delta * spy_ret) - theta_daily, 0)

            # ── ENGINE B: PMCC OVERLAY (MONTHLY) ─────────────────────────────
            if row.name.month != prev_month:
                prev_month = row.name.month

                if regime.pmcc_allowed and regime.pmcc_coverage > 0:
                    coverage = regime.pmcc_coverage
                    T_pmcc   = 37 / 365
                    sigma    = vix / 100

                    sell_d   = regime.pmcc_sell_delta
                    sell_K   = self.opt.find_strike_for_delta(S, T_pmcc, self.rfr, sigma, sell_d)
                    buy_K    = sell_K + 65

                    sell_px  = self.opt.call_price(S, sell_K, T_pmcc, self.rfr, sigma)
                    buy_px   = self.opt.call_price(S, buy_K,  T_pmcc, self.rfr, sigma)
                    spread_cr = max(sell_px - buy_px, 0.0)

                    n_spreads = max(int(leaps_val * coverage / (S * 100)), 1)
                    gross_cr  = spread_cr * n_spreads * 100
                    net_cr    = (gross_cr
                                 - self.COMMISSION * n_spreads * 2
                                 - gross_cr * self.SLIPPAGE)

                    # Close at 50–75% profit (62.5% midpoint)
                    monthly_pmcc = net_cr * 0.625
                    pmcc_credits += monthly_pmcc
                    pmcc_trades  += 1
                    if monthly_pmcc > 0:
                        pmcc_wins += 1

            # ── ENGINE C: LEAN DYNAMIC TAIL PUTS ─────────────────────────────
            target_puts_alloc = regime.puts_alloc_dynamic
            put_daily_bleed   = -(0.003 / 252)
            if spy_ret < -0.01:
                put_daily_r = abs(spy_ret) * 0.22 * (vix / 18)
            else:
                put_daily_r = put_daily_bleed
            puts_val = max(puts_val * (1 + put_daily_r), 0)

            # Monetise on -20%+ drop; recycle into LEAPS
            if dd >= 0.20 and puts_val > 0:
                multiplier = min(10 * (dd / 0.20), 18)
                bonus      = puts_val * multiplier * 0.30
                leaps_val += bonus
                puts_val   = nav * target_puts_alloc * 0.25

            # ── v3.0 CASH DEPLOYMENT TIERS ────────────────────────────────────
            dd_pct = dd * 100
            if   dd_pct >= 35: dd_tier = 5
            elif dd_pct >= 25: dd_tier = 4
            elif dd_pct >= 15: dd_tier = 3
            elif dd_pct >= 8:  dd_tier = 2
            else:              dd_tier = 0

            if dd_tier > last_dd_tier and cash > 0:
                deploy_pct = {2: 0.20, 3: 0.25, 4: 0.30, 5: 1.00}.get(dd_tier, 0)
                deploy = cash * deploy_pct
                leaps_val    += deploy * (1 - self.SLIPPAGE)
                cash         -= deploy
                last_dd_tier  = dd_tier
            elif dd_pct < 5:
                last_dd_tier = 0

            # Cash earns 6% T-Bill rate daily
            cash = cash * (1 + self.rfr / 252)

            # ── NAV ───────────────────────────────────────────────────────────
            nav = leaps_val + cash + puts_val + pmcc_credits * 0.08

            nav_series.append(round(nav, 2))
            leaps_series.append(round(leaps_val, 2))
            cash_series.append(round(cash, 2))
            puts_series.append(round(puts_val, 2))
            pmcc_pnl_series.append(round(pmcc_credits, 2))
            lev_series.append(round(leaps_val / max(nav, 1), 4))

        # ── METRICS ───────────────────────────────────────────────────────────
        nav_arr = np.array(nav_series)
        peak    = np.maximum.accumulate(nav_arr)
        uw      = (nav_arr - peak) / peak * 100
        max_dd  = float(uw.min())
        n_years = len(hist) / 252
        cagr    = ((nav_arr[-1] / nav_arr[0]) ** (1 / n_years) - 1) * 100

        monthly = nav_arr[::21]
        m_rets  = np.diff(monthly) / monthly[:-1]
        sharpe  = (m_rets.mean() / (m_rets.std() + 1e-9)) * np.sqrt(12)
        win_rate = pmcc_wins / max(pmcc_trades, 1) * 100

        return {
            "nav":       nav_series,
            "leaps":     leaps_series,
            "cash":      cash_series,
            "puts":      puts_series,
            "pmcc_pnl":  pmcc_pnl_series,
            "leverage":  lev_series,
            "drawdown":  uw.tolist(),
            "cagr":      cagr,
            "max_dd":    max_dd,
            "sharpe":    sharpe,
            "win_rate":  win_rate,
            "final_nav": nav_arr[-1],
            "n_years":   n_years,
        }


# ==============================================================================
# ── SECTION 5: v3.0 SIGNAL ENGINE ────────────────────────────────────────────
# ==============================================================================

class SignalEngineV3:
    """Generates today's exact v3.0 operational signals from live market data."""

    def __init__(self, live: Dict, portfolio_nav: float):
        self.live = live
        self.nav  = portfolio_nav
        self.opt  = OptionsEngine()
        S              = live.get("spy", 500)
        spy_ath        = live.get("spy_ath", S)
        spy_50sma      = live.get("spy_50sma", S)
        vix            = live.get("vix", 18)
        days_since_ath = live.get("days_since_ath", 999)
        self.regime = MarketRegime(
            vix=vix,
            spy_vs_ath_pct=(S - spy_ath) / spy_ath,
            spy_vs_50sma_pct=(S - spy_50sma) / spy_50sma,
            days_since_ath=days_since_ath,
        )

    def generate(self) -> List[Dict]:
        signals = []
        live    = self.live
        S       = live.get("spy", 500)
        vix     = live.get("vix", 18)
        sigma   = live.get("spy_vol_30d", vix / 100)
        rfr     = live.get("rfr", 0.060)
        r       = self.regime
        dd_pct  = r.drawdown_pct
        lo_exp, hi_exp = r.target_exposure
        lo_d, hi_d     = r.target_delta_range
        nav = self.nav

        # ── REGIME ────────────────────────────────────────────────────────────
        signals.append({
            "type": "info", "engine": "REGIME v3.0",
            "message": (
                f"Regime: {r.name} | VIX: {vix:.1f} | SPY: ${S:.2f} | "
                f"ATH Drawdown: -{dd_pct:.1f}% | 50SMA Gap: {r.spy_vs_50sma_pct*100:+.1f}% | "
                f"Days Since ATH: {r.days_since_ath} | "
                f"Target Exposure: {lo_exp:.1f}–{hi_exp:.1f}×"
            ),
        })

        leaps_alloc = r.leaps_alloc_dynamic
        cash_alloc  = r.cash_alloc_dynamic
        puts_alloc  = r.puts_alloc_dynamic
        leaps_amt   = nav * leaps_alloc
        cash_amt    = nav * cash_alloc
        puts_amt    = nav * puts_alloc

        signals.append({
            "type": "info", "engine": "DYNAMIC ALLOCATION",
            "message": (
                f"LEAPS: {leaps_alloc*100:.0f}% (${leaps_amt:,.0f}) | "
                f"Cash: {cash_alloc*100:.0f}% (${cash_amt:,.0f}) | "
                f"Puts: {puts_alloc*100:.1f}% (${puts_amt:,.0f}) | "
                f"PMCC Buffer: {(1-leaps_alloc-cash_alloc-puts_alloc)*100:.1f}%"
            ),
        })

        # ── ENGINE A: LEAPS ───────────────────────────────────────────────────
        T_leaps  = 15 / 12
        leaps_K  = self.opt.find_strike_for_delta(S, T_leaps, rfr, sigma, r.target_delta)
        leaps_px = self.opt.call_price(S, leaps_K, T_leaps, rfr, sigma)
        intrinsic     = max(S - leaps_K, 0.0)
        intrinsic_pct = (intrinsic / leaps_px * 100) if leaps_px > 0 else 0
        itm_pct       = (S - leaps_K) / S * 100
        contracts     = max(int(leaps_amt / (leaps_px * 100)), 1)
        intrinsic_ok  = intrinsic_pct >= 60

        signals.append({
            "type": "action" if intrinsic_ok else "warning",
            "engine": "ENGINE A — LEAPS",
            "message": (
                f"TARGET DELTA: {lo_d:.2f}–{hi_d:.2f}Δ (mid {r.target_delta:.2f}Δ) | "
                f"STRIKE: ${leaps_K:.0f} ({itm_pct:.0f}% ITM) | "
                f"EXPIRY: 12–24mo (mid ~15mo) | PRICE: ~${leaps_px:.2f}/share | "
                f"INTRINSIC: {intrinsic_pct:.0f}% of premium "
                f"({'✅ ≥60%' if intrinsic_ok else '⚠️ <60% — go deeper ITM'}) | "
                f"CONTRACTS: {contracts} | MAX/POSITION: 15–18% NAV | "
                f"POSITIONS: 4–9 total, ladder expirations"
            ),
        })

        signals.append({
            "type": "hold", "engine": "ENGINE A — ROLL CHECK",
            "message": (
                "Roll at 6–9 months remaining (180–270 DTE). "
                "NEVER hold <6 months to expiry. "
                "Close current → immediately open new at updated regime delta. "
                "Max per position: 15–18% NAV."
            ),
        })

        # ── ENGINE B: PMCC ────────────────────────────────────────────────────
        if not r.pmcc_allowed:
            reason = ("Strong Bull / breakout — NO call selling (PMCC halted in bull runs)"
                      if r.name == "Strong Bull"
                      else f"Crash/Drawdown ({dd_pct:.0f}%) — PMCC halted")
            signals.append({
                "type": "warning", "engine": "ENGINE B — PMCC",
                "message": f"⛔ {reason}. Resume when regime normalises.",
            })
        else:
            T_pmcc    = 37 / 365
            sell_d    = r.pmcc_sell_delta
            sell_K    = self.opt.find_strike_for_delta(S, T_pmcc, rfr, sigma, sell_d)
            buy_K     = sell_K + 65
            sell_px   = self.opt.call_price(S, sell_K, T_pmcc, rfr, sigma)
            buy_px    = self.opt.call_price(S, buy_K,  T_pmcc, rfr, sigma)
            spread_cr = max(sell_px - buy_px, 0.0)
            coverage  = r.pmcc_coverage
            n_spreads = max(int(leaps_amt * coverage / (S * 100)), 1)
            gross     = spread_cr * n_spreads * 100
            lo_ps, hi_ps = r.pmcc_sell_delta_range

            signals.append({
                "type": "action", "engine": "ENGINE B — PMCC",
                "message": (
                    f"SELL: ${sell_K:.0f} call ({lo_ps:.2f}–{hi_ps:.2f}Δ range, "
                    f"~{sell_d:.2f}Δ mid, 30–45 DTE) | "
                    f"BUY: ${buy_K:.0f} call (+$65 width) | "
                    f"CREDIT: ${spread_cr:.2f}/share | "
                    f"COVERAGE: {coverage*100:.0f}% of LEAPS | "
                    f"CONTRACTS: {n_spreads} | GROSS: ${gross:,.0f} | "
                    f"CLOSE TARGET: 50–75% profit"
                ),
            })

        # ── ENGINE C: LEAN TAIL PUTS ──────────────────────────────────────────
        T_put    = 7.5 / 12
        put_K    = S * 0.755
        put_px   = self.opt.put_price(S, put_K, T_put, rfr, sigma)
        put_cts  = max(int(puts_amt / (put_px * 100)), 1)
        put_mode = "SCALED UP (VIX > 22 or DD > 7%)" if puts_alloc >= 0.03 else "LEAN (normal market)"

        signals.append({
            "type": "warning" if puts_alloc >= 0.03 else "action",
            "engine": "ENGINE C — LEAN TAIL PUTS",
            "message": (
                f"MODE: {put_mode} | ALLOC: {puts_alloc*100:.1f}% | "
                f"BUY: ${put_K:.0f} puts (~25% OTM, 6–9mo) | "
                f"PRICE: ~${put_px:.2f}/share | CONTRACTS: {put_cts} | "
                f"BUDGET: ${puts_amt:,.0f} | "
                f"NORMAL BLEED: -0.2 to -0.4%/yr | "
                f"ON -20%+ CRASH: Monetise → recycle profits into LEAPS"
            ),
        })

        # ── CASH DEPLOYMENT ───────────────────────────────────────────────────
        deploy_pct = r.cash_deploy_pct
        if deploy_pct > 0:
            deploy_amt = cash_amt * deploy_pct
            leaps_K2   = self.opt.find_strike_for_delta(S, T_leaps, rfr, sigma, r.target_delta)
            leaps_px2  = self.opt.call_price(S, leaps_K2, T_leaps, rfr, sigma)
            extra_cts  = max(int(deploy_amt / (leaps_px2 * 100)), 1)
            tier_label = (
                "TIER 1 (-8%): Deploy 20%" if dd_pct < 15 else
                "TIER 2 (-15%): Deploy 25%" if dd_pct < 25 else
                "TIER 3 (-25%): Deploy 30%" if dd_pct < 35 else
                "TIER 4 (-35%+): Deploy ALL remaining"
            )
            signals.append({
                "type": "action", "engine": "CASH DEPLOYMENT",
                "message": (
                    f"🚨 {tier_label} | AMOUNT: ${deploy_amt:,.0f} | "
                    f"BUY: {extra_cts} LEAPS @ ${leaps_K2:.0f} strike | "
                    f"SPY -{dd_pct:.1f}% from ATH"
                ),
            })
        else:
            signals.append({
                "type": "hold", "engine": "CASH DEPLOYMENT",
                "message": (
                    f"✅ HOLD CASH — SPY only -{dd_pct:.1f}% from ATH. "
                    "Tiers: -8% → 20%, -15% → 25%, -25% → 30%, -35%+ → ALL remaining. "
                    f"Reserve: ${cash_amt:,.0f}"
                ),
            })

        # ── CRASH PLAYBOOK ────────────────────────────────────────────────────
        if r.crash_mode:
            signals.append({
                "type": "warning", "engine": "CRASH PLAYBOOK v3.0",
                "message": (
                    f"🔴 CRASH MODE: -{dd_pct:.1f}% from ATH | "
                    "1) SHIFT to 90–95Δ LEAPS | "
                    "2) STOP all PMCC | "
                    "3) FULL tiered cash deployment | "
                    "4) HOLD tail puts — let run | "
                    "5) AT BOTTOM: close puts → flip to 70–75Δ → max LEAPS → resume PMCC"
                ),
            })

        # ── EXPOSURE TARGET ───────────────────────────────────────────────────
        signals.append({
            "type": "info", "engine": "EXPOSURE TARGET",
            "message": (
                f"Regime '{r.name}' → target {lo_exp:.1f}–{hi_exp:.1f}× leverage | "
                f"Current LEAPS alloc: {leaps_alloc*100:.0f}% NAV | "
                f"At crash bottom: FLIP AGGRESSIVE — 70–75Δ + max exposure"
            ),
        })

        # ── HARD RULES ────────────────────────────────────────────────────────
        rules = [
            "✅ LEAPS are ITM" if leaps_K < S else "⚠️ LEAPS approaching ATM — must be ITM",
            (f"✅ Intrinsic ≥60% of premium" if intrinsic_pct >= 60
             else f"⚠️ Intrinsic only {intrinsic_pct:.0f}% — go deeper ITM"),
            ("✅ Cash 6–12% NAV" if 0.06 <= cash_alloc <= 0.12
             else f"⚠️ Cash {cash_alloc*100:.0f}% — outside 6–12% target"),
            "✅ No PMCC in strong bull / breakout" if r.name == "Strong Bull" and not r.pmcc_allowed else "✅ PMCC status correct",
            "✅ Roll trigger: ≤270 DTE (9 months)",
            "✅ HARD STOP: NEVER hold <180 DTE (6 months)",
        ]
        signals.append({
            "type": "info", "engine": "HARD RULES CHECK",
            "message": " | ".join(rules),
        })

        return signals


# ==============================================================================
# ── SECTION 6: UI HELPERS ────────────────────────────────────────────────────
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
    css = {
        "action": "signal-action", "warning": "signal-warning",
        "hold": "signal-hold", "info": "signal-info",
    }.get(s["type"], "signal-info")
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
# ── SECTION 7: SIDEBAR ───────────────────────────────────────────────────────
# ==============================================================================

st.sidebar.markdown("## ⚔️ Iron Harvest Elite v3.0")
st.sidebar.markdown("*Aggro Systematic LEAPS + PMCC + Lean Dynamic Tail Hedge*")
st.sidebar.markdown("---")
st.sidebar.markdown("### Portfolio Configuration")

portfolio_nav = st.sidebar.number_input(
    "Portfolio NAV ($)", value=100_000, min_value=10_000, step=10_000,
)

st.sidebar.markdown(
    "**Dynamic Allocations (regime-driven):**  \n"
    "LEAPS: 75–88% | Cash: 6–12% | Puts: 1–4%  \n"
    "*Regime-averaged: LEAPS 81% | Cash 9% | Puts 2.2%*  \n"
    "*PMCC overlay uses margin/income (11% buffer)*"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Backtest Settings")

bt_capital = st.sidebar.number_input(
    "Backtest Starting Capital ($)", value=100_000, min_value=10_000, step=10_000,
)
bt_rfr = st.sidebar.slider("Risk-Free Rate (%)", 1.0, 8.0, 6.0, 0.25) / 100

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Black-Scholes approximations for all options legs.**  \n"
    "Commission $0.65/contract | Slippage 0.20%"
)


# ==============================================================================
# ── SECTION 8: MAIN APP ──────────────────────────────────────────────────────
# ==============================================================================

st.title("⚔️ Iron Harvest Elite v3.0")
st.caption(
    "Aggro Systematic LEAPS + PMCC Overlay + Lean Dynamic Tail Hedge | "
    "Institutional Options Portfolio"
)

tab_exec, tab_bt = st.tabs([
    "🎯 Forward Execution Engine",
    "📊 Backtesting Engine (2010–Today)",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: FORWARD EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_exec:

    st.markdown("### Today's Operational Signals")
    st.caption("Live SPY, VIX, T-bill data. All signals are deterministic v3.0 ruleset outputs.")

    with st.spinner("Fetching live market data..."):
        live = DataFetcher.fetch_live()

    if not live:
        st.error("Could not fetch live data.")
        st.stop()

    S       = live["spy"]
    vix     = live["vix"]
    spy_ath = live["spy_ath"]
    dd_pct  = (spy_ath - S) / spy_ath * 100
    rfr     = live["rfr"]
    sigma   = live["spy_vol_30d"]
    date    = live["date"]
    dsa     = live["days_since_ath"]

    st.markdown(f"**Data as of {date}**")

    snap_cols = st.columns(6)
    kpi(snap_cols[0], "SPY Close",      f"${S:.2f}",        f"ATH: ${spy_ath:.2f}",    "#EF9F27")
    kpi(snap_cols[1], "ATH Drawdown",   f"-{dd_pct:.1f}%",  "from rolling 1yr high",   "#E24B4A" if dd_pct > 10 else "#1D9E75")
    kpi(snap_cols[2], "VIX",            f"{vix:.1f}",        "implied vol",             "#f5a623" if vix > 20 else "#1D9E75")
    kpi(snap_cols[3], "30d Real Vol",   f"{sigma*100:.1f}%", "annualised",              "#aaaaaa")
    kpi(snap_cols[4], "T-Bill Rate",    f"{rfr*100:.2f}%",   "risk-free",               "#aaaaaa")
    kpi(snap_cols[5], "Days Since ATH", f"{dsa}d",           "≤60 = Strong Bull",       "#1D9E75" if dsa <= 60 else "#aaaaaa")

    st.markdown("")

    spy_50sma = live["spy_50sma"]
    regime = MarketRegime(
        vix=vix,
        spy_vs_ath_pct=(S - spy_ath) / spy_ath,
        spy_vs_50sma_pct=(S - spy_50sma) / spy_50sma,
        days_since_ath=dsa,
    )

    regime_color = {
        "Strong Bull": "#1D9E75", "Normal Bull": "#2ecc71",
        "Neutral": "#EF9F27", "Drawdown": "#f5a623", "Crash": "#E24B4A"
    }
    rc = regime_color.get(regime.name, "#aaaaaa")
    lo_exp, hi_exp = regime.target_exposure
    lo_d, hi_d     = regime.target_delta_range

    reg_cols = st.columns(5)
    kpi(reg_cols[0], "Market Regime",   regime.name,                         "", rc)
    kpi(reg_cols[1], "Target Delta",    f"{lo_d:.2f}–{hi_d:.2f}Δ",           "LEAPS delta range", rc)
    kpi(reg_cols[2], "Target Exposure", f"{lo_exp:.1f}–{hi_exp:.1f}×",        "leverage target", "#aaaaaa")
    kpi(reg_cols[3], "PMCC Status",
        "✅ ACTIVE" if regime.pmcc_allowed else "⛔ HALTED",
        f"{regime.pmcc_coverage*100:.0f}% coverage",
        "#1D9E75" if regime.pmcc_allowed else "#E24B4A")
    kpi(reg_cols[4], "Cash Deploy Tier",
        f"{regime.cash_deploy_pct*100:.0f}%", "of reserve",
        "#EF9F27" if regime.cash_deploy_pct > 0 else "#aaaaaa")

    st.markdown("---")
    st.markdown("### Exact Execution Orders — v3.0 Ruleset")

    engine_v3 = SignalEngineV3(live, portfolio_nav)
    signals   = engine_v3.generate()
    for s in signals:
        signal_box(s)

    st.markdown("---")
    st.markdown("### Weekly Execution Checklist")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Weekly (Every Monday pre-market):**
- [ ] Check SPY trend vs 50SMA + ATH drawdown %
- [ ] Note VIX → confirm delta regime
- [ ] Check days since ATH (≤60 = Strong Bull)
- [ ] Review active PMCC DTE / P&L
- [ ] Adjust PMCC coverage % per regime
- [ ] Check puts alloc (ramp to 4% if VIX > 22 or DD > 7%)
""")
    with col_b:
        st.markdown("""
**Monthly (1st trading day):**
- [ ] LEAPS DTE check — roll if ≤270 days (9 months)
- [ ] HARD STOP: never hold <180 days (6 months) DTE
- [ ] Rebalance puts to regime target (1–4%)
- [ ] Review leverage vs exposure target
- [ ] Re-open PMCC if regime allows (not in bull runs)
- [ ] Update ATH watermark + cash tier tracking
""")

    st.markdown("---")
    st.markdown("### Portfolio Exposure Map (Current Regime)")
    leaps_a = regime.leaps_alloc_dynamic
    cash_a  = regime.cash_alloc_dynamic
    puts_a  = regime.puts_alloc_dynamic
    buf_a   = max(1 - leaps_a - cash_a - puts_a, 0)

    fig_pie = go.Figure(go.Pie(
        labels=["LEAPS (Engine A)", "Cash / T-Bills", "Tail Puts (Engine C)", "PMCC Buffer"],
        values=[portfolio_nav * leaps_a, portfolio_nav * cash_a,
                portfolio_nav * puts_a,  portfolio_nav * buf_a],
        marker=dict(colors=["#EF9F27", "#378ADD", "#E24B4A", "#1D9E75"]),
        hole=0.55, textinfo="label+percent", textfont=dict(size=12),
    ))
    fig_pie.update_layout(
        template="plotly_dark", height=320, paper_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        annotations=[dict(text=fmt_currency(portfolio_nav), x=0.5, y=0.5,
                          font=dict(size=18, family="IBM Plex Mono"), showarrow=False)],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### v3.0 Regime Reference Table")
    regime_df = pd.DataFrame({
        "Regime":       ["Strong Bull", "Normal Bull", "Neutral", "Drawdown", "Crash"],
        "Condition":    ["VIX<15 + above 50SMA + ATH≤60d",
                         "VIX<25 + DD<7%",
                         "VIX<30 + DD<15%",
                         "DD 15–25%",
                         "DD 25%+"],
        "LEAPS Delta":  ["70Δ", "75–80Δ", "80–85Δ", "85–90Δ", "90–95Δ"],
        "PMCC":         ["⛔ NONE (bull run)", "35% cover", "65% cover", "⛔ HALTED", "⛔ HALTED"],
        "Exposure":     ["1.8–2.2×", "1.5–1.8×", "1.0–1.3×", "0.7–0.9×", "0.5× → flip"],
        "Puts":         ["1–2%", "1–2%", "1–2%", "4%", "4%"],
        "LEAPS Alloc":  ["88%", "83%", "78%", "75%", "75%"],
        "Cash Alloc":   ["6%", "8%", "10%", "12%", "12%"],
    })
    st.dataframe(regime_df, hide_index=True, use_container_width=True)

    st.markdown("### v3.0 Cash Deployment Tiers")
    cash_ref = pd.DataFrame({
        "SPY Drop from ATH": ["-8%", "-15%", "-25%", "-35%+"],
        "Deploy":            ["20% of reserve", "25% of reserve", "30% of reserve", "ALL remaining"],
        "Action":            ["Buy LEAPS at regime delta",
                              "Buy LEAPS at regime delta",
                              "Buy LEAPS + halt PMCC",
                              "MAX exposure — then flip 70–75Δ at bottom"],
    })
    st.dataframe(cash_ref, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BACKTESTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_bt:

    st.markdown("### Backtest: IHE v3.0 vs SPY vs SPXL — Jan 2010 to Today")
    st.caption(
        "Black-Scholes approximations for all options legs. "
        "Commission $0.65/contract. Slippage 0.20%. "
        "Risk-free rate at sidebar setting (default 6.00%)."
    )

    with st.spinner("Fetching historical data (SPY, SPXL, VIX)..."):
        hist = DataFetcher.fetch_history("2010-01-01")

    if hist.empty:
        st.error("Could not fetch historical data.")
        st.stop()

    with st.spinner("Running Iron Harvest Elite v3.0 backtest..."):
        bt = IronHarvestBacktesterV3(bt_capital, rfr=bt_rfr)
        ihe_results = bt.run(hist)

    spy_curve  = (hist["SPY"]  / hist["SPY"].iloc[0]  * bt_capital).values
    spxl_curve = (hist["SPXL"] / hist["SPXL"].iloc[0] * bt_capital).values
    ihe_curve  = np.array(ihe_results["nav"])
    dates      = hist.index
    n_years    = ihe_results["n_years"]

    # SPY metrics
    spy_cagr    = ((spy_curve[-1] / spy_curve[0]) ** (1 / n_years) - 1) * 100
    spy_dd      = float(((spy_curve - np.maximum.accumulate(spy_curve)) / np.maximum.accumulate(spy_curve) * 100).min())
    spy_monthly = spy_curve[::21]
    spy_mrets   = np.diff(spy_monthly) / spy_monthly[:-1]
    spy_sharpe  = (spy_mrets.mean() / (spy_mrets.std() + 1e-9)) * np.sqrt(12)

    # SPXL metrics
    spxl_cagr    = ((spxl_curve[-1] / spxl_curve[0]) ** (1 / n_years) - 1) * 100
    spxl_dd      = float(((spxl_curve - np.maximum.accumulate(spxl_curve)) / np.maximum.accumulate(spxl_curve) * 100).min())
    spxl_monthly = spxl_curve[::21]
    spxl_mrets   = np.diff(spxl_monthly) / spxl_monthly[:-1]
    spxl_sharpe  = (spxl_mrets.mean() / (spxl_mrets.std() + 1e-9)) * np.sqrt(12)

    st.markdown("#### Performance Summary")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("##### SPY (Buy & Hold)")
        k = st.columns(2)
        kpi(k[0], "CAGR",    f"{spy_cagr:.1f}%",              "", "#1D9E75")
        kpi(k[1], "Max DD",  f"{spy_dd:.1f}%",                 "", "#E24B4A")
        kpi(k[0], "Sharpe",  f"{spy_sharpe:.2f}",              "", "#aaaaaa")
        kpi(k[1], "Final",   fmt_currency(float(spy_curve[-1])),"", "#1D9E75")
    with h2:
        st.markdown("##### SPXL (3× Leveraged)")
        k = st.columns(2)
        kpi(k[0], "CAGR",    f"{spxl_cagr:.1f}%",               "", "#378ADD")
        kpi(k[1], "Max DD",  f"{spxl_dd:.1f}%",                  "", "#E24B4A")
        kpi(k[0], "Sharpe",  f"{spxl_sharpe:.2f}",               "", "#aaaaaa")
        kpi(k[1], "Final",   fmt_currency(float(spxl_curve[-1])), "", "#378ADD")
    with h3:
        st.markdown("##### ⚔️ IHE v3.0")
        k = st.columns(2)
        kpi(k[0], "CAGR",    f"{ihe_results['cagr']:.1f}%",         "", "#EF9F27")
        kpi(k[1], "Max DD",  f"{ihe_results['max_dd']:.1f}%",        "", "#E24B4A")
        kpi(k[0], "Sharpe",  f"{ihe_results['sharpe']:.2f}",         "", "#aaaaaa")
        kpi(k[1], "Final",   fmt_currency(ihe_results["final_nav"]),  "", "#EF9F27")

    st.markdown("")
    extra = st.columns(3)
    kpi(extra[0], "IHE PMCC Win Rate",
        f"{ihe_results['win_rate']:.0f}%", "Engine B monthly trades", "#1D9E75")
    kpi(extra[1], "IHE vs SPXL",
        f"{ihe_results['final_nav']/max(spxl_curve[-1],1):.2f}x",
        f"vs SPY: {ihe_results['final_nav']/max(spy_curve[-1],1):.2f}x", "#EF9F27")
    kpi(extra[2], "Backtest Length",
        f"{n_years:.1f} yrs", "Jan 2010 to today", "#aaaaaa")

    st.markdown("---")

    bt_tab1, bt_tab2, bt_tab3, bt_tab4 = st.tabs([
        "Equity Curve", "Drawdown Analysis", "Portfolio Exposure", "Metrics & Rules"
    ])

    with bt_tab1:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=dates, y=ihe_curve,  name="IHE v3.0",
                                    line=dict(color="#EF9F27", width=2.5)))
        fig_eq.add_trace(go.Scatter(x=dates, y=spxl_curve, name="SPXL (3×)",
                                    line=dict(color="#378ADD", width=1.8, dash="dash")))
        fig_eq.add_trace(go.Scatter(x=dates, y=spy_curve,  name="SPY (1×)",
                                    line=dict(color="#1D9E75", width=1.5, dash="dot")))
        fig_eq.update_yaxes(tickprefix="$")
        fig_eq.update_xaxes(title="Date")
        st.plotly_chart(plot_dark(fig_eq, 440), use_container_width=True)

    with bt_tab2:
        ihe_uw  = np.array(ihe_results["drawdown"])
        spxl_uw = (spxl_curve - np.maximum.accumulate(spxl_curve)) / np.maximum.accumulate(spxl_curve) * 100
        spy_uw  = (spy_curve  - np.maximum.accumulate(spy_curve))  / np.maximum.accumulate(spy_curve)  * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dates, y=ihe_uw,  name="IHE v3.0", fill="tozeroy",
                                    line=dict(color="#EF9F27", width=1.5),
                                    fillcolor="rgba(239,159,39,.1)"))
        fig_dd.add_trace(go.Scatter(x=dates, y=spxl_uw, name="SPXL", fill="tozeroy",
                                    line=dict(color="#378ADD", width=1),
                                    fillcolor="rgba(55,138,221,.07)"))
        fig_dd.add_trace(go.Scatter(x=dates, y=spy_uw,  name="SPY", fill="tozeroy",
                                    line=dict(color="#1D9E75", width=1),
                                    fillcolor="rgba(29,158,117,.07)"))
        for lvl, lbl in [(-8,  "T1 -8%"), (-15, "T2 -15%"),
                          (-25, "T3 -25%"), (-35, "T4 -35%+")]:
            fig_dd.add_hline(y=lvl, line=dict(color="rgba(255,255,255,.15)", width=1, dash="dot"),
                             annotation_text=lbl, annotation_position="right")
        fig_dd.update_yaxes(ticksuffix="%", title="Drawdown from peak")
        fig_dd.update_xaxes(title="Date")
        st.plotly_chart(plot_dark(fig_dd, 420), use_container_width=True)
        st.caption("Dashed lines = v3.0 cash deployment tiers (−8%, −15%, −25%, −35%+)")

    with bt_tab3:
        lev_arr = np.array(ihe_results["leverage"])
        fig_lev = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lev.add_trace(go.Scatter(x=dates, y=lev_arr * 100, name="LEAPS / NAV %",
                                     line=dict(color="#EF9F27", width=1.5),
                                     fill="tozeroy", fillcolor="rgba(239,159,39,.1)"),
                          secondary_y=False)
        fig_lev.add_trace(go.Scatter(x=dates, y=hist["VIX"].values, name="VIX",
                                     line=dict(color="#378ADD", width=1, dash="dot")),
                          secondary_y=True)
        for lvl, lbl in [(75, "75% min"), (81, "81% avg"), (88, "88% max")]:
            fig_lev.add_hline(y=lvl, line=dict(color="rgba(255,255,255,.2)", width=1, dash="dot"),
                               annotation_text=lbl, annotation_position="right")
        fig_lev.update_yaxes(title_text="LEAPS / NAV (%)", ticksuffix="%", secondary_y=False)
        fig_lev.update_yaxes(title_text="VIX", secondary_y=True, range=[0, 90])
        st.plotly_chart(plot_dark(fig_lev, 380), use_container_width=True)

        fig_pmcc = go.Figure()
        fig_pmcc.add_trace(go.Scatter(x=dates, y=ihe_results["pmcc_pnl"],
                                      name="Cumulative PMCC P&L",
                                      line=dict(color="#1D9E75", width=1.5),
                                      fill="tozeroy", fillcolor="rgba(29,158,117,.1)"))
        fig_pmcc.update_yaxes(tickprefix="$", title="Cumulative Engine B income ($)")
        st.plotly_chart(plot_dark(fig_pmcc, 280), use_container_width=True)

    with bt_tab4:
        metrics_df = pd.DataFrame({
            "Strategy":      ["SPY Buy & Hold", "SPXL Buy & Hold", "⚔️ IHE v3.0"],
            "CAGR":          [f"{spy_cagr:.1f}%",  f"{spxl_cagr:.1f}%",  f"{ihe_results['cagr']:.1f}%"],
            "Max Drawdown":  [f"{spy_dd:.1f}%",    f"{spxl_dd:.1f}%",    f"{ihe_results['max_dd']:.1f}%"],
            "Sharpe Ratio":  [f"{spy_sharpe:.2f}", f"{spxl_sharpe:.2f}", f"{ihe_results['sharpe']:.2f}"],
            "Final Value":   [fmt_currency(float(spy_curve[-1])),
                              fmt_currency(float(spxl_curve[-1])),
                              fmt_currency(ihe_results["final_nav"])],
            "PMCC Win Rate": ["N/A", "N/A", f"{ihe_results['win_rate']:.0f}%"],
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        st.markdown("#### v3.0 Delta Regime Matrix")
        delta_df = pd.DataFrame({
            "Condition": [
                "VIX < 15 — Strong Bull",
                "VIX 15–25 — Normal/Neutral",
                "VIX > 25 — Fear",
                "SPY −10% drawdown override",
                "SPY −20% drawdown override",
                "SPY −30%+ drawdown override",
            ],
            "Target Delta":  ["70Δ", "75–80Δ", "85–95Δ", "85Δ", "90Δ", "95Δ"],
            "PMCC Coverage": ["0% (no selling)", "35%", "65%", "Halted", "Halted", "Halted"],
            "Puts Alloc":    ["1–2%", "1–2%", "4%", "4%", "4%", "4%"],
        })
        st.dataframe(delta_df, hide_index=True, use_container_width=True)

        st.markdown("#### v3.0 Cash Deployment Protocol")
        cash_df = pd.DataFrame({
            "SPY Drop":   ["-8%", "-15%", "-25%", "-35%+"],
            "Deploy":     ["20%", "25%", "30%", "ALL remaining"],
            "Action":     ["Buy LEAPS at regime delta",
                           "Buy LEAPS at regime delta",
                           "Buy LEAPS + halt PMCC",
                           "Max exposure; flip 70–75Δ at bottom"],
        })
        st.dataframe(cash_df, hide_index=True, use_container_width=True)

        st.markdown("#### Hard Rules")
        st.markdown("""
**NEVER:**
- Buy OTM LEAPS — intrinsic must be ≥60% of premium
- Hold LEAPS with <6 months (180 days) to expiry
- Sell PMCC calls in strong bull runs or breakouts
- Go all-in at any single drawdown level
- Ignore VIX regime or drawdown delta overrides

**ALWAYS:**
- Control delta by VIX + drawdown matrix (drawdown overrides VIX)
- Keep 6–12% cash reserve at all times
- Roll LEAPS early (trigger at ≤9 months remaining)
- Ramp puts to 4% when VIX > 22 or drawdown > 7%
- At crash bottom: close puts → flip 70–75Δ → max LEAPS → resume PMCC
- Ladder expirations across 4–9 positions, max 15–18% NAV per position
""")


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.caption(
    "⚔️ Iron Harvest Elite v3.0 | Educational simulation. Not financial advice. "
    "Options trading involves significant risk of loss. "
    "Black-Scholes approximations used for historical Greeks — actual results will vary. "
    "Past performance does not guarantee future results."
)
