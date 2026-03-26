# ⚔️ Iron Harvest Elite

Institutional-grade systematic options strategy platform built with Streamlit.

## Architecture

```
app.py
├── OptionsEngine          Black-Scholes pricing, delta, theta, strike search
├── MarketRegime           Regime classifier (VIX-based delta + coverage rules)
├── DataFetcher            yfinance live + historical data with caching
├── IronHarvestBacktester  Daily event-driven backtest engine (2010–today)
└── SignalEngine           Generates today's exact operational signals
```

## Two Tabs

### 1. Forward Execution Engine
Pulls live SPY, VIX, and T-bill data and outputs:
- Current market regime (Strong Bull / Neutral / Fear)
- Exact LEAPS orders: strike, delta, contracts, cost
- Call spread orders: sell strike, buy strike, credit, contracts
- Puts orders: strike, contracts, allocation
- Cash deployment signal (which tier triggered)
- Crash playbook activation if applicable
- Weekly/monthly checklist

### 2. Backtesting Engine (Jan 2010 → Today)
Compares three strategies:
- **SPY** buy and hold (1× benchmark)
- **SPXL** buy and hold (3× benchmark, actual prices)
- **Iron Harvest Elite** (full simulation)

Outputs: CAGR, Max Drawdown, Sharpe Ratio, equity curve, drawdown chart, exposure chart, CS income chart.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo → set `app.py` → Deploy

## Strategy Summary

| Component | Allocation | Engine |
|---|---|---|
| SPY LEAPS (12–18mo, 20–30% ITM) | 70–75% | A — Growth |
| Cash / T-Bills | 10–15% | Drawdown fuel |
| Protective Puts (20–30% OTM, 6–9mo) | 3–5% | C — Tail hedge |
| Call Spreads (0.20–0.30Δ, 30–45 DTE) | Overlay | B — Income |

### Delta System (non-negotiable)
| VIX | Regime | Target Delta |
|---|---|---|
| < 15 | Strong Bull | 75–80Δ |
| 15–25 | Neutral | 80–85Δ |
| > 25 | Fear/Crash | 90–95Δ |

### Cash Deployment
| SPY Drawdown | Action |
|---|---|
| −5% | Deploy 25% of reserve |
| −10% | Deploy 25% of reserve |
| −20% | Deploy 25% + halt call selling |
| −30% | Deploy 100% remaining |

## Disclaimer

Educational simulation only. Not financial advice. Options trading involves significant risk of total loss of premium paid. Consult a licensed financial advisor before trading.
