# SPY LEAPS vs SPXL vs VOO Simulator

Interactive Streamlit app comparing three investment strategies:
- **SPY LEAPS** — Deep-ITM calls (Jan 2027/2028, δ≈0.90, roll every 9mo)
- **SPXL** — Direxion 3× Daily S&P 500 ETF
- **VOO** — Vanguard S&P 500 ETF (benchmark)

## Features
- Portfolio growth chart over time
- Annual return breakdown (gross gain, theta cost, net)
- Historical crash drawdown comparison with real option math
- Capital efficiency chart ($500 deployed across each instrument)
- All charts update live with every slider change

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/spy-leaps-simulator
cd spy-leaps-simulator
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point → Deploy

## Math

**LEAPS annual return on capital:**
```
old_premium = itm_pct × 1.20
new_intrinsic = max(ann_return + itm_pct, 0)
new_premium = new_intrinsic × 1.20
return = new_premium / old_premium − 1 − theta − roll_cost
```

**LEAPS drawdown (real option math):**
- If crash < ITM%: option stays ITM, proportional intrinsic loss
- If crash > ITM%: option goes OTM, drops to small time value only
- Max loss: 100% of premium paid (not more)

**SPXL vol drag (Ito's lemma):**
```
drag = L(L−1)/2 × σ²
net_return = 3 × r_SPY − drag − 0.91%
```

## Disclaimer

Educational simulation only. Not financial advice. Options involve risk of total loss of premium. Consult a licensed advisor before trading.
