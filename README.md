# Portfolio Risk Engine

A multi-asset quantitative risk analysis tool built in Python, covering equities and fixed income in a single framework. The engine combines GARCH(1,1) volatility modelling, yield curve bootstrapping, bond pricing analytics, and Monte Carlo simulation to compute Value-at-Risk and Expected Shortfall across asset classes.

**[Live Demo](https://your-streamlit-url.streamlit.app)** · Built with Python, Streamlit, and Plotly

---

## What It Does

### Equity Risk
- Fits GARCH(1,1) with Student-t innovations to each stock, producing time-varying volatility forecasts
- Runs Monte Carlo simulation using GARCH-forecast covariance to compute portfolio VaR and Expected Shortfall
- Computes Sharpe ratio, volatility persistence, and half-life of shocks
- Pulls live price data and quarterly fundamentals via Yahoo Finance

### Fixed Income
- Bootstraps a zero-coupon yield curve from US Treasury par yields (live or manual input)
- Prices coupon-bearing bonds off the bootstrapped curve using exact discounting
- Computes Macaulay duration, modified duration, convexity, and DV01 via numerical derivatives
- Runs scenario analysis under parallel yield curve shifts with full repricing
- Compares full repricing against duration-convexity approximation to demonstrate where the linear approximation breaks down

### Multi-Asset Risk
- Models daily yield changes with GARCH and translates yield volatility into bond P&L through duration
- Estimates cross-asset correlation between equity returns and yield changes from historical data
- Runs joint Monte Carlo simulation across equities and bonds to produce a combined portfolio VaR
- Quantifies the diversification benefit of holding both asset classes versus treating them independently

---

## Technical Architecture

```
app.py
├── DataEngine          Fetches equity prices, computes log returns
├── RiskEngine          GARCH fitting, Monte Carlo VaR/ES, Sharpe ratio
├── FundamentalsEngine  Quarterly financials from Yahoo Finance
├── YieldCurveEngine    Par yield ingestion, zero-rate bootstrap, curve shifting
├── BondEngine          Bond pricing, YTM, duration/convexity, scenario analysis
├── MultiAssetRiskEngine Joint equity-bond VaR with cross-asset correlation
└── Streamlit UI        Interactive dashboard with Plotly visualisations
```

### Key Methods

**Yield Curve Bootstrap** — Par yields at standard tenors (3M to 30Y) are bootstrapped into zero-coupon rates. Short-end instruments (under 1Y) are treated as zero-coupon. For longer tenors, the algorithm solves for the discount factor that prices a par bond at 100, using previously bootstrapped zero rates for intermediate coupon payments.

**Bond Analytics** — Duration and convexity are computed numerically by bumping the yield to maturity by +/- 1 basis point and observing the resulting price changes. This avoids closed-form formula errors and mirrors how production systems handle instruments with irregular features.

**Multi-Asset VaR** — Equity risk is modelled through GARCH-forecast return distributions. Bond risk is modelled by fitting GARCH to daily yield changes (using 10Y Treasury as the risk factor) and translating simulated yield shocks into P&L via the duration-convexity approximation. Both are simulated jointly using their historical correlation structure, and the combined distribution produces a portfolio-level VaR that captures diversification effects.

---

## Running Locally

```bash
git clone https://github.com/kavishetty0402/portfolio-risk-engine.git
cd portfolio-risk-engine
pip install -r requirements.txt
streamlit run app.py
```

### Requirements

```
streamlit
yfinance
pandas
numpy
plotly
arch
scipy
```

Python 3.9 or higher.

---

## Usage

1. Enter equity symbols and investment amounts in the sidebar
2. Toggle **Include Bonds** to add fixed income positions (coupon rate, maturity, face value, investment size)
3. Click **Run Analysis**

The dashboard produces:
- Portfolio-level VaR and Expected Shortfall metrics
- GARCH conditional volatility time series for each stock
- Simulated return distribution with VaR/ES markers
- US Treasury yield curve (par yields vs bootstrapped zero rates)
- Bond-level analytics: price, YTM, duration, convexity, DV01, and price-yield curves
- Scenario P&L under interest rate shocks (+/- 25 to 200 basis points)
- Multi-asset risk decomposition with diversification benefit and cross-asset correlation matrix

---

## Repository Structure

```
portfolio-risk-engine/
├── app.py                 Main application (data, risk, FI engines + UI)
├── requirements.txt       Python dependencies
├── README.md
├── LICENSE
└── screenshots/
    ├── equity-risk.png
    ├── yield-curve.png
    ├── scenario-analysis.png
    └── multi-asset.png
```

---

## Quantitative References

- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
- Hull, J.C. (2022). *Options, Futures, and Other Derivatives*, 11th Edition. Chapters 4 (Interest Rates), 22 (Value at Risk).
- Fabozzi, F.J. (2007). *Fixed Income Analysis*, 2nd Edition. Duration, convexity, and yield curve construction.

---

## License

MIT

---

*Built by [Kavish Shetty](https://linkedin.com/in/kavishshetty) — UCL Mathematics with Economics, Class of 2025.*
