import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from arch import arch_model
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Portfolio Risk Engine",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# DATA ENGINE
# ============================================================

class DataEngine:

    def __init__(self, tickers, period="2y"):
        self.tickers = tickers
        self.period = period
        self.prices = None
        self.returns = None

    def fetch_prices(self):
        data = yf.download(self.tickers, period=self.period, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=self.tickers[0])
        data = data.dropna(axis=1, how='all').dropna()
        self.prices = data
        return data

    def compute_log_returns(self):
        if self.prices is None:
            self.fetch_prices()
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        return self.returns

    def get_summary_stats(self):
        if self.returns is None:
            self.compute_log_returns()
        stats = pd.DataFrame({
            'Mean (Daily)': self.returns.mean(),
            'Std (Daily)': self.returns.std(),
            'Std (Annual)': self.returns.std() * np.sqrt(252),
            'Skewness': self.returns.skew(),
            'Kurtosis': self.returns.kurtosis(),
            'Min': self.returns.min(),
            'Max': self.returns.max()
        })
        return stats.T


# ============================================================
# RISK ENGINE
# ============================================================

class RiskEngine:

    def __init__(self, returns):
        self.returns = returns
        self.garch_models = {}
        self.garch_results = {}

    def fit_garch(self, ticker):
        series = self.returns[ticker] * 100
        model = arch_model(series, vol='Garch', p=1, q=1, dist='t')
        result = model.fit(disp='off')
        self.garch_models[ticker] = model
        self.garch_results[ticker] = result
        return result

    def fit_all(self):
        for ticker in self.returns.columns:
            self.fit_garch(ticker)

    def get_garch_summary(self, ticker):
        if ticker not in self.garch_results:
            self.fit_garch(ticker)
        result = self.garch_results[ticker]
        params = result.params
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        persistence = alpha + beta
        current_vol = result.conditional_volatility.iloc[-1] / 100
        forecast = result.forecast(horizon=1)
        forecast_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
        return {
            'ticker': ticker,
            'alpha': alpha,
            'beta': beta,
            'persistence': persistence,
            'half_life': half_life,
            'current_vol_daily': current_vol,
            'current_vol_annual': current_vol * np.sqrt(252),
            'forecast_vol_daily': forecast_vol,
            'forecast_vol_annual': forecast_vol * np.sqrt(252),
            'conditional_volatility': result.conditional_volatility / 100
        }

    def compute_portfolio_var(self, weights, confidence=0.95, n_simulations=10000):
        if not self.garch_results:
            self.fit_all()
        weights = np.array(weights)
        weights = weights / weights.sum()
        means = self.returns.mean().values
        forecast_vols = []
        for ticker in self.returns.columns:
            result = self.garch_results[ticker]
            forecast = result.forecast(horizon=1)
            vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            forecast_vols.append(vol)
        forecast_vols = np.array(forecast_vols)
        corr_matrix = self.returns.corr().values
        cov_matrix = np.outer(forecast_vols, forecast_vols) * corr_matrix
        simulated_returns = np.random.multivariate_normal(means, cov_matrix, size=n_simulations)
        portfolio_returns = simulated_returns @ weights
        var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
        losses = portfolio_returns[portfolio_returns <= -var]
        es = -losses.mean() if len(losses) > 0 else var
        return {
            'var': var, 'es': es, 'confidence': confidence,
            'simulated_returns': portfolio_returns,
            'weights': weights, 'forecast_vols': forecast_vols
        }

    def compute_sharpe_ratio(self, weights, risk_free_rate=0.04):
        weights = np.array(weights)
        weights = weights / weights.sum()
        portfolio_daily_return = self.returns.mean() @ weights
        portfolio_annual_return = portfolio_daily_return * 252
        cov_matrix = self.returns.cov().values
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)
        sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        return {
            'sharpe_ratio': sharpe_ratio,
            'portfolio_annual_return': portfolio_annual_return,
            'portfolio_volatility': portfolio_volatility,
            'excess_return': portfolio_annual_return - risk_free_rate,
            'risk_free_rate': risk_free_rate
        }


# ============================================================
# FUNDAMENTALS ENGINE
# ============================================================

class FundamentalsEngine:

    def __init__(self, tickers):
        self.tickers = tickers
        self.data = {}

    def fetch_fundamentals(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            qi = stock.quarterly_income_stmt
            result = {
                'name': info.get('shortName', ticker.replace('.NS', '')),
                'market_cap': info.get('marketCap'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'financial_currency': info.get('financialCurrency', 'N/A'),
                'stock_currency': info.get('currency', 'N/A'),
                'quarterly_pat': None, 'quarterly_revenue': None, 'available': False
            }
            if qi is not None and not qi.empty:
                if 'Net Income' in qi.index:
                    pat = qi.loc['Net Income'].dropna()
                    if len(pat) > 0:
                        result['quarterly_pat'] = pat
                if 'Total Revenue' in qi.index:
                    rev = qi.loc['Total Revenue'].dropna()
                    if len(rev) > 0:
                        result['quarterly_revenue'] = rev
                result['available'] = result['quarterly_pat'] is not None
            self.data[ticker] = result
        except Exception as e:
            st.warning(f"Could not fetch fundamentals for {ticker}: {e}")
            self.data[ticker] = {
                'name': ticker.replace('.NS', ''), 'market_cap': None,
                'trailing_pe': None, 'forward_pe': None,
                'financial_currency': 'N/A', 'stock_currency': 'N/A',
                'quarterly_pat': None, 'quarterly_revenue': None, 'available': False
            }

    def fetch_all(self):
        for ticker in self.tickers:
            self.fetch_fundamentals(ticker)


# ============================================================
# YIELD CURVE ENGINE
# ============================================================

class YieldCurveEngine:

    DEFAULT_CURVE = {
        0.25: 4.30, 0.5: 4.25, 1.0: 4.10, 2.0: 3.95,
        3.0: 3.85, 5.0: 3.80, 7.0: 3.85, 10.0: 3.95,
        20.0: 4.25, 30.0: 4.35
    }

    TREASURY_TICKERS = {
        0.25: '^IRX',
        5.0: '^FVX',
        10.0: '^TNX',
        30.0: '^TYX'
    }

    def __init__(self):
        self.tenors = np.array([])
        self.par_yields = np.array([])
        self.zero_rates = np.array([])
        self.discount_factors = np.array([])

    def fetch_live_yields(self):
        """Fetch Treasury yields from yfinance, fill gaps with defaults."""
        yields = dict(self.DEFAULT_CURVE)
        for tenor, ticker in self.TREASURY_TICKERS.items():
            try:
                data = yf.download(ticker, period='5d', progress=False)['Close']
                if len(data) > 0:
                    val = float(data.iloc[-1])
                    if val > 0:
                        yields[tenor] = val
            except:
                pass
        self._build_from_dict(yields)
        return yields

    def set_manual_curve(self, tenors_yields_dict):
        """Set curve manually. Input: {tenor_years: yield_percent}."""
        self._build_from_dict(tenors_yields_dict)

    def _build_from_dict(self, yields_dict):
        tenors = sorted(yields_dict.keys())
        self.tenors = np.array(tenors)
        self.par_yields = np.array([yields_dict[t] / 100.0 for t in tenors])
        self._bootstrap()

    def _bootstrap(self):
        """Bootstrap zero-coupon rates from par yields."""
        n = len(self.tenors)
        self.zero_rates = np.zeros(n)
        self.discount_factors = np.zeros(n)

        for i in range(n):
            T = self.tenors[i]
            y = self.par_yields[i]

            if T <= 1.0:
                self.zero_rates[i] = y
                self.discount_factors[i] = np.exp(-y * T)
            else:
                c = y / 2.0
                coupon_pv = 0.0
                payment_times = np.arange(0.5, T, 0.5)
                for t_j in payment_times:
                    z_j = self._interpolate_partial(t_j, i)
                    coupon_pv += c * np.exp(-z_j * t_j)
                df_T = (1.0 - coupon_pv) / (1.0 + c)
                if df_T > 0:
                    self.zero_rates[i] = -np.log(df_T) / T
                else:
                    self.zero_rates[i] = y
                self.discount_factors[i] = np.exp(-self.zero_rates[i] * T)

    def _interpolate_partial(self, t, up_to_idx):
        """Interpolate zero rate using only rates bootstrapped so far."""
        if up_to_idx == 0:
            return float(self.par_yields[0])
        avail_t = self.tenors[:up_to_idx]
        avail_z = self.zero_rates[:up_to_idx]
        if t <= avail_t[0]:
            return float(avail_z[0])
        if t >= avail_t[-1]:
            return float(avail_z[-1])
        idx = np.searchsorted(avail_t, t)
        t0, t1 = avail_t[idx - 1], avail_t[idx]
        z0, z1 = avail_z[idx - 1], avail_z[idx]
        return float(z0 + (z1 - z0) * (t - t0) / (t1 - t0))

    def interpolate_zero(self, t):
        """Interpolate zero rate for arbitrary tenor using full curve."""
        if len(self.zero_rates) == 0:
            return 0.04
        if t <= self.tenors[0]:
            return float(self.zero_rates[0])
        if t >= self.tenors[-1]:
            return float(self.zero_rates[-1])
        if len(self.tenors) >= 4:
            cs = CubicSpline(self.tenors, self.zero_rates)
            return float(cs(t))
        idx = np.searchsorted(self.tenors, t)
        t0, t1 = self.tenors[idx - 1], self.tenors[idx]
        z0, z1 = self.zero_rates[idx - 1], self.zero_rates[idx]
        return float(z0 + (z1 - z0) * (t - t0) / (t1 - t0))

    def get_discount_factor(self, t):
        return np.exp(-self.interpolate_zero(t) * t)

    def get_shifted_curve(self, shift_bps, mode='parallel'):
        """Return new YieldCurveEngine with shifted par yields."""
        shift = shift_bps / 100.0  # par yields stored as pct for dict
        shifted_dict = {}
        for i, t in enumerate(self.tenors):
            base_pct = self.par_yields[i] * 100.0
            if mode == 'parallel':
                shifted_dict[t] = base_pct + shift
            elif mode == 'steepener':
                factor = (t - self.tenors[0]) / (self.tenors[-1] - self.tenors[0]) if self.tenors[-1] != self.tenors[0] else 0
                shifted_dict[t] = base_pct + shift * (2 * factor - 1)
            elif mode == 'flattener':
                factor = (t - self.tenors[0]) / (self.tenors[-1] - self.tenors[0]) if self.tenors[-1] != self.tenors[0] else 0
                shifted_dict[t] = base_pct + shift * (1 - 2 * factor)
            else:
                shifted_dict[t] = base_pct + shift
        new_curve = YieldCurveEngine()
        new_curve.set_manual_curve(shifted_dict)
        return new_curve


# ============================================================
# BOND ENGINE
# ============================================================

class BondEngine:

    def __init__(self, curve):
        self.curve = curve

    def price_bond(self, coupon_rate, maturity, face_value=100.0, frequency=2):
        """Price a bond using the bootstrapped zero curve."""
        c = coupon_rate * face_value / frequency
        dt = 1.0 / frequency
        payment_times = np.arange(dt, maturity + dt / 2, dt)
        price = 0.0
        for t in payment_times:
            df = self.curve.get_discount_factor(t)
            if t >= maturity - dt / 2:
                price += (c + face_value) * df
            else:
                price += c * df
        return price

    def _price_at_flat_yield(self, y, coupon_rate, maturity, face_value=100.0, frequency=2):
        """Price bond discounting all cash flows at a single flat yield."""
        c = coupon_rate * face_value / frequency
        n = int(round(maturity * frequency))
        yf = y / frequency
        if abs(yf) < 1e-12:
            return c * n + face_value
        price = sum(c / (1 + yf) ** (i + 1) for i in range(n))
        price += face_value / (1 + yf) ** n
        return price

    def compute_ytm(self, price, coupon_rate, maturity, face_value=100.0, frequency=2):
        """Compute yield to maturity via Brent's method."""
        def f(y):
            return self._price_at_flat_yield(y, coupon_rate, maturity, face_value, frequency) - price
        try:
            return brentq(f, -0.05, 2.0, xtol=1e-10)
        except:
            return None

    def compute_analytics(self, coupon_rate, maturity, face_value=100.0, frequency=2):
        """Compute price, YTM, duration, convexity, DV01 using numerical derivatives."""
        price = self.price_bond(coupon_rate, maturity, face_value, frequency)
        ytm = self.compute_ytm(price, coupon_rate, maturity, face_value, frequency)

        if ytm is None:
            return None

        dy = 0.0001  # 1 basis point
        p_up = self._price_at_flat_yield(ytm + dy, coupon_rate, maturity, face_value, frequency)
        p_down = self._price_at_flat_yield(ytm - dy, coupon_rate, maturity, face_value, frequency)

        mod_dur = -(p_up - p_down) / (2 * dy * price)
        convexity = (p_up + p_down - 2 * price) / (dy ** 2 * price)
        mac_dur = mod_dur * (1 + ytm / frequency)
        dv01 = mod_dur * price / 100 * 0.01  # dollar value per $100 face per 1bp

        return {
            'price': price,
            'ytm': ytm,
            'macaulay_duration': mac_dur,
            'modified_duration': mod_dur,
            'convexity': convexity,
            'dv01': dv01,
            'face_value': face_value,
            'coupon_rate': coupon_rate,
            'maturity': maturity
        }

    def scenario_analysis(self, coupon_rate, maturity, face_value=100.0, frequency=2,
                          shifts_bps=None):
        """Reprice a bond under parallel yield curve shifts. Returns list of scenario results."""
        if shifts_bps is None:
            shifts_bps = [-200, -100, -50, -25, 0, 25, 50, 100, 200]

        base_analytics = self.compute_analytics(coupon_rate, maturity, face_value, frequency)
        if base_analytics is None:
            return None

        base_price = base_analytics['price']
        mod_dur = base_analytics['modified_duration']
        conv = base_analytics['convexity']

        results = []
        for shift in shifts_bps:
            shifted_curve = self.curve.get_shifted_curve(shift)
            shifted_engine = BondEngine(shifted_curve)
            new_price = shifted_engine.price_bond(coupon_rate, maturity, face_value, frequency)

            # Full repricing P&L
            pnl_full = new_price - base_price

            # Duration approximation
            dy = shift / 10000.0
            pnl_dur_approx = -mod_dur * base_price * dy + 0.5 * conv * base_price * dy ** 2

            results.append({
                'shift_bps': shift,
                'new_price': new_price,
                'pnl_full': pnl_full,
                'pnl_pct': pnl_full / base_price * 100,
                'pnl_duration_approx': pnl_dur_approx,
                'approx_error': pnl_full - pnl_dur_approx
            })

        return {'base': base_analytics, 'scenarios': results}

    def price_yield_curve(self, coupon_rate, maturity, face_value=100.0, frequency=2,
                          yield_range=None):
        """Generate price-yield relationship data for plotting."""
        if yield_range is None:
            analytics = self.compute_analytics(coupon_rate, maturity, face_value, frequency)
            if analytics:
                center = analytics['ytm']
            else:
                center = 0.04
            yield_range = np.linspace(max(0.001, center - 0.03), center + 0.03, 100)

        prices = [self._price_at_flat_yield(y, coupon_rate, maturity, face_value, frequency)
                  for y in yield_range]
        return yield_range, prices


# ============================================================
# MULTI-ASSET RISK ENGINE
# ============================================================

class MultiAssetRiskEngine:
    """Computes combined VaR across equities and bonds."""

    def __init__(self, equity_risk_engine=None, bond_engine=None, bonds=None):
        self.equity_re = equity_risk_engine
        self.bond_engine = bond_engine
        self.bonds = bonds or []

    def compute_bond_portfolio_risk(self, yield_history, confidence=0.95, n_sims=10000):
        """
        Compute VaR for a bond portfolio using GARCH on yield changes.
        yield_history: Series of daily yield levels (e.g. 10Y Treasury).
        """
        yield_changes = yield_history.diff().dropna() / 100  # convert pct to decimal

        # Fit GARCH to yield changes
        series_scaled = yield_changes * 10000  # scale to bps for numerical stability
        try:
            model = arch_model(series_scaled, vol='Garch', p=1, q=1, dist='t')
            garch_result = model.fit(disp='off')
            forecast = garch_result.forecast(horizon=1)
            forecast_vol_bps = np.sqrt(forecast.variance.iloc[-1, 0])
            forecast_vol = forecast_vol_bps / 10000
        except:
            forecast_vol = yield_changes.std()

        mean_change = yield_changes.mean()

        # Simulate yield changes
        sim_yield_changes = np.random.normal(mean_change, forecast_vol, n_sims)

        # Translate to portfolio P&L via duration
        total_pnl = np.zeros(n_sims)
        bond_details = []

        for bond in self.bonds:
            analytics = self.bond_engine.compute_analytics(
                bond['coupon_rate'], bond['maturity'],
                bond['face_value'], bond.get('frequency', 2)
            )
            if analytics is None:
                continue

            # Number of bonds = investment / price per bond * 100 (face)
            n_bonds = bond['investment'] / (analytics['price'] / analytics['face_value'] * analytics['face_value'])
            position_value = n_bonds * analytics['price']

            # P&L = -ModDur * Value * dy + 0.5 * Conv * Value * dy^2
            bond_pnl = (-analytics['modified_duration'] * position_value * sim_yield_changes +
                        0.5 * analytics['convexity'] * position_value * sim_yield_changes ** 2)
            total_pnl += bond_pnl

            bond_details.append({
                'bond': bond,
                'analytics': analytics,
                'position_value': position_value,
                'n_bonds': n_bonds
            })

        var = -np.percentile(total_pnl, (1 - confidence) * 100)
        losses = total_pnl[total_pnl <= -var]
        es = -losses.mean() if len(losses) > 0 else var

        return {
            'var': var, 'es': es, 'confidence': confidence,
            'simulated_pnl': total_pnl,
            'forecast_yield_vol_bps': forecast_vol * 10000,
            'bond_details': bond_details
        }

    def compute_combined_var(self, equity_weights, equity_total_investment,
                             yield_history, confidence=0.95, n_sims=10000):
        """Compute combined VaR across equities and bonds with correlation."""
        if self.equity_re is None or self.bond_engine is None:
            return None

        equity_returns = self.equity_re.returns
        yield_changes = yh.reindex(equity_returns.index).diff().dropna() / 100
        common_idx = equity_returns.index.intersection(yield_changes.index)
        eq_aligned = equity_returns.loc[common_idx]
        yc_aligned = yield_changes.loc[common_idx]

        # Equity simulation (using GARCH forecast vols)
        eq_weights = np.array(equity_weights)
        eq_weights = eq_weights / eq_weights.sum()
        eq_means = eq_aligned.mean().values
        eq_forecast_vols = []
        for ticker in eq_aligned.columns:
            if ticker in self.equity_re.garch_results:
                result = self.equity_re.garch_results[ticker]
                forecast = result.forecast(horizon=1)
                vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            else:
                vol = eq_aligned[ticker].std()
            eq_forecast_vols.append(vol)
        eq_forecast_vols = np.array(eq_forecast_vols)

        # Yield change statistics
        yc_mean = float(yc_aligned.mean())
        try:
            yc_scaled = yc_aligned * 10000
            model = arch_model(yc_scaled, vol='Garch', p=1, q=1, dist='t')
            gr = model.fit(disp='off')
            fc = gr.forecast(horizon=1)
            yc_vol = np.sqrt(fc.variance.iloc[-1, 0]) / 10000
        except:
            yc_vol = float(yc_aligned.std())

        # Build combined mean and covariance
        n_eq = len(eq_aligned.columns)
        combined_means = np.append(eq_means, yc_mean)
        combined_vols = np.append(eq_forecast_vols, yc_vol)

        # Correlation: equities + yield changes
        combined_data = pd.concat([eq_aligned, yc_aligned.rename('yield_change')], axis=1).dropna()
        corr = combined_data.corr().values
        cov = np.outer(combined_vols, combined_vols) * corr

        # Ensure positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Simulate
        sims = np.random.multivariate_normal(combined_means, cov, size=n_sims)
        eq_sims = sims[:, :n_eq]
        yc_sims = sims[:, n_eq]

        # Equity P&L
        eq_portfolio_returns = eq_sims @ eq_weights
        eq_pnl = eq_portfolio_returns * equity_total_investment

        # Bond P&L
        bond_pnl = np.zeros(n_sims)
        for bond in self.bonds:
            analytics = self.bond_engine.compute_analytics(
                bond['coupon_rate'], bond['maturity'],
                bond['face_value'], bond.get('frequency', 2)
            )
            if analytics is None:
                continue
            n_bonds = bond['investment'] / (analytics['price'] / analytics['face_value'] * analytics['face_value'])
            position_value = n_bonds * analytics['price']
            bond_pnl += (-analytics['modified_duration'] * position_value * yc_sims +
                         0.5 * analytics['convexity'] * position_value * yc_sims ** 2)

        total_pnl = eq_pnl + bond_pnl
        var = -np.percentile(total_pnl, (1 - confidence) * 100)
        losses = total_pnl[total_pnl <= -var]
        es = -losses.mean() if len(losses) > 0 else var

        # Correlation between equity and bond returns
        eq_bond_corr = np.corrcoef(eq_portfolio_returns, yc_sims)[0, 1]

        return {
            'var': var, 'es': es, 'confidence': confidence,
            'total_pnl': total_pnl, 'eq_pnl': eq_pnl, 'bond_pnl': bond_pnl,
            'eq_bond_correlation': eq_bond_corr,
            'equity_total': equity_total_investment,
            'bond_total': sum(b['investment'] for b in self.bonds),
            'portfolio_total': equity_total_investment + sum(b['investment'] for b in self.bonds)
        }


# ============================================================
# HELPERS
# ============================================================

def format_financial_number(value, currency):
    if pd.isna(value) or value is None:
        return "N/A"
    if currency == 'INR':
        cr = value / 1e7
        if abs(cr) >= 100000:
            return f"₹{cr / 100000:,.2f} L Cr"
        return f"₹{cr:,.0f} Cr"
    else:
        if abs(value) >= 1e12:
            return f"${value / 1e12:,.2f}T"
        elif abs(value) >= 1e9:
            return f"${value / 1e9:,.2f}B"
        elif abs(value) >= 1e6:
            return f"${value / 1e6:,.1f}M"
        else:
            return f"${value:,.0f}"


def compute_yoy_growth(series):
    growth = {}
    dates = series.index.sort_values(ascending=False)
    for date in dates:
        target = date - pd.DateOffset(years=1)
        matches = [d for d in dates if abs((d - target).days) <= 45 and d != date]
        if matches:
            prior = matches[0]
            prior_val = series[prior]
            current_val = series[date]
            if prior_val != 0 and not pd.isna(prior_val) and not pd.isna(current_val):
                growth[date] = (current_val - prior_val) / abs(prior_val) * 100
            else:
                growth[date] = None
        else:
            growth[date] = None
    return growth


def fetch_yield_history(period="2y"):
    """Fetch 10Y Treasury yield history for bond risk modelling."""
    try:
        data = yf.download('^TNX', period=period, progress=False)['Close']
        if isinstance(data, pd.DataFrame):
            data = data.squeeze()
        if len(data) > 0:
            return data
    except:
        pass
    return None


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📊 Portfolio Risk Engine")
st.markdown("**Multi-asset risk analysis powered by GARCH volatility modelling**")

# ---- SIDEBAR ----

with st.sidebar:
    st.header("Portfolio Setup")

    # -- Equities --
    st.subheader("Equities")
    market = st.radio("Market", ["US Stocks", "India (NSE)"], horizontal=True)

    if market == "India (NSE)":
        default_tickers = "RELIANCE, TCS, HDFCBANK"
        suffix = ".NS"
        currency = "₹"
        example = "e.g., RELIANCE, TCS, INFY"
    else:
        default_tickers = "AAPL, MSFT, NVDA"
        suffix = ""
        currency = "$"
        example = "e.g., AAPL, MSFT, NVDA"

    st.markdown(f"*{example}*")
    ticker_input = st.text_input("Enter stock symbols (comma-separated)", default_tickers)

    raw_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    tickers = []
    for t in raw_tickers:
        if market == "India (NSE)" and not t.endswith(".NS"):
            tickers.append(f"{t}{suffix}")
        else:
            tickers.append(t)

    st.markdown("**Investment Amounts**")
    investments = {}
    for ticker in tickers:
        display_name = ticker.replace(".NS", "")
        investments[ticker] = st.number_input(
            f"{display_name}", min_value=0, value=10000, step=1000, key=f"inv_{ticker}"
        )

    total_equity_investment = sum(investments.values())
    if total_equity_investment > 0:
        weights = [investments[t] / total_equity_investment for t in tickers]
    else:
        weights = [1 / len(tickers)] * len(tickers) if tickers else []

    # -- Fixed Income --
    st.markdown("---")
    st.subheader("Fixed Income")
    include_bonds = st.checkbox("Include Bonds", value=False)

    bonds = []
    if include_bonds:
        n_bonds = st.number_input("Number of bonds", min_value=1, max_value=5, value=1)
        for i in range(int(n_bonds)):
            with st.expander(f"Bond {i + 1}", expanded=(i == 0)):
                label = st.text_input("Label", f"Bond {i + 1}", key=f"blabel_{i}")
                coupon = st.number_input("Coupon Rate (%)", 0.0, 15.0, 4.0, 0.125, key=f"coup_{i}")
                mat = st.number_input("Maturity (years)", 0.5, 30.0, 5.0, 0.5, key=f"mat_{i}")
                fv = st.number_input("Face Value ($)", 100, 100000, 1000, 100, key=f"fv_{i}")
                bond_inv = st.number_input("Investment ($)", 0, 10000000, 10000, 1000, key=f"binv_{i}")
                bonds.append({
                    'label': label,
                    'coupon_rate': coupon / 100.0,
                    'maturity': mat,
                    'face_value': float(fv),
                    'investment': bond_inv,
                    'frequency': 2
                })

    # -- Settings --
    st.markdown("---")
    period = st.selectbox("Historical Data", ["1y", "2y", "5y"], index=2)
    confidence = st.selectbox("Confidence Level", [0.95, 0.99], index=0,
                              format_func=lambda x: f"{x:.0%}")
    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)


# ---- MAIN ANALYSIS ----

if run_analysis:

    has_equities = len(tickers) > 0 and total_equity_investment > 0
    has_bonds = include_bonds and len(bonds) > 0 and any(b['investment'] > 0 for b in bonds)

    # -- Equity data --
    returns = None
    risk_engine = None
    var_result = None
    sharpe_result = None
    fundamentals_engine = None

    if has_equities:
        with st.spinner("Fetching equity data..."):
            try:
                data_engine = DataEngine(tickers, period=period)
                returns = data_engine.compute_log_returns()
                if returns.empty or len(returns.columns) == 0:
                    st.error("Could not fetch equity data. Check ticker symbols.")
                    has_equities = False
                else:
                    valid_tickers = list(returns.columns)
                    if len(valid_tickers) < len(tickers):
                        dropped = set(tickers) - set(valid_tickers)
                        st.warning(f"No data for: {', '.join(t.replace('.NS', '') for t in dropped)}")
                        tickers = valid_tickers
                        total_equity_investment = sum(investments.get(t, 10000) for t in tickers)
                        if total_equity_investment > 0:
                            weights = [investments.get(t, 10000) / total_equity_investment for t in tickers]
                        else:
                            weights = [1 / len(tickers)] * len(tickers)
            except Exception as e:
                st.error(f"Error fetching equity data: {e}")
                has_equities = False

        if has_equities:
            with st.spinner("Fitting GARCH models..."):
                risk_engine = RiskEngine(returns)
                risk_engine.fit_all()
            with st.spinner("Running Monte Carlo simulation..."):
                var_result = risk_engine.compute_portfolio_var(weights, confidence=confidence)
            with st.spinner("Calculating performance metrics..."):
                sharpe_result = risk_engine.compute_sharpe_ratio(weights)
            with st.spinner("Fetching fundamentals..."):
                fundamentals_engine = FundamentalsEngine(tickers)
                fundamentals_engine.fetch_all()

    # -- Fixed Income data --
    curve_engine = None
    bond_engine = None
    yield_history = None
    bond_risk_result = None
    combined_result = None

    if has_bonds:
        with st.spinner("Building yield curve..."):
            curve_engine = YieldCurveEngine()
            curve_engine.fetch_live_yields()
            bond_engine = BondEngine(curve_engine)

        with st.spinner("Fetching yield history for risk modelling..."):
            yield_history = fetch_yield_history(period=period)

        if yield_history is not None and len(yield_history) > 50:
            with st.spinner("Computing bond portfolio risk..."):
                ma_engine = MultiAssetRiskEngine(risk_engine, bond_engine, bonds)
                bond_risk_result = ma_engine.compute_bond_portfolio_risk(
                    yield_history, confidence=confidence)

                if has_equities and risk_engine is not None:
                    with st.spinner("Computing multi-asset VaR..."):
                        combined_result = ma_engine.compute_combined_var(
                            weights, total_equity_investment,
                            yield_history, confidence=confidence)

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    st.markdown("---")
    st.header("Portfolio Risk Summary")

    total_investment = total_equity_investment + sum(b['investment'] for b in bonds)

    # -- Summary metrics --
    if has_equities and has_bonds and combined_result:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"Combined {confidence:.0%} VaR",
                  f"${combined_result['var']:,.0f}",
                  f"{combined_result['var'] / combined_result['portfolio_total']:.2%} of portfolio")
        c2.metric(f"Combined {confidence:.0%} ES",
                  f"${combined_result['es']:,.0f}",
                  f"{combined_result['es'] / combined_result['portfolio_total']:.2%} of portfolio")
        c3.metric("Equity Allocation",
                  f"${combined_result['equity_total']:,.0f}",
                  f"{combined_result['equity_total'] / combined_result['portfolio_total']:.0%}")
        c4.metric("Bond Allocation",
                  f"${combined_result['bond_total']:,.0f}",
                  f"{combined_result['bond_total'] / combined_result['portfolio_total']:.0%}")
        c5.metric("Equity-Bond Correlation",
                  f"{combined_result['eq_bond_correlation']:.2f}",
                  "Diversification" if combined_result['eq_bond_correlation'] < 0 else "Co-movement")

    elif has_equities and var_result:
        c1, c2, c3, c4, c5 = st.columns(5)
        var_amount = var_result['var'] * total_equity_investment
        es_amount = var_result['es'] * total_equity_investment
        c1.metric(f"1-Day {confidence:.0%} VaR", f"{var_result['var']:.2%}",
                  f"{currency}{var_amount:,.0f}")
        c2.metric(f"1-Day {confidence:.0%} ES", f"{var_result['es']:.2%}",
                  f"{currency}{es_amount:,.0f}")
        portfolio_vol = np.sqrt(
            np.array(weights) @ (np.outer(var_result['forecast_vols'], var_result['forecast_vols'])
                                 * returns.corr().values) @ np.array(weights))
        annual_vol = portfolio_vol * np.sqrt(252)
        c3.metric("Portfolio Volatility", f"{annual_vol:.1%}", "Annualized")
        c4.metric("Sharpe Ratio", f"{sharpe_result['sharpe_ratio']:.2f}",
                  f"Return: {sharpe_result['portfolio_annual_return']:.1%}")
        avg_persistence = np.mean([risk_engine.get_garch_summary(t)['persistence'] for t in tickers])
        avg_half_life = np.mean([risk_engine.get_garch_summary(t)['half_life'] for t in tickers])
        c5.metric("Avg Persistence", f"{avg_persistence:.2f}",
                  f"~{avg_half_life:.0f} day half-life")

    elif has_bonds and bond_risk_result:
        bond_total = sum(b['investment'] for b in bonds)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Bond Portfolio {confidence:.0%} VaR",
                  f"${bond_risk_result['var']:,.0f}",
                  f"{bond_risk_result['var'] / bond_total:.2%} of portfolio")
        c2.metric(f"Bond Portfolio {confidence:.0%} ES",
                  f"${bond_risk_result['es']:,.0f}",
                  f"{bond_risk_result['es'] / bond_total:.2%} of portfolio")
        c3.metric("Yield Forecast Vol",
                  f"{bond_risk_result['forecast_yield_vol_bps']:.1f} bps/day",
                  "GARCH(1,1) estimate")

    st.markdown("---")

    # -- Build tabs --
    tab_names = []
    if has_equities:
        tab_names += ["📈 Volatility", "📊 Return Distribution", "📋 Stock Details", "📑 Fundamentals"]
    if has_bonds:
        tab_names += ["🏦 Yield Curve", "💰 Bond Analytics", "⚡ Scenario Analysis"]
    if has_equities and has_bonds and combined_result:
        tab_names += ["🔗 Multi-Asset Risk"]

    if not tab_names:
        st.warning("No equities or bonds to analyse.")
        st.stop()

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # ---- EQUITY TABS ----
    if has_equities:

        # Volatility
        with tabs[tab_idx]:
            st.subheader("Historical Volatility (GARCH Estimates)")
            fig = go.Figure()
            for ticker in tickers:
                summary = risk_engine.get_garch_summary(ticker)
                cond_vol = summary['conditional_volatility'] * np.sqrt(252)
                display_name = ticker.replace(".NS", "")
                fig.add_trace(go.Scatter(x=returns.index, y=cond_vol, name=display_name, mode='lines'))
            fig.update_layout(yaxis_title="Annualized Volatility", xaxis_title="Date",
                              hovermode="x unified",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02), height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info("**Reading this chart:** Spikes show periods of market stress. "
                    "Notice how volatility clusters — turbulent periods persist for weeks.")
        tab_idx += 1

        # Return Distribution
        with tabs[tab_idx]:
            st.subheader("Simulated Portfolio Returns")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=var_result['simulated_returns'], nbinsx=100,
                                       name="Simulated Returns", marker_color='steelblue', opacity=0.7))
            fig.add_vline(x=-var_result['var'], line_dash="dash", line_color="red",
                          annotation_text=f"VaR ({confidence:.0%})", annotation_position="top left")
            fig.add_vline(x=-var_result['es'], line_dash="dash", line_color="darkred",
                          annotation_text=f"ES ({confidence:.0%})", annotation_position="top left")
            fig.update_layout(xaxis_title="Daily Return", yaxis_title="Frequency",
                              showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"**Reading this chart:** The red dashed line is VaR — on "
                    f"{(1 - confidence) * 100:.0f}% of days, losses exceed this. "
                    f"The dark red line is ES — the average loss on those bad days.")
        tab_idx += 1

        # Stock Details
        with tabs[tab_idx]:
            st.subheader("Individual Stock Analysis")
            for ticker in tickers:
                summary = risk_engine.get_garch_summary(ticker)
                display_name = ticker.replace(".NS", "")
                with st.expander(f"**{display_name}**", expanded=True):
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Forecast Vol (Annual)", f"{summary['forecast_vol_annual']:.1%}")
                    cc2.metric("Alpha (Reactivity)", f"{summary['alpha']:.3f}")
                    cc3.metric("Beta (Memory)", f"{summary['beta']:.3f}")
                    cc4.metric("Persistence", f"{summary['persistence']:.3f}")
                    if summary['persistence'] > 0.95:
                        st.warning(f"High persistence ({summary['persistence']:.2f}). "
                                   f"Shocks take ~{summary['half_life']:.0f} days to halve.")
                    elif summary['persistence'] < 0.85:
                        st.success(f"Low persistence ({summary['persistence']:.2f}). "
                                   f"Market calms within ~{summary['half_life']:.0f} days.")
        tab_idx += 1

        # Fundamentals
        with tabs[tab_idx]:
            st.subheader("Fundamental Analysis")
            st.caption("Quarterly financial data from Yahoo Finance.")
            for ticker in tickers:
                fund = fundamentals_engine.data.get(ticker, {})
                display_name = ticker.replace('.NS', '')
                with st.expander(f"**{display_name}**", expanded=True):
                    if not fund.get('available', False):
                        st.warning(f"Fundamental data not available for {display_name}.")
                        continue
                    fin_currency = fund['financial_currency']
                    stock_currency = fund['stock_currency']
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("Market Cap",
                               format_financial_number(fund['market_cap'], stock_currency)
                               if fund['market_cap'] else "N/A")
                    fc2.metric("Trailing P/E",
                               f"{fund['trailing_pe']:.1f}" if fund['trailing_pe'] else "N/A")
                    fc3.metric("Forward P/E",
                               f"{fund['forward_pe']:.1f}" if fund['forward_pe'] else "N/A")
                    if fund['quarterly_pat'] is not None:
                        pat = fund['quarterly_pat']
                        yoy = compute_yoy_growth(pat)
                        st.markdown("**Quarterly Net Income**")
                        rows = []
                        for date in pat.index.sort_values(ascending=False):
                            growth = yoy.get(date)
                            rows.append({
                                'Quarter': date.strftime('%b %Y'),
                                f'Net Income ({fin_currency})': format_financial_number(pat[date],
                                                                                       fin_currency),
                                'YoY Growth': f"{growth:+.1f}%" if growth is not None else "N/A"
                            })
                        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        tab_idx += 1

    # ---- FIXED INCOME TABS ----
    if has_bonds:

        # Yield Curve
        with tabs[tab_idx]:
            st.subheader("US Treasury Yield Curve")

            col_curve1, col_curve2 = st.columns([2, 1])

            with col_curve1:
                # Par yield curve
                fine_tenors = np.linspace(0.25, 30, 200)
                fine_par = np.interp(fine_tenors, curve_engine.tenors, curve_engine.par_yields * 100)
                fine_zero = [curve_engine.interpolate_zero(t) * 100 for t in fine_tenors]

                fig_curve = go.Figure()
                fig_curve.add_trace(go.Scatter(
                    x=fine_tenors, y=fine_par, name='Par Yield', mode='lines',
                    line=dict(color='#2962FF', width=2)))
                fig_curve.add_trace(go.Scatter(
                    x=fine_tenors, y=fine_zero, name='Zero Rate (Bootstrapped)', mode='lines',
                    line=dict(color='#FF6D00', width=2, dash='dot')))
                fig_curve.add_trace(go.Scatter(
                    x=curve_engine.tenors, y=curve_engine.par_yields * 100,
                    name='Observed Par Yields', mode='markers',
                    marker=dict(color='#2962FF', size=8)))

                # Mark bond maturities on the curve
                for bond in bonds:
                    z = curve_engine.interpolate_zero(bond['maturity'])
                    fig_curve.add_trace(go.Scatter(
                        x=[bond['maturity']], y=[z * 100],
                        name=f"{bond['label']} ({bond['maturity']}Y)",
                        mode='markers', marker=dict(size=12, symbol='diamond', color='red')))

                fig_curve.update_layout(
                    xaxis_title="Maturity (Years)", yaxis_title="Yield (%)",
                    hovermode="x unified", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_curve, use_container_width=True)

            with col_curve2:
                st.markdown("**Curve Data**")
                curve_df = pd.DataFrame({
                    'Tenor': curve_engine.tenors,
                    'Par Yield (%)': curve_engine.par_yields * 100,
                    'Zero Rate (%)': curve_engine.zero_rates * 100,
                    'Discount Factor': curve_engine.discount_factors
                })
                curve_df['Tenor'] = curve_df['Tenor'].apply(lambda x: f"{x:.1f}Y")
                st.dataframe(curve_df, hide_index=True, use_container_width=True)

            st.info("**Par yield** is the coupon rate at which a bond prices at par. "
                    "**Zero rate** is the yield on a zero-coupon bond, bootstrapped from par yields. "
                    "The spread between them reflects the coupon reinvestment effect. "
                    "Red diamonds mark your bond maturities on the curve.")
        tab_idx += 1

        # Bond Analytics
        with tabs[tab_idx]:
            st.subheader("Bond Portfolio Analytics")

            for i, bond in enumerate(bonds):
                analytics = bond_engine.compute_analytics(
                    bond['coupon_rate'], bond['maturity'], bond['face_value'], bond['frequency'])
                if analytics is None:
                    st.warning(f"Could not price {bond['label']}")
                    continue

                with st.expander(f"**{bond['label']}** — {bond['coupon_rate'] * 100:.2f}% coupon, "
                                 f"{bond['maturity']:.1f}Y maturity", expanded=True):
                    bc1, bc2, bc3, bc4, bc5, bc6 = st.columns(6)
                    bc1.metric("Price", f"${analytics['price']:.4f}")
                    bc2.metric("YTM", f"{analytics['ytm'] * 100:.3f}%")
                    bc3.metric("Macaulay Dur.", f"{analytics['macaulay_duration']:.3f}Y")
                    bc4.metric("Modified Dur.", f"{analytics['modified_duration']:.3f}")
                    bc5.metric("Convexity", f"{analytics['convexity']:.2f}")
                    bc6.metric("DV01 (per $100)", f"${analytics['dv01']:.4f}")

                    # Position info
                    n_bonds_held = bond['investment'] / (analytics['price'] / bond['face_value'] * bond['face_value'])
                    position_dv01 = analytics['dv01'] * n_bonds_held * (bond['face_value'] / 100)

                    st.caption(f"Position: ${bond['investment']:,.0f} invested "
                               f"({n_bonds_held:,.1f} bonds at ${analytics['price']:.2f} per "
                               f"${bond['face_value']:.0f} face) | "
                               f"Position DV01: ${position_dv01:,.2f}")

                    # Price-Yield chart
                    yields_range, prices_range = bond_engine.price_yield_curve(
                        bond['coupon_rate'], bond['maturity'], bond['face_value'], bond['frequency'])

                    fig_py = go.Figure()
                    fig_py.add_trace(go.Scatter(
                        x=[y * 100 for y in yields_range], y=prices_range,
                        mode='lines', name='Price-Yield',
                        line=dict(color='#2962FF', width=2)))
                    fig_py.add_trace(go.Scatter(
                        x=[analytics['ytm'] * 100], y=[analytics['price']],
                        mode='markers', name='Current',
                        marker=dict(color='red', size=10)))
                    fig_py.update_layout(
                        xaxis_title="Yield (%)", yaxis_title=f"Price (per ${bond['face_value']:.0f} face)",
                        height=300, showlegend=False)
                    st.plotly_chart(fig_py, use_container_width=True)
                    st.caption("The curvature of this line IS convexity. A bond with higher "
                               "convexity gains more when rates fall than it loses when rates rise "
                               "by the same amount.")

            # Portfolio-level summary
            if len(bonds) > 1:
                st.markdown("---")
                st.markdown("**Portfolio Summary**")
                total_bond_investment = sum(b['investment'] for b in bonds)
                port_dur = 0
                port_dv01 = 0
                for bond in bonds:
                    a = bond_engine.compute_analytics(
                        bond['coupon_rate'], bond['maturity'], bond['face_value'], bond['frequency'])
                    if a:
                        w = bond['investment'] / total_bond_investment
                        port_dur += w * a['modified_duration']
                        n_b = bond['investment'] / (a['price'] / bond['face_value'] * bond['face_value'])
                        port_dv01 += a['dv01'] * n_b * (bond['face_value'] / 100)
                ps1, ps2, ps3 = st.columns(3)
                ps1.metric("Portfolio Modified Duration", f"{port_dur:.3f}")
                ps2.metric("Portfolio DV01", f"${port_dv01:,.2f}")
                ps3.metric("Total Bond Investment", f"${total_bond_investment:,.0f}")
        tab_idx += 1

        # Scenario Analysis
        with tabs[tab_idx]:
            st.subheader("Interest Rate Scenario Analysis")

            shifts = [-200, -100, -50, -25, 0, 25, 50, 100, 200]

            # Aggregate portfolio scenario
            total_pnl_by_shift = {s: 0.0 for s in shifts}
            total_pnl_pct_by_shift = {s: 0.0 for s in shifts}
            total_base_value = 0.0

            for bond in bonds:
                result = bond_engine.scenario_analysis(
                    bond['coupon_rate'], bond['maturity'], bond['face_value'],
                    bond['frequency'], shifts)
                if result is None:
                    continue
                base_price = result['base']['price']
                n_bonds_held = bond['investment'] / (base_price / bond['face_value'] * bond['face_value'])
                total_base_value += n_bonds_held * base_price
                for s in result['scenarios']:
                    total_pnl_by_shift[s['shift_bps']] += s['pnl_full'] * n_bonds_held

            if total_base_value > 0:
                for s in shifts:
                    total_pnl_pct_by_shift[s] = total_pnl_by_shift[s] / total_base_value * 100

            # P&L waterfall chart
            colors = ['#2E7D32' if total_pnl_by_shift[s] >= 0 else '#C62828' for s in shifts]
            fig_scenario = go.Figure()
            fig_scenario.add_trace(go.Bar(
                x=[f"{s:+d}bps" for s in shifts],
                y=[total_pnl_by_shift[s] for s in shifts],
                marker_color=colors,
                text=[f"${total_pnl_by_shift[s]:+,.0f}" for s in shifts],
                textposition='outside'))
            fig_scenario.update_layout(
                xaxis_title="Yield Curve Shift", yaxis_title="Portfolio P&L ($)",
                height=400, showlegend=False)
            st.plotly_chart(fig_scenario, use_container_width=True)

            # Duration approximation comparison
            st.markdown("**Full Repricing vs Duration Approximation**")
            st.caption("This table shows how well the duration-convexity approximation "
                       "(ΔP ≈ -Dur × P × Δy + ½ × Conv × P × Δy²) matches full repricing. "
                       "The error grows for larger rate moves, demonstrating why convexity matters.")

            comparison_rows = []
            for bond in bonds:
                result = bond_engine.scenario_analysis(
                    bond['coupon_rate'], bond['maturity'], bond['face_value'],
                    bond['frequency'], shifts)
                if result is None:
                    continue
                for s in result['scenarios']:
                    comparison_rows.append({
                        'Bond': bond['label'],
                        'Shift (bps)': f"{s['shift_bps']:+d}",
                        'Full Repricing': f"${s['pnl_full']:+.4f}",
                        'Dur-Conv Approx': f"${s['pnl_duration_approx']:+.4f}",
                        'Approx Error': f"${s['approx_error']:+.6f}"
                    })
            if comparison_rows:
                st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

            # Shifted curve visualization
            st.markdown("---")
            st.markdown("**Shifted Yield Curves**")
            fig_shifted = go.Figure()
            fine_tenors = np.linspace(0.25, 30, 100)
            base_zero = [curve_engine.interpolate_zero(t) * 100 for t in fine_tenors]
            fig_shifted.add_trace(go.Scatter(x=fine_tenors, y=base_zero,
                                             name='Base Curve', mode='lines',
                                             line=dict(color='black', width=2)))
            shift_colors = {-100: '#1565C0', -50: '#42A5F5', 50: '#EF5350', 100: '#B71C1C'}
            for shift_bps, color in shift_colors.items():
                shifted = curve_engine.get_shifted_curve(shift_bps)
                shifted_zero = [shifted.interpolate_zero(t) * 100 for t in fine_tenors]
                fig_shifted.add_trace(go.Scatter(x=fine_tenors, y=shifted_zero,
                                                  name=f"{shift_bps:+d}bps",
                                                  mode='lines', line=dict(color=color, dash='dash')))
            fig_shifted.update_layout(
                xaxis_title="Maturity (Years)", yaxis_title="Zero Rate (%)",
                height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_shifted, use_container_width=True)
        tab_idx += 1

    # ---- MULTI-ASSET TAB ----
    if has_equities and has_bonds and combined_result:
        with tabs[tab_idx]:
            st.subheader("Multi-Asset Risk Decomposition")

            # P&L distribution comparison
            fig_multi = go.Figure()
            fig_multi.add_trace(go.Histogram(
                x=combined_result['eq_pnl'], nbinsx=80, name='Equity P&L',
                marker_color='steelblue', opacity=0.5))
            fig_multi.add_trace(go.Histogram(
                x=combined_result['bond_pnl'], nbinsx=80, name='Bond P&L',
                marker_color='#FF6D00', opacity=0.5))
            fig_multi.add_trace(go.Histogram(
                x=combined_result['total_pnl'], nbinsx=80, name='Combined P&L',
                marker_color='#2E7D32', opacity=0.5))
            fig_multi.add_vline(x=-combined_result['var'], line_dash="dash", line_color="red",
                                annotation_text=f"Combined VaR ({confidence:.0%})")
            fig_multi.update_layout(
                xaxis_title="Daily P&L ($)", yaxis_title="Frequency",
                barmode='overlay', height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig_multi, use_container_width=True)

            # Diversification benefit
            eq_var = -np.percentile(combined_result['eq_pnl'], (1 - confidence) * 100)
            bond_var = -np.percentile(combined_result['bond_pnl'], (1 - confidence) * 100)
            undiversified = eq_var + bond_var
            diversified = combined_result['var']
            div_benefit = undiversified - diversified

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Equity VaR", f"${eq_var:,.0f}")
            mc2.metric("Bond VaR", f"${bond_var:,.0f}")
            mc3.metric("Undiversified VaR", f"${undiversified:,.0f}",
                       "Sum of parts")
            mc4.metric("Diversification Benefit", f"${div_benefit:,.0f}",
                       f"{div_benefit / undiversified:.1%} reduction" if undiversified > 0 else "")

            st.info(f"**Diversification at work.** Because equity returns and yield changes have "
                    f"a correlation of {combined_result['eq_bond_correlation']:.2f}, the combined "
                    f"portfolio VaR (${diversified:,.0f}) is less than the simple sum of equity "
                    f"and bond VaR (${undiversified:,.0f}). The ${div_benefit:,.0f} reduction "
                    f"is the diversification benefit of holding both asset classes.")

            # Correlation matrix
            if returns is not None and yield_history is not None:
                st.markdown("**Cross-Asset Correlation Matrix**")
                eq_rets = returns.copy()
                yc = yield_history.reindex(eq_rets.index).diff().dropna()
                combined_data = pd.concat([eq_rets, yc.rename('10Y Yield Chg')], axis=1).dropna()
                corr_matrix = combined_data.corr()
                display_names = [t.replace('.NS', '') for t in corr_matrix.columns[:-1]] + ['10Y Yield Chg']
                corr_matrix.columns = display_names
                corr_matrix.index = display_names

                fig_corr = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',
                                     zmin=-1, zmax=1, aspect='auto')
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption("Negative correlation between equities and yield changes means that "
                           "when stocks fall, bonds may rally (flight to quality), providing a "
                           "natural hedge.")
        tab_idx += 1

    # ---- WHAT THIS MEANS ----
    st.markdown("---")
    st.header("What This Means")

    if has_equities and has_bonds and combined_result:
        st.markdown(f"""
        Based on your **${combined_result['portfolio_total']:,.0f}** multi-asset portfolio
        (${combined_result['equity_total']:,.0f} equities + ${combined_result['bond_total']:,.0f} bonds):

        - **On {(1 - confidence) * 100:.0f}% of trading days**, your combined portfolio could lose more than
          **${combined_result['var']:,.0f}**
        - **When those bad days happen**, the average loss would be around **${combined_result['es']:,.0f}**
        - **Diversification saves you ${div_benefit:,.0f}** compared to holding equities and bonds
          as separate, uncorrelated portfolios
        - **Equity-bond correlation is {combined_result['eq_bond_correlation']:.2f}** — {"negative correlation provides a natural hedge" if combined_result['eq_bond_correlation'] < 0 else "positive correlation means both asset classes tend to move together"}
        """)
    elif has_equities and var_result:
        worst_day_loss = var_result['var'] * total_equity_investment
        avg_bad_day_loss = var_result['es'] * total_equity_investment
        avg_persistence = np.mean([risk_engine.get_garch_summary(t)['persistence'] for t in tickers])
        avg_half_life = np.mean([risk_engine.get_garch_summary(t)['half_life'] for t in tickers])
        sharpe_ratio = sharpe_result['sharpe_ratio']
        excess_return = sharpe_result['excess_return']
        st.markdown(f"""
        Based on current market conditions and your {currency}{total_equity_investment:,.0f} portfolio:

        - **On {(1 - confidence) * 100:.0f}% of trading days** (roughly {int((1 - confidence) * 252)} days/year),
          you could lose more than **{currency}{worst_day_loss:,.0f}**
        - **When those bad days happen**, the average loss would be around
          **{currency}{avg_bad_day_loss:,.0f}**
        - **Volatility persistence is {avg_persistence:.2f}** — when markets get turbulent, expect it
          to last ~{avg_half_life:.0f} trading days
        - **Your Sharpe Ratio of {sharpe_ratio:.2f}** means you earn {excess_return:.2%} above the
          risk-free rate per unit of risk
        """)
    elif has_bonds and bond_risk_result:
        bond_total = sum(b['investment'] for b in bonds)
        st.markdown(f"""
        Based on your **${bond_total:,.0f}** bond portfolio:

        - **On {(1 - confidence) * 100:.0f}% of trading days**, your bond portfolio could lose more than
          **${bond_risk_result['var']:,.0f}**
        - **When those bad days happen**, the average loss would be around
          **${bond_risk_result['es']:,.0f}**
        - **GARCH-forecasted yield volatility is {bond_risk_result['forecast_yield_vol_bps']:.1f} bps/day**,
          which drives the VaR computation through duration-weighted price sensitivity
        """)

else:
    # ---- LANDING PAGE ----
    st.markdown("---")
    st.markdown("""
    ### How to Use

    1. **Enter equity symbols** and investment amounts in the sidebar
    2. **Toggle "Include Bonds"** to add fixed income positions
    3. **Click "Run Analysis"** to compute risk metrics across your portfolio

    ### What You'll Get

    **Equities:**
    - Value at Risk (VaR) and Expected Shortfall via GARCH + Monte Carlo
    - Sharpe Ratio and volatility persistence analysis
    - Individual stock GARCH decomposition and fundamental data

    **Fixed Income:**
    - Live US Treasury yield curve with zero-rate bootstrapping
    - Bond pricing, duration (Macaulay and Modified), convexity, and DV01
    - Interest rate scenario analysis with full repricing vs duration approximation
    - Price-yield relationship visualisation

    **Multi-Asset:**
    - Combined VaR across equities and bonds with cross-asset correlation
    - Diversification benefit quantification
    - Cross-asset correlation matrix (equity returns vs yield changes)

    ---

    *Built with GARCH(1,1) volatility modelling, yield curve bootstrapping, and Monte Carlo simulation.*
    """)
