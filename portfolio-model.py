# Markowitz portfolio model - all of the calculations behind portfolio-app.py
# Nothing in this file depends on Streamlit, so it can also be used from a notebook

import logging

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from scipy import optimize

# yfinance logs an error for every unknown ticker, which is an expected event here
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

MAX_TICKERS = 20
RISK_FREE_TICKER = '^FVX'

# Historical data
START_DATE = '2015-01-01'
NUM_TRADING_DAYS = 252
# fewer common trading days than this gives meaningless means and covariances
MIN_OBSERVATIONS = 60

NUM_PORTFOLIOS = 10000
NUM_FRONTIER_POINTS = 50
# variances are small numbers, so the default tolerance would stop the solver too early
SOLVER_OPTIONS = {'ftol': 1e-10, 'maxiter': 1000}

# only these Yahoo security types are offered by the search box
SEARCH_QUOTE_TYPES = ('EQUITY', 'ETF')


# ---------------------------------------------------------------------------
# Yahoo Finance data
# ---------------------------------------------------------------------------

def clean_ticker(ticker):
    return ticker.strip().upper()


def search_tickers(query, max_results=8):
    # returns (label, symbol) pairs, the format expected by the app's search box
    if not query or not query.strip():
        return []
    try:
        quotes = yf.Search(query.strip(), max_results=max_results, news_count=0).quotes
    except Exception:
        # a failed search must not interrupt typing, the check on Add still runs
        return []

    results = []
    for quote in quotes:
        symbol = quote.get('symbol')
        if symbol and quote.get('quoteType') in SEARCH_QUOTE_TYPES:
            name = quote.get('shortname') or quote.get('longname') or ''
            results.append((f"{symbol} - {name} ({quote['quoteType']})", symbol))
    return results


def validate_ticker(ticker):
    # Yahoo returns an empty table (not an error) when a ticker does not exist
    return not yf.Ticker(clean_ticker(ticker)).history(period='5d').empty


def get_risk_free_rate():
    # ^FVX is quoted in percent (4.75 means 4.75%), the model needs a decimal (0.0475)
    closes = yf.Ticker(RISK_FREE_TICKER).history(period='5d')['Close'].dropna()
    if closes.empty:
        raise ValueError(f"No data returned for {RISK_FREE_TICKER}")
    return closes.iloc[-1] / 100


def download_data(tickers, start_date=START_DATE, end_date=None):
    end_date = end_date or str(date.today())
    stock_data = {}

    for stock in tickers:
        close = yf.Ticker(stock).history(start=start_date, end=end_date)['Close']
        if close.empty:
            raise ValueError(f"No price history returned for {stock}")
        # exchanges report in their own timezone, strip it so that the same
        # calendar day lines up across tickers (e.g. AAPL in New York, VOD.L in London)
        close.index = close.index.tz_localize(None).normalize()
        stock_data[stock] = close

    return pd.DataFrame(stock_data)


def calculate_return(data):
    # keep only the days on which every ticker traded, so that all means and
    # covariances are measured over the same window
    prices = data.dropna()
    # NORMALISATION - to measure all the variables in comparable metrics
    log_return = np.log(prices / prices.shift(1))
    return log_return[1:]  # removes first row as it is all NaN


def annualise(returns):
    # instead of daily metrics we are after annual metrics
    mean = returns.mean().to_numpy() * NUM_TRADING_DAYS
    cov = returns.cov().to_numpy() * NUM_TRADING_DAYS
    return mean, cov


# ---------------------------------------------------------------------------
# Markowitz optimisation
# ---------------------------------------------------------------------------

def statistics(weights, mean, cov, risk_free_rate):
    portfolio_return = np.sum(mean * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])


def generate_portfolios(mean, cov, num_portfolios=NUM_PORTFOLIOS):
    # Generating portfolios with random weights
    weights = np.random.random((num_portfolios, len(mean)))
    weights /= weights.sum(axis=1, keepdims=True)

    # the same formulas as statistics(), applied to every row at once
    portfolio_means = weights @ mean
    portfolio_risks = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov, weights))

    return portfolio_means, portfolio_risks


def weight_constraint(x):
    # what are the constraints? sum of the weights = 1, therefore sum w - 1 = 0
    return np.sum(x) - 1


def weight_bounds(num_assets):
    # long only - the weights can be 1 at the most when investing 100% of money into a stock
    return tuple((0, 1) for _ in range(num_assets))


def clean_weights(weights):
    # removes the tiny values the solver leaves behind (e.g. -1e-17)
    weights = np.clip(weights, 0, 1)
    weights[weights < 1e-6] = 0
    return weights / weights.sum()


def min_function_sharpe(weights, mean, cov, risk_free_rate):
    # scipy optimise module can find the minimum of a given function
    # the maximum of a f(x) is the minimum of -f(x)
    return -statistics(weights, mean, cov, risk_free_rate)[2]


def optimize_portfolio(mean, cov, risk_free_rate):
    # finds the tangency portfolio: the weights with the highest Sharpe ratio
    num_assets = len(mean)
    optimum = optimize.minimize(
        fun=min_function_sharpe,
        x0=np.full(num_assets, 1 / num_assets),
        args=(mean, cov, risk_free_rate),
        method='SLSQP',
        bounds=weight_bounds(num_assets),
        constraints={'type': 'eq', 'fun': weight_constraint},
        options=SOLVER_OPTIONS,
    )
    if not optimum.success:
        raise ValueError(f"Optimisation failed: {optimum.message}")
    return clean_weights(optimum.x)


def min_variance(mean, cov, target_return=None, x0=None):
    # the lowest risk portfolio, optionally for a required level of return
    num_assets = len(mean)
    constraints = [{'type': 'eq', 'fun': weight_constraint}]
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(mean * x) - target_return})

    return optimize.minimize(
        fun=lambda x: np.dot(x.T, np.dot(cov, x)),
        x0=np.full(num_assets, 1 / num_assets) if x0 is None else x0,
        method='SLSQP',
        bounds=weight_bounds(num_assets),
        constraints=constraints,
        options=SOLVER_OPTIONS,
    )


def efficient_frontier(mean, cov, num_points=NUM_FRONTIER_POINTS):
    # only returns above the minimum variance portfolio are efficient, the highest
    # return available to a long only investor is 100% in the best asset
    lowest_risk = min_variance(mean, cov)
    if not lowest_risk.success:
        raise ValueError(f"Efficient frontier failed: {lowest_risk.message}")

    frontier_means = []
    frontier_risks = []
    x0 = lowest_risk.x

    for target_return in np.linspace(np.sum(mean * lowest_risk.x), mean.max(), num_points):
        # each solution is a good starting guess for the next, slightly higher, target
        result = min_variance(mean, cov, target_return, x0)
        if result.success:
            x0 = result.x
            frontier_means.append(target_return)
            frontier_risks.append(np.sqrt(result.fun))

    return np.array(frontier_means), np.array(frontier_risks)


def run_model(tickers, risk_free_rate):
    # the complete pipeline: prices -> returns -> annual statistics -> optimal weights
    returns = calculate_return(download_data(tickers))
    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(
            f"These tickers only share {len(returns)} trading days of history, "
            f"at least {MIN_OBSERVATIONS} are needed. "
            "Remove the most recently listed ticker and try again."
        )

    mean, cov = annualise(returns)
    weights = optimize_portfolio(mean, cov, risk_free_rate)
    portfolio_means, portfolio_risks = generate_portfolios(mean, cov)
    frontier_means, frontier_risks = efficient_frontier(mean, cov)
    optimal_return, optimal_risk, optimal_sharpe = statistics(weights, mean, cov, risk_free_rate)

    return {
        'tickers': tickers,
        'risk_free_rate': risk_free_rate,
        'weights': weights,
        'optimal_return': optimal_return,
        'optimal_risk': optimal_risk,
        'optimal_sharpe': optimal_sharpe,
        'portfolio_means': portfolio_means,
        'portfolio_risks': portfolio_risks,
        'frontier_means': frontier_means,
        'frontier_risks': frontier_risks,
        'start_date': returns.index[0].date(),
        'end_date': returns.index[-1].date(),
        'num_days': len(returns),
        # no long only portfolio can beat the risk free rate in this case
        'beats_risk_free': bool(mean.max() > risk_free_rate),
    }


def weights_table(result):
    table = pd.DataFrame({'Ticker': result['tickers'], 'Optimal weight': result['weights']})
    return table.sort_values('Optimal weight', ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def show_optimal_portfolio(result):
    risk_free_rate = result['risk_free_rate']
    means = result['portfolio_means']
    risks = result['portfolio_risks']
    frontier_means = result['frontier_means']
    frontier_risks = result['frontier_risks']

    # a Figure object rather than plt.figure(), as pyplot's global state is not
    # safe when Streamlit serves several sessions at once
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()

    points = ax.scatter(risks, means, c=(means - risk_free_rate) / risks, marker='o', s=8, alpha=0.6, zorder=1)
    fig.colorbar(points, label='Sharpe Ratio')
    ax.plot(frontier_risks, frontier_means, color='black', linewidth=2, label='Efficient frontier', zorder=2)

    # the line must not stretch the axes, so fix the limits before it is drawn
    x_max = max(risks.max(), frontier_risks.max()) * 1.05
    all_y = np.concatenate([means, frontier_means, [risk_free_rate, result['optimal_return']]])
    y_padding = (all_y.max() - all_y.min()) * 0.08
    ax.set_xlim(0, x_max)
    ax.set_ylim(all_y.min() - y_padding, all_y.max() + y_padding)

    # capital market line: r_f + Sharpe * volatility
    ax.plot([0, x_max], [risk_free_rate, risk_free_rate + result['optimal_sharpe'] * x_max], color='tab:red',
            linestyle='--', linewidth=1.5, label='Tangent line (capital market line)', zorder=3)
    ax.plot(result['optimal_risk'], result['optimal_return'], 'g*', markersize=20, markeredgecolor='white',
            linestyle='none', label='Optimal portfolio', zorder=4)

    # label the risk free rate where the tangent line meets the y-axis
    ax.plot(0, risk_free_rate, 'o', color='tab:red', markersize=7, clip_on=False, zorder=5)
    ax.annotate(f'$r_f$ = {risk_free_rate:.2%}', xy=(0, risk_free_rate), xytext=(10, -14),
                textcoords='offset points', color='black', fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Expected Volatility')
    ax.set_ylabel('Expected Returns')
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc='upper left')

    return fig
