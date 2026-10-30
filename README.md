# Markowitz Portfolio Optimiser

A Streamlit web app that applies Markowitz modern portfolio theory to any list of up to 20 stocks or ETFs. It downloads price history from Yahoo Finance, then finds the long-only weights with the highest risk-adjusted return and plots them against the efficient frontier.

## Running the app

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run portfolio-app.py
```

1. Type a ticker or company name and choose a search result, or type an exact ticker.
2. Press **Add**. The ticker is checked against Yahoo Finance, and unknown tickers show "Ticker not found".
3. Add at least two tickers, up to a maximum of 20. Click a ticker in the list to remove it.
4. Press **Run** to see the chart and the table of optimal weights.

## Project files

| File | Purpose |
|---|---|
| `portfolio-model.py` | All data, optimisation and chart functions. It does not depend on Streamlit. |
| `portfolio-app.py` | The Streamlit containers, session state and caching. |
| `requirements.txt` | The package versions the app was tested with. |
| `Quantitative_Finance_Algorithms.ipynb` | The original prototype that the model is based on. |

The hyphen in `portfolio-model.py` means Python's `import` statement can't load it, so the app uses `importlib.import_module('portfolio-model')`.

## From Yahoo Finance data to returns

The app uses [yfinance](https://github.com/ranaroussi/yfinance), an open-source Python library that reads Yahoo Finance's public endpoints. It is not an official Yahoo product, so requests can occasionally be rate limited or change behaviour.

| Purpose | Call | Notes |
|---|---|---|
| Search as you type | `yf.Search(query)` | Results are limited to equities and ETFs. |
| Check a ticker exists | `yf.Ticker(t).history(period='5d')` | An unknown ticker returns an empty table, not an error. |
| Risk-free rate | `yf.Ticker('^FVX').history(period='5d')` | The latest close of the US 5 year treasury yield. |
| Price history | `yf.Ticker(t).history(start, end)` | Daily data from 1 January 2015 to the most recent full trading day. |

The prices then go through these transformations:

1. **Adjusted close.** `history()` adjusts closing prices for splits and dividends by default, so returns reflect total return rather than price change alone.
2. **Date alignment.** Timezones are removed so that the same calendar day lines up across exchanges. Only days on which every ticker traded are kept. One recently listed ticker therefore shortens the window for all of them, and the app displays the window it used.
3. **Daily log returns.** For price $P_t$ on day $t$:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

4. **Annualisation.** With 252 trading days in a year, and $\bar{r}_i$ the mean daily return of asset $i$:

$$\mu_i = 252 \, \bar{r}_i \qquad \Sigma = 252 \, \mathrm{Cov}(r)$$

5. **Risk-free rate.** `^FVX` is quoted in percent, so a quote of 4.75 is divided by 100 to give 0.0475.

## The Markowitz model

Harry Markowitz's 1952 insight was that an asset's risk should be judged by what it does to the whole portfolio. Assets that don't move together partly cancel out each other's swings, so a portfolio can be less volatile than its individual holdings.

For a vector of portfolio weights $w$:

$$\text{Expected return: } \mu_p = w^\top \mu \qquad\qquad \text{Volatility: } \sigma_p = \sqrt{w^\top \Sigma \, w}$$

Return is a simple weighted average, but volatility depends on the covariances between assets. That is where the diversification benefit comes from.

This app applies two constraints: the weights sum to 1, and each weight is between 0 and 1. That means the portfolio is fully invested with no short selling or leverage.

## The efficient frontier

A portfolio is **efficient** if no other portfolio offers a higher return for the same risk. The efficient frontier is the curve formed by all such portfolios, and the chart draws it as a black line.

To trace it, the app solves one optimisation for each of 50 target returns $\mu^*$, spaced from the minimum variance portfolio's return up to the best single asset's return:

$$\min_w \; w^\top \Sigma \, w \quad \text{subject to} \quad w^\top \mu = \mu^*, \quad \sum_i w_i = 1, \quad 0 \le w_i \le 1$$

The coloured dots are 10,000 portfolios with random weights, coloured by Sharpe ratio. All of them sit on or below the frontier. With many tickers the cloud bunches up well below the curve, because random weights tend to average out into similar, evenly spread portfolios. The optimiser doesn't use these dots. They are there to show the range of possible outcomes.

## The optimal portfolio and the tangent line

The **Sharpe ratio** measures the return earned above the risk-free rate for each unit of risk:

$$S = \frac{\mu_p - r_f}{\sigma_p}$$

The app finds the weights that maximise $S$ using SciPy's SLSQP solver. This is the **tangency portfolio**, shown as a green star.

A straight line drawn from the risk-free rate on the y-axis has a slope equal to the Sharpe ratio of the portfolio it passes through. The steepest line that still reaches the frontier just touches it at the tangency portfolio. This is the **capital market line**, shown as a dashed red line:

$$\mu = r_f + S_{\text{optimal}} \, \sigma$$

Any point on that line can be reached by combining the tangency portfolio with the risk-free asset, and those combinations beat everything else on the frontier apart from the tangency point itself.

## CAPM

The Capital Asset Pricing Model (Sharpe, Lintner and Mossin, in the 1960s) builds on Markowitz by asking what happens if every investor behaves this way. If everyone holds the same expected returns and covariances, everyone holds the same tangency portfolio, so it must be the market portfolio. An asset's expected return then depends only on its exposure to the market:

$$E[R_i] = r_f + \beta_i \left( E[R_m] - r_f \right) \qquad \text{where} \quad \beta_i = \frac{\mathrm{Cov}(R_i, R_m)}{\mathrm{Var}(R_m)}$$

- $r_f$ is the compensation for time, earned with no risk.
- $E[R_m] - r_f$ is the market risk premium.
- $\beta_i$ measures how strongly the asset moves with the market.

CAPM's implication is that only market risk is rewarded, since diversification can remove the rest. This app doesn't estimate betas, but it uses the same building blocks: the risk-free rate and the capital market line. `CAPM.py` in this repository contains a separate beta calculation.

## Limitations

- **The past is not a forecast.** Expected returns are historical averages, which are noisy and don't reliably predict future returns. The optimiser also favours whichever assets happened to do well, so results are best treated as a description of the sample period and not as investment advice.
- **Results depend on the window.** A short shared history, such as from a recent IPO, can dominate the result. The app warns when the window is under a year and refuses to run below 60 trading days.
- **Currencies are not converted.** Each ticker's returns are in its own trading currency, so mixing exchanges such as `AAPL` and `VOD.L` ignores exchange rate effects.
- **Some simplifications are approximate.** Portfolio returns are calculated as a weighted sum of log returns, as in the prototype, although log returns only add exactly across time and not across assets. The treasury yield is also compared to log returns without conversion. Both effects are small at typical return levels.
- **No costs.** Trading costs and taxes are ignored.
