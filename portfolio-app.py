# Streamlit interface for the Markowitz model - run with: streamlit run portfolio-app.py
# All of the calculations live in portfolio-model.py

import importlib

import streamlit as st
from streamlit_searchbox import st_searchbox

# the hyphen in the file name rules out a normal "import portfolio-model" statement
model = importlib.import_module('portfolio-model')

CACHE_SECONDS = 3600
NUM_TICKER_COLUMNS = 5


# ---------------------------------------------------------------------------
# Cached calls to Yahoo Finance
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def search_tickers(query):
    return model.search_tickers(query)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_risk_free_rate():
    return model.get_risk_free_rate()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def run_model(tickers, risk_free_rate):
    return model.run_model(tickers, risk_free_rate)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    st.session_state.setdefault('tickers', [])
    st.session_state.setdefault('result', None)
    # the search box only empties when it is recreated with a new key
    st.session_state.setdefault('search_box_id', 0)


def add_ticker(raw_ticker):
    # returns an error message, or None if the ticker was added
    ticker = model.clean_ticker(raw_ticker or '')
    if not ticker:
        return "Type a ticker or choose one from the search results first."
    if ticker in st.session_state.tickers:
        return f"{ticker} is already in the list."
    if len(st.session_state.tickers) >= model.MAX_TICKERS:
        return f"The list is limited to {model.MAX_TICKERS} tickers."

    try:
        found = model.validate_ticker(ticker)
    except Exception:
        return "Could not reach Yahoo Finance. Please try again."
    if not found:
        return f"Ticker not found: {ticker}"

    st.session_state.tickers.append(ticker)
    st.session_state.result = None
    st.session_state.search_box_id += 1
    return None


def remove_ticker(ticker):
    st.session_state.tickers.remove(ticker)
    st.session_state.result = None


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

def show_risk_free_rate():
    # returns the rate, or None if it could not be downloaded
    with st.container(border=True):
        try:
            risk_free_rate = get_risk_free_rate()
        except Exception:
            st.markdown("**Risk free rate (r_f)**")
            st.error("Could not download the US 5 year treasury rate from Yahoo Finance.")
            return None
        st.metric("Risk free rate (r_f)", f"{risk_free_rate:.2%}",
                  help=f"Latest US 5 year treasury yield, Yahoo ticker {model.RISK_FREE_TICKER}")
    return risk_free_rate


def show_ticker_input():
    with st.container(border=True):
        st.markdown("**Add a stock or ETF**")
        # top aligned so that the button does not move when the search results open
        search_col, button_col = st.columns([5, 1], vertical_alignment='top')
        with search_col:
            selection = st_searchbox(
                search_tickers,
                placeholder="Search by ticker or company name, e.g. AAPL or Apple",
                key=f"ticker_search_{st.session_state.search_box_id}",
                default_use_searchterm=True,
            )
        with button_col:
            add_clicked = st.button("Add", width='stretch')

        if add_clicked:
            error = add_ticker(selection)
            if error:
                st.error(error)
            else:
                st.rerun()


def show_ticker_list():
    with st.container(border=True):
        tickers = st.session_state.tickers
        st.markdown(f"**Tickers ({len(tickers)}/{model.MAX_TICKERS})**")
        if not tickers:
            st.caption("No tickers added yet. Add at least two to run the model.")

        columns = st.columns(NUM_TICKER_COLUMNS)
        for i, ticker in enumerate(tickers):
            columns[i % NUM_TICKER_COLUMNS].button(
                f"{ticker} ✕", key=f"remove_{ticker}", help=f"Remove {ticker}", width='stretch',
                on_click=remove_ticker, args=(ticker,),
            )


def show_run_button(risk_free_rate):
    tickers = st.session_state.tickers
    can_run = len(tickers) >= 2 and risk_free_rate is not None
    if st.button("Run", type='primary', disabled=not can_run):
        with st.spinner("Downloading prices and optimising..."):
            try:
                st.session_state.result = run_model(tuple(tickers), risk_free_rate)
            except ValueError as error:
                st.error(str(error))
            except Exception:
                st.error("Could not download the price history from Yahoo Finance. Please try again.")


def show_results(result):
    st.divider()
    st.subheader("Optimal portfolio")

    if not result['beats_risk_free']:
        st.warning(
            "None of these assets has a historical return above the risk free rate, so no mix of them "
            "beats simply holding treasuries. The weights below are shown for reference only."
        )
    st.caption(
        f"Based on {result['num_days']:,} trading days from {result['start_date']} to {result['end_date']}, "
        "the period shared by every ticker."
    )
    if result['num_days'] < model.NUM_TRADING_DAYS:
        st.warning("These tickers share less than one year of history, so treat the results with caution.")

    return_col, risk_col, sharpe_col = st.columns(3)
    return_col.metric("Expected return", f"{result['optimal_return']:.2%}")
    risk_col.metric("Expected volatility", f"{result['optimal_risk']:.2%}")
    sharpe_col.metric("Sharpe ratio", f"{result['optimal_sharpe']:.2f}")

    st.pyplot(model.show_optimal_portfolio(result))

    st.dataframe(
        model.weights_table(result),
        hide_index=True,
        width='stretch',
        column_config={'Optimal weight': st.column_config.NumberColumn(format='percent')},
    )


def main():
    st.set_page_config(page_title="Markowitz Portfolio Optimiser", page_icon="📈", layout='centered')
    init_state()

    st.title("Markowitz Portfolio Optimiser")
    st.caption("Find the mix of stocks and ETFs with the highest historical return per unit of risk.")

    risk_free_rate = show_risk_free_rate()
    show_ticker_input()
    show_ticker_list()
    show_run_button(risk_free_rate)

    if st.session_state.result is not None:
        show_results(st.session_state.result)


main()
