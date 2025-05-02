import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import time
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt  # New import for static charting

# Set page configuration
st.set_page_config(page_title="Strategy Backtest App")
st.title("Strategy Backtest App")


# ---------------------------
# Utility Function: Compute RSI
# ---------------------------
def compute_RSI(series, window=14):
    delta = series.diff()
    gain = delta.copy()
    loss = -delta.copy()
    gain[gain < 0] = 0
    loss[loss < 0] = 0

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.round(2)


# ---------------------------
# Data Download Function with Caching & Loading Indicator
# ---------------------------
@st.cache_data
def load_data(ticker, start_date, end_date):
    # create a session with a realistic User-Agent
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    })

    with st.spinner("Downloading data..."):
        # first attempt
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            session=session,
            progress=False
        )
        # retry once if empty
        if data.empty:
            time.sleep(1)
            data = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date,
                session=session,
                progress=False
            )

        # round and normalize
        df = data.round(2)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [col.split(',')[0] for col in df.columns]
        df.reset_index(inplace=True)

        mask = (
            (df['Date'] >= pd.to_datetime(start_date)) &
            (df['Date'] <= pd.to_datetime(end_date))
        )
        df = df.loc[mask]

        # if still empty, raise so we see why
        if df.empty:
            raise RuntimeError(
                f"yfinance returned no data for {ticker} "
                f"from {start_date} to {end_date}. "
                "Check your network, user-agent, or Yahoo’s endpoints."
            )

        return df


# ---------------------------
# Portfolio and Metrics Functions
# ---------------------------
def calculate_portfolio(df, initial_capital=0):
    df['Holdings'] = (df['Signal'] * df['Close']).round(2)
    df['Cash'] = (initial_capital - (df['Position'].fillna(0) * df['Close']).cumsum()).round(2)
    df['Portfolio'] = (df['Holdings'] + df['Cash']).round(2)
    return df


def calculate_trades(df):
    trades = []
    open_trade = None
    for _, row in df.iterrows():
        pos = row['Position']
        if pos == 1 and open_trade is None:
            open_trade = row
        elif pos == -1 and open_trade is not None:
            trades.append(row['Close'] - open_trade['Close'])
            open_trade = None
    return trades


def calculate_trade_log(df):
    trades = []
    open_trade = None
    for _, row in df.iterrows():
        pos = row['Position']
        if pos == 1 and open_trade is None:
            open_trade = row
        elif pos == -1 and open_trade is not None:
            trades.append({
                "date entry": open_trade['Date'],
                "price entry": open_trade['Close'],
                "date exit":    row['Date'],
                "price exit":   row['Close'],
                "Return":       round(row['Close'] - open_trade['Close'], 2)
            })
            open_trade = None
    return trades


def compute_metrics(trades):
    if trades:
        num_trades       = len(trades)
        points_mean      = np.mean(trades)
        total_gain_loss  = np.sum(trades)
        wins             = [t for t in trades if t > 0]
        percent_wins     = (len(wins) / num_trades) * 100
        total_wins       = np.sum(wins)
        losses           = [t for t in trades if t < 0]
        total_losses     = np.sum(losses)
        profit_factor    = total_wins / abs(total_losses) if total_losses != 0 else np.nan
        t_stat, t_pvalue = stats.ttest_1samp(trades, 0)
    else:
        num_trades = 0
        points_mean = total_gain_loss = profit_factor = t_stat = t_pvalue = np.nan
        percent_wins = 0
    return {
        "num_trades":       num_trades,
        "points_mean":      points_mean,
        "total_gain_loss":  total_gain_loss,
        "percent_wins":     percent_wins,
        "profit_factor":    profit_factor,
        "t_stat":           t_stat,
        "t_pvalue":         t_pvalue
    }


# ---------------------------
# Static Chart Plotting Function
# ---------------------------
def plot_equity_line_static(trades, ticker):
    cumulative_equity = np.cumsum(trades)
    trade_numbers     = list(range(1, len(cumulative_equity) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trade_numbers, cumulative_equity, linestyle='-', label='Equity Curve', color='blue')
    ax.set_title(f"{ticker} Equity Line")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative Profit (Points)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# ---------------------------
# Strategy Execution Function
# ---------------------------
def execute_strategy(df, strategy, params):
    # ... (same as before; no changes here)
    # Copy your existing implementation of Moving Average, Momentum,
    # RSI_MA, and Streak strategies, unmodified.
    return df


# ---------------------------
# Sidebar: Inputs
# ---------------------------
with st.sidebar.expander("Data Settings", expanded=True):
    ticker         = st.text_input("Enter Ticker Symbol", value="AAPL")
    min_date       = pd.to_datetime('2000-01-01').date()
    max_date       = datetime.date.today()
    start_date     = st.date_input("Start Date",  value=min_date, min_value=min_date, max_value=max_date)
    end_date       = st.date_input("End Date",    value=max_date, min_value=min_date, max_value=max_date)

with st.sidebar.expander("Strategy Parameters", expanded=True):
    strategy              = st.selectbox("Select Strategy", [
        "Moving Average Crossover", "Momentum", "RSI_MA Strategy", "Streak Strategy"
    ])
    strategy_params       = {}
    # ... (same as before; no changes here for your sliders and params)


# ---------------------------
# Main Execution
# ---------------------------
if ticker:
    try:
        raw_df = load_data(ticker, start_date, end_date)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # download raw data button
    st.sidebar.download_button(
        label     = "Download Raw Ticker Data as CSV",
        data      = raw_df.to_csv(index=False),
        file_name = f"{ticker}_raw_data.csv",
        mime      = "text/csv"
    )

    # run strategy
    with st.spinner("Executing strategy..."):
        df = execute_strategy(raw_df.copy(), strategy, strategy_params)
    with st.spinner("Calculating portfolio metrics..."):
        df = calculate_portfolio(df)

    # trades & equity line
    trades    = calculate_trades(df)
    trade_log = calculate_trade_log(df)

    st.subheader("Equity Line")
    plot_equity_line_static(trades, ticker)

    # metrics display
    metrics = compute_metrics(trades)
    st.subheader("Strategy Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Number of Trades",       f"{metrics['num_trades']}")
    c2.metric("Mean Points per Trade",  f"{metrics['points_mean']:.2f}")
    c3.metric("Total Gain/Loss",        f"{metrics['total_gain_loss']:.2f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Percentage Wins",        f"{metrics['percent_wins']:.2f} %")
    c5.metric("Profit Factor",          f"{metrics['profit_factor']:.2f}")
    c6.metric("T-Test tscore",          f"{metrics['t_stat']:.3f}")

    # trade log table
    if trade_log:
        st.subheader("Trade List")
        trade_log_df = pd.DataFrame(trade_log)
        trade_log_df['date entry'] = pd.to_datetime(trade_log_df['date entry']).dt.strftime('%Y-%m-%d')
        trade_log_df['date exit']  = pd.to_datetime(trade_log_df['date exit']).dt.strftime('%Y-%m-%d')
        st.dataframe(trade_log_df)
        st.download_button(
            label     = "Download Trade List as CSV",
            data      = trade_log_df.to_csv(index=False),
            file_name = f"{ticker}_trade_list.csv",
            mime      = "text/csv"
        )

    # final backtest data download
    df_display = df.copy()
    df_display['Date'] = pd.to_datetime(df_display['Date']).dt.strftime('%Y-%m-%d')
    st.download_button(
        label     = "Download Backtest Data as CSV",
        data      = df_display.to_csv(index=False),
        file_name = f"{ticker}_backtest_data.csv",
        mime      = "text/csv"
    )
