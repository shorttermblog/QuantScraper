import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt  # New import for static charting

# Set page configuration without specifying a wide layout
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

    # Apply Wilder's smoothing using an exponential weighted moving average
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
    with st.spinner("Downloading data..."):
        

        
        
        try:
        data = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
        st.error(f"yfinance download failed: {e}")
        return pd.DataFrame()

        df = data.round(2)
        # Flatten MultiIndex if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [col.split(',')[0] for col in df.columns]
        df.reset_index(inplace=True)
        mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
        return df.loc[mask]


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
            trade_profit = row['Close'] - open_trade['Close']
            trades.append(trade_profit)
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
            trade = {
                "date entry": open_trade['Date'],
                "price entry": open_trade['Close'],
                "date exit": row['Date'],
                "price exit": row['Close'],
                "Return": round(row['Close'] - open_trade['Close'], 2)
            }
            trades.append(trade)
            open_trade = None
    return trades


def compute_metrics(trades):
    if trades:
        num_trades = len(trades)
        points_mean = np.mean(trades)
        total_gain_loss = np.sum(trades)
        wins = [t for t in trades if t > 0]
        percent_wins = (len(wins) / num_trades) * 100
        total_wins = np.sum(wins)
        losses = [t for t in trades if t < 0]
        total_losses = np.sum(losses)
        profit_factor = total_wins / abs(total_losses) if total_losses != 0 else np.nan
        t_stat, t_pvalue = stats.ttest_1samp(trades, 0)
    else:
        num_trades = 0
        points_mean = total_gain_loss = profit_factor = t_stat = t_pvalue = np.nan
        percent_wins = 0
    return {
        "num_trades": num_trades,
        "points_mean": points_mean,
        "total_gain_loss": total_gain_loss,
        "percent_wins": percent_wins,
        "profit_factor": profit_factor,
        "t_stat": t_stat,
        "t_pvalue": t_pvalue
    }


# ---------------------------
# Static Chart Plotting Function using Matplotlib for Equity Line
# ---------------------------
def plot_equity_line_static(trades, ticker):
    cumulative_equity = np.cumsum(trades)
    trade_numbers = list(range(1, len(cumulative_equity) + 1))

    # Create a static chart with the same features and colors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trade_numbers, cumulative_equity, linestyle='-', label='Equity Curve', color='blue')
    ax.set_title(f"{ticker} Equity Line")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative Profit (Points)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# ---------------------------
# Strategy Execution Function for Extensibility
# ---------------------------
def execute_strategy(df, strategy, params):
    if strategy == "Moving Average Crossover":
        st.markdown(
            "<h2 style='font-size:20px;'>The strategy buys when the short term average crosses above the long term average</h2>",
            unsafe_allow_html=True)
        st.markdown(
            "<h2 style='font-size:20px;'>The position is closed when the short term average crosses below the long term average</h2>",
            unsafe_allow_html=True)
        short_window = params.get("short_window")
        long_window = params.get("long_window")
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean().round(2)
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean().round(2)
        df['Signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)
        df['Position'] = df['Signal'].diff().fillna(0)

    elif strategy == "Momentum":
        st.markdown(
            "<h2 style='font-size:20px;'>The strategy buys when the Momentum value is greater than the threshold</h2>",
            unsafe_allow_html=True)
        st.markdown(
            "<h2 style='font-size:20px;'>The position is closed when the Momentum value falls below the threshold</h2>",
            unsafe_allow_html=True)
        momentum_window = params.get("momentum_window")
        threshold = params.get("threshold")
        df['Momentum'] = df['Close'].pct_change(periods=momentum_window).round(2)
        df['Signal'] = np.where(df['Momentum'] > threshold, 1, 0)
        df['Position'] = df['Signal'].diff().fillna(0)

    elif strategy == "RSI_MA Strategy":
        st.markdown(
            "<h2 style='font-size:20px;'>The strategy buys when the RSI is below the threshold and the price is above the Slow Moving Average</h2>",
            unsafe_allow_html=True)
        st.markdown(
            "<h2 style='font-size:20px;'>The position is closed when the price is above the Fast Moving Average</h2>",
            unsafe_allow_html=True)
        rsi_window = params.get("rsi_window", 14)
        rsi_thresh = params.get("rsi_thresh", 30)
        df["RSI"] = compute_RSI(df["Close"], window=rsi_window)
        slow_ma = params.get("slow_ma", 50)
        fast_ma = params.get("fast_ma", 10)
        df["Slow_MA"] = df["Close"].rolling(window=slow_ma).mean().round(2)
        df["Fast_MA"] = df["Close"].rolling(window=fast_ma).mean().round(2)

        signals = []
        in_position = False

        for row in df.itertuples(index=False):
            if not in_position and (row.RSI < rsi_thresh and row.Close > row.Slow_MA):
                signals.append(1)
                in_position = True
            elif in_position and (row.Close > row.Fast_MA):
                signals.append(0)
                in_position = False
            else:
                signals.append(1 if in_position else 0)

        df["Signal"] = signals
        df["Position"] = df["Signal"].diff().fillna(0)

    elif strategy == "Streak Strategy":
        st.markdown(
            "<h2 style='font-size:20px;'>This strategy buys after a specified number of consecutive closes moving in the same direction "
            "(up if positive, down if negative)</h2>",
            unsafe_allow_html=True)
        st.markdown(
            "<h2 style='font-size:20px;'> The position is closed after holding the position for a fixed number of days.</h2>",
            unsafe_allow_html=True)
        streak_length = params.get("streak_length")
        holding_period = params.get("holding_period")
        df['Signal'] = 0
        in_trade = False
        trade_end_index = None
        streak = 0

        for i in range(1, len(df)):
            if in_trade:
                if i == trade_end_index:
                    df.at[i, 'Signal'] = 0
                    in_trade = False
                    streak = 0
                else:
                    df.at[i, 'Signal'] = 1
                continue

            if streak_length > 0:
                if df.at[i, 'Close'] > df.at[i - 1, 'Close']:
                    streak += 1
                else:
                    streak = 0
            elif streak_length < 0:
                if df.at[i, 'Close'] < df.at[i - 1, 'Close']:
                    streak += 1
                else:
                    streak = 0

            if streak >= abs(streak_length):
                df.at[i, 'Signal'] = 1
                in_trade = True
                trade_end_index = min(i + holding_period, len(df) - 1)

        df['Position'] = df['Signal'].diff().fillna(0)

    return df


# ---------------------------
# Sidebar: Data Settings and Strategy Parameters
# ---------------------------
with st.sidebar.expander("Data Settings", expanded=True):
    ticker = st.text_input("Enter Ticker Symbol", value="AAPL")
    fixed_min_date = pd.to_datetime('2000-01-01').date()
    default_max_date = datetime.date.today()
    start_date = st.date_input("Start Date", value=fixed_min_date, min_value=fixed_min_date, max_value=default_max_date)
    end_date = st.date_input("End Date", value=default_max_date, min_value=fixed_min_date, max_value=default_max_date)

with st.sidebar.expander("Strategy Parameters", expanded=True):
    strategy = st.selectbox("Select Strategy",
                            options=["Moving Average Crossover", "Momentum", "RSI_MA Strategy", "Streak Strategy"])
    strategy_params = {}
    if strategy == "Moving Average Crossover":
        st.markdown("#### Moving Average Parameters")
        strategy_params["short_window"] = st.slider("Short Moving Average Window", min_value=3, max_value=50, value=5,
                                                    step=1)
        strategy_params["long_window"] = st.slider("Long Moving Average Window", min_value=10, max_value=200, value=100,
                                                   step=1)
    elif strategy == "Momentum":
        st.markdown("#### Momentum Strategy Parameters")
        strategy_params["momentum_window"] = st.slider("Momentum Window", min_value=2, max_value=30, value=10, step=1)
        strategy_params["threshold"] = st.slider("Momentum Threshold", min_value=0.0, max_value=0.2, value=0.04,
                                                 step=0.005)
    elif strategy == "RSI_MA Strategy":
        st.markdown("#### RSI_MA Strategy Parameters")
        strategy_params["rsi_window"] = st.slider("RSI Window", min_value=2, max_value=30, value=2, step=1)
        strategy_params["rsi_thresh"] = st.slider("RSI Threshold", min_value=0, max_value=50, value=5, step=1)
        strategy_params["slow_ma"] = st.slider("Slow MA Period", min_value=10, max_value=200, value=200, step=1)
        strategy_params["fast_ma"] = st.slider("Fast MA Period", min_value=3, max_value=50, value=5, step=1)
    elif strategy == "Streak Strategy":
        st.markdown("#### Streak Strategy Parameters")
        strategy_params["streak_length"] = st.select_slider(
            "Streak Length (consecutive closes up or down)",
            options=[-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6],
            value=3
        )
        strategy_params["holding_period"] = st.slider("Holding Period (days)", min_value=1, max_value=10, value=3,
                                                      step=1)

# ---------------------------
# Main Execution: Data Download, Strategy, and Visualization
# ---------------------------
if ticker:
    raw_df = load_data(ticker, start_date, end_date)
    if raw_df.empty:
        st.info(f"No data found for ticker {ticker} between {start_date} and {end_date}.")
    else:
        st.sidebar.download_button(label="Download Raw Ticker Data as CSV",
                                   data=raw_df.to_csv(index=False),
                                   file_name=f"{ticker}_raw_data.csv",
                                   mime="text/csv")
        df = raw_df.copy()
        with st.spinner("Executing strategy..."):
            df = execute_strategy(df, strategy, strategy_params)
        with st.spinner("Calculating portfolio metrics..."):
            df = calculate_portfolio(df)

        trades = calculate_trades(df)
        trade_log = calculate_trade_log(df)

        st.subheader("Equity Line")
        # Replace the interactive Plotly chart with the static Matplotlib chart
        plot_equity_line_static(trades, ticker)

        metrics = compute_metrics(trades)
        st.subheader("Strategy Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Trades", f"{metrics['num_trades']}")
        col2.metric("Mean Points per Trade", f"{round(metrics['points_mean'], 2)}")
        col3.metric("Total Gain/Loss", f"{round(metrics['total_gain_loss'], 2)}")
        col4, col5, col6 = st.columns(3)
        col4.metric("Percentage Wins", f"{round(metrics['percent_wins'], 2)} %")
        col5.metric("Profit Factor", f"{round(metrics['profit_factor'], 2)}")
        col6.metric("T-Test tscore", f"{round(metrics['t_stat'], 3)}")

        if trade_log:
            st.subheader("Trade List")
            trade_log_df = pd.DataFrame(trade_log)
            trade_log_df['date entry'] = pd.to_datetime(trade_log_df['date entry']).dt.strftime('%Y-%m-%d')
            trade_log_df['date exit'] = pd.to_datetime(trade_log_df['date exit']).dt.strftime('%Y-%m-%d')
            st.dataframe(trade_log_df)
            st.download_button(label="Download Trade List as CSV",
                               data=trade_log_df.to_csv(index=False),
                               file_name=f"{ticker}_trade_list.csv",
                               mime="text/csv")

        df_display = df.copy()
        df_display['Date'] = pd.to_datetime(df_display['Date']).dt.strftime('%Y-%m-%d')
        st.download_button(label="Download Backtest Data as CSV",
                           data=df_display.to_csv(index=False),
                           file_name=f"{ticker}_backtest_data.csv",
                           mime="text/csv")
