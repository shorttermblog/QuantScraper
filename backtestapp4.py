import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import yfinance as yf
from scipy import stats

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
    return rsi


# ---------------------------
# Data Download Function with Caching & Loading Indicator
# ---------------------------
@st.cache_data
def load_data(ticker, start_date, end_date):
    with st.spinner("Downloading data..."):
        df = yf.download(ticker, start=start_date, end=end_date)
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
    """Calculate portfolio value based on generated signals."""
    df['Holdings'] = df['Signal'] * df['Close']
    df['Cash'] = initial_capital - (df['Position'].fillna(0) * df['Close']).cumsum()
    df['Portfolio'] = df['Holdings'] + df['Cash']
    return df


def calculate_trades(df):
    """Loop over the dataframe to compute trade profits."""
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
    """Loop over the dataframe to compute detailed trade logs.
    Each trade record includes: date entry, price entry, date exit, price exit, gain/loss.
    """
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
                "Return": row['Close'] - open_trade['Close']
            }
            trades.append(trade)
            open_trade = None
    return trades


def compute_metrics(trades):
    """Calculate and return trade metrics."""
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
# Plotting Functions using Plotly for Interactivity
# ---------------------------
def plot_equity_line(trades, ticker):
    """Plot the cumulative equity curve based on trade profits with ticker as title."""
    cumulative_equity = np.cumsum(trades)
    trade_numbers = list(range(1, len(cumulative_equity) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trade_numbers, y=cumulative_equity, mode='lines+markers', name='Equity Curve'))
    fig.update_layout(title=f"{ticker} Equity Line",
                      xaxis_title="Trade Number",
                      yaxis_title="Cumulative Profit (Points)")
    st.plotly_chart(fig, use_container_width=True)


def plot_price_chart(df, strategy, params, ticker):
    """Plot interactive price chart with signals and ticker in title."""
    fig = go.Figure()
    # Plot the close price line
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', opacity=0.5))

    if strategy == "Moving Average Crossover":
        short_window = params.get("short_window")
        long_window = params.get("long_window")
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_short'], mode='lines', name=f'SMA {short_window}'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_long'], mode='lines', name=f'SMA {long_window}'))
        # Use Position to identify signals
        buy_signals = df[df['Position'] == 1]
        sell_signals = df[df['Position'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['SMA_short'], mode='markers',
                                 marker_symbol='triangle-up', marker_size=10, marker_color='green',
                                 name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['SMA_short'], mode='markers',
                                 marker_symbol='triangle-down', marker_size=10, marker_color='red',
                                 name='Sell Signal'))

    elif strategy == "Momentum":
        # Use Position column for buy/sell signals
        buy_signals = df[df['Position'] == 1]
        sell_signals = df[df['Position'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers',
                                 marker_symbol='triangle-up', marker_size=10, marker_color='green',
                                 name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers',
                                 marker_symbol='triangle-down', marker_size=10, marker_color='red',
                                 name='Sell Signal'))

    elif strategy == "RSI_MA Strategy":
        # Plot the moving averages used in the RSI strategy
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Slow_MA'], mode='lines', name='Slow MA'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Fast_MA'], mode='lines', name='Fast MA'))
        # Use the same approach as the other strategies by using the Position column for signals
        buy_signals = df[df['Position'] == 1]
        sell_signals = df[df['Position'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers',
                                 marker_symbol='triangle-up', marker_size=10, marker_color='green',
                                 name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers',
                                 marker_symbol='triangle-down', marker_size=10, marker_color='red',
                                 name='Sell Signal'))

    elif strategy == "Streak Strategy":
        # Use the Position column for signals
        buy_signals = df[df['Position'] == 1]
        sell_signals = df[df['Position'] == -1]
        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers',
                                 marker_symbol='triangle-up', marker_size=10, marker_color='green',
                                 name='Buy Signal'))
        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers',
                                 marker_symbol='triangle-down', marker_size=10, marker_color='red',
                                 name='Sell Signal'))

    fig.update_layout(title=f"{ticker} Price Chart with Strategy Signals",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)


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
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
        df['Signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, 0)
        df['Position'] = df['Signal'].diff().fillna(0)

    elif strategy == "Momentum":
        momentum_window = params.get("momentum_window")
        threshold = params.get("threshold")
        df['Momentum'] = df['Close'].pct_change(periods=momentum_window)
        df['Signal'] = np.where(df['Momentum'] > threshold, 1, 0)
        df['Position'] = df['Signal'].diff().fillna(0)

    elif strategy == "RSI_MA Strategy":
        rsi_window = params.get("rsi_window", 14)
        rsi_thresh = params.get("rsi_thresh", 30)
        df["RSI"] = compute_RSI(df["Close"], window=rsi_window)
        slow_ma = params.get("slow_ma", 50)
        fast_ma = params.get("fast_ma", 10)
        df["Slow_MA"] = df["Close"].rolling(window=slow_ma).mean()
        df["Fast_MA"] = df["Close"].rolling(window=fast_ma).mean()

        signals = []
        in_position = False

        for row in df.itertuples(index=False):
            # Entry: if not in position, RSI is below threshold, and close is above slow MA
            if not in_position and (row.RSI < rsi_thresh and row.Close > row.Slow_MA):
                signals.append(1)
                in_position = True
            # Exit: if in position and close is above fast MA, exit the trade
            elif in_position and (row.Close > row.Fast_MA):
                signals.append(0)
                in_position = False
            else:
                signals.append(1 if in_position else 0)

        df["Signal"] = signals
        df["Position"] = df["Signal"].diff().fillna(0)

    elif strategy == "Streak Strategy":
        st.markdown("<h2 style='font-size:20px;'>Streak Strategy</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p>This strategy buys after a specified number of consecutive closes moving in the same direction (up if positive, down if negative) and sells after holding the position for a fixed number of days.</p>",
            unsafe_allow_html=True)
        streak_length = params.get("streak_length")
        holding_period = params.get("holding_period")
        # Initialize the Signal column with zeros
        df['Signal'] = 0
        in_trade = False
        trade_end_index = None
        streak = 0

        # Iterate over the data starting from the second row
        for i in range(1, len(df)):
            if in_trade:
                # Check if it is time to exit the trade
                if i == trade_end_index:
                    # Mark exit signal
                    df.at[i, 'Signal'] = 0
                    in_trade = False
                    streak = 0
                else:
                    # Maintain the buy signal throughout the holding period
                    df.at[i, 'Signal'] = 1
                continue

            # Not in a trade; update the streak count based on the streak direction
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

            # When the streak meets or exceeds the absolute threshold, generate a buy signal
            if streak >= abs(streak_length):
                df.at[i, 'Signal'] = 1  # Buy signal
                in_trade = True
                trade_end_index = min(i + holding_period, len(df) - 1)

        # Compute the Position based on the change in Signal
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
        plot_equity_line(trades, ticker)

        metrics = compute_metrics(trades)
        st.subheader("Strategy Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Trades", f"{metrics['num_trades']}")
        col2.metric("Mean Points per Trade", f"{round(metrics['points_mean'], 2)}")
        col3.metric("Total Gain/Loss", f"{round(metrics['total_gain_loss'], 2)}")
        col4, col5, col6 = st.columns(3)
        col4.metric("Percentage Wins", f"{round(metrics['percent_wins'], 2)} %")
        col5.metric("Profit Factor", f"{round(metrics['profit_factor'], 2)}")
        #col6.metric("T-Test p-value", f"{round(metrics['t_pvalue'], 3)}")
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

        st.subheader("Price Chart with Signals")
        plot_price_chart(df, strategy, strategy_params, ticker)

        st.subheader("Backtest Data")
        df_display = df.copy()
        df_display['Date'] = pd.to_datetime(df_display['Date']).dt.strftime('%Y-%m-%d')
        st.dataframe(df_display)
        st.download_button(label="Download Backtest Data as CSV",
                           data=df_display.to_csv(index=False),
                           file_name=f"{ticker}_backtest_data.csv",
                           mime="text/csv")
