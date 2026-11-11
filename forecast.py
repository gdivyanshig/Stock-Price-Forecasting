"""Stock data loading, Prophet forecasting and plotting."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet


def load_stock(ticker, years=5):
    """Download daily closing prices and return a DataFrame with Prophet's columns: ds (date), y (price)."""
    data = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a one-column table
        close = close.iloc[:, 0]
    df = pd.DataFrame({"ds": pd.to_datetime(close.index).tz_localize(None), "y": close.to_numpy()})
    return df.dropna().drop_duplicates("ds").sort_values("ds").reset_index(drop=True)


def run_forecast(df, days):
    """Fit Prophet on all the data and forecast the next `days` business days."""
    model = Prophet(daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=days, freq="B")  # B = business days
    return model, model.predict(future)


def evaluate(df, days):
    """Hide the last `days` prices, predict them, and return (MAE, MAPE %)."""
    train, test = df.iloc[:-days], df.iloc[-days:]
    pred = Prophet(daily_seasonality=False).fit(train).predict(test[["ds"]])
    actual, predicted = test["y"].to_numpy(), pred["yhat"].to_numpy()
    mae = np.abs(actual - predicted).mean()
    mape = (np.abs(actual - predicted) / np.abs(actual)).mean() * 100
    return mae, mape


def plot_forecast(df, forecast, days, ticker):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["ds"], df["y"], color="red", alpha=0.6, label="Actual Close Price")
    ax.plot(forecast["ds"], forecast["yhat"], color="blue", label="Forecasted Close Price")
    future = forecast.iloc[-days:]
    ax.fill_between(future["ds"], future["yhat_lower"], future["yhat_upper"],
                    color="blue", alpha=0.2, label="Uncertainty range")
    ax.set_title(f"{ticker} Stock Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig
