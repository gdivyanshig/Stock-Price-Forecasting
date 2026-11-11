# 📈 Stock Price Forecasting with Prophet

A simple Streamlit app that forecasts stock prices using [Facebook Prophet](https://facebook.github.io/prophet/).
Enter any ticker, and the app downloads the price history, predicts the next few weeks, and shows how accurate
the model is on data it has not seen. Built during my ML Engineering internship at APL Apollo Tubes.

## Features
- Works with any Yahoo Finance ticker (`TSLA`, `AAPL`, `RELIANCE.NS`, ...)
- Forecasts 7 to 90 business days ahead with an **uncertainty range**
- **Backtest accuracy** (MAE and MAPE) calculated on the most recent prices the model did not train on
- Trend and seasonality charts showing what the model learned
- Download the forecast as CSV

## Why Prophet?
- Stock data has **gaps** (weekends and holidays), and Prophet handles missing days without extra work.
- It models **trend changes** and **weekly and yearly patterns** automatically.
- It returns **uncertainty intervals**, which matter because stock prices are noisy.
- It needs little tuning, so the code stays simple and easy to explain.

## Tech stack
Python · Pandas · NumPy · Matplotlib · Prophet · Streamlit · yfinance

## Project structure
```
├── app.py            # Streamlit user interface
├── forecast.py       # Data download, Prophet model, evaluation, plotting
├── requirements.txt
└── README.md
```

## Getting started
```bash
git clone https://github.com/<your-username>/prophet-forecasting-app.git
cd prophet-forecasting-app
pip install -r requirements.txt
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## How it works
1. **Load and clean**: download daily closing prices, rename columns to Prophet's `ds` (date) and `y` (price), remove missing values and duplicates.
2. **Fit**: train Prophet on the full history.
3. **Forecast**: predict the next N business days.
4. **Evaluate**: hide the last N days, train on the rest, predict the hidden days, and report the error.
5. **Visualise**: plot actual vs forecast with the uncertainty band, plus the trend and seasonality components.

## Screenshots
<!-- Add screenshots after running the app, for example: -->
<!-- ![Forecast](screenshots/forecast.png) -->

## Limitations
- Stock prices behave close to a random walk, so forecasts are only illustrative. **This is not financial advice.**
- Prophet uses only past prices and ignores news, earnings and market events.
- Accuracy is checked on a single hold-out window.

## Future improvements
- Rolling cross-validation for a more reliable accuracy estimate
- Add extra regressors such as trading volume or market indices
- Compare against ARIMA or LSTM models

## Author
**Divyanshi Gupta** · B.Tech CSE (AI & ML), SRM Institute of Science and Technology
[LinkedIn](https://www.linkedin.com/) · [GitHub](https://github.com/)
