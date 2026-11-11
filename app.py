"""Streamlit app:  streamlit run app.py"""
import streamlit as st

import forecast as fc

st.set_page_config(page_title="Stock Price Forecasting", page_icon="📈", layout="wide")
st.title("📈 Stock Price Forecasting with Prophet")

ticker = st.sidebar.text_input("Stock ticker (e.g. TSLA, AAPL, RELIANCE.NS)", "TSLA").upper().strip()
years = st.sidebar.slider("Years of history", 2, 10, 5)
days = st.sidebar.slider("Days to forecast", 7, 90, 30)


@st.cache_data(show_spinner="Downloading prices…", ttl=3600)
def get_data(ticker, years):
    return fc.load_stock(ticker, years)


try:
    df = get_data(ticker, years)
except Exception as error:
    st.error(f"Could not load data: {error}")
    st.stop()

with st.spinner("Training model…"):
    model, forecast = fc.run_forecast(df, days)
    mae, mape = fc.evaluate(df, days)

col1, col2, col3 = st.columns(3)
col1.metric("Latest close", f"${df['y'].iloc[-1]:,.2f}")
col2.metric("Backtest MAE", f"${mae:.2f}")
col3.metric("Backtest MAPE", f"{mape:.2f}%")
st.caption(f"Accuracy is measured by hiding the last {days} days, predicting them, and comparing to the real prices.")

st.pyplot(fc.plot_forecast(df, forecast, days, ticker))

with st.expander("Trend and seasonality learned by the model"):
    st.pyplot(model.plot_components(forecast))

result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)
result.columns = ["Date", "Predicted", "Lower", "Upper"]
st.subheader("Forecast table")
st.dataframe(result.round(2), hide_index=True)
st.download_button("Download forecast as CSV", result.to_csv(index=False), "forecast.csv", "text/csv")
st.caption("Educational project. Not financial advice.")
