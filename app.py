import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(
    page_title="StockScope AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    h1 { margin-bottom: .1rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        padding: .75rem .9rem;
        border-radius: 14px;
    }
    .small-note {font-size: .86rem; opacity: .75;}

    @media (max-width: 700px) {
        .block-container {
            padding-top: .45rem;
            padding-left: .75rem;
            padding-right: .75rem;
        }
        h1 { font-size: 1.85rem !important; }
        h2 { font-size: 1.45rem !important; }
        h3 { font-size: 1.15rem !important; }
        div[data-testid="stMetric"] { padding: .55rem .65rem; }
        div[data-testid="stPlotlyChart"] { margin-left: -.35rem; margin-right: -.35rem; }
        .stButton button { min-height: 46px; }
        input { font-size: 16px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 StockScope AI")
st.caption("Type a ticker, inspect the stock, and see an experimental model forecast.")

with st.form("stock_lookup", border=False):
    q1, q2, q3 = st.columns([1.35, 1, 1])
    with q1:
        ticker_input = st.text_input(
            "Ticker symbol",
            value="AAPL",
            placeholder="AAPL, NVDA, TSLA...",
        ).upper().strip()
    with q2:
        period = st.selectbox(
            "Chart history",
            ["3mo", "6mo", "1y", "2y", "5y", "10y"],
            index=2,
        )
    with q3:
        forecast_days = st.select_slider(
            "Forecast days",
            options=[5, 10, 15, 20, 30, 45, 60],
            value=20,
        )
    run = st.form_submit_button("Analyze Stock", type="primary", use_container_width=True)

st.caption("Examples: AAPL · MSFT · NVDA · TSLA · AMZN · GOOGL · META")


def fmt_money(x):
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"${float(x):,.2f}"
    except Exception:
        return "—"


def fmt_big(x):
    try:
        if x is None or pd.isna(x):
            return "—"
        x = float(x)
        for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
            if abs(x) >= scale:
                return f"{x/scale:.2f}{unit}"
        return f"{x:,.0f}"
    except Exception:
        return "—"


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df):
    x = df.copy()
    x["Return_1d"] = x["Close"].pct_change()
    x["MA5"] = x["Close"].rolling(5).mean()
    x["MA10"] = x["Close"].rolling(10).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA50"] = x["Close"].rolling(50).mean()
    x["Volatility20"] = x["Return_1d"].rolling(20).std()
    x["RSI14"] = rsi(x["Close"], 14)
    x["Momentum5"] = x["Close"].pct_change(5)
    x["Momentum20"] = x["Close"].pct_change(20)
    x["VolumeChange"] = x["Volume"].pct_change()
    return x


@st.cache_data(ttl=900, show_spinner=False)
def load_stock(ticker_symbol, chart_period):
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=chart_period, auto_adjust=False)
    model_hist = ticker.history(period="5y", auto_adjust=False)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    return hist, model_hist, info


@st.cache_data(ttl=3600, show_spinner=False)
def make_forecast_cached(model_hist, days):
    feat = add_features(model_hist)
    feature_cols = [
        "Close", "MA5", "MA10", "MA20", "MA50", "Volatility20",
        "RSI14", "Momentum5", "Momentum20", "VolumeChange",
    ]
    feat["Target"] = feat["Close"].shift(-1)
    train = feat.dropna(subset=feature_cols + ["Target"]).copy()

    if len(train) < 100:
        raise ValueError("Not enough historical data to create a useful forecast for this ticker.")

    X = train[feature_cols]
    y = train["Target"]

    tscv = TimeSeriesSplit(n_splits=4)
    maes = []
    for tr_idx, te_idx in tscv.split(X):
        model_cv = RandomForestRegressor(
            n_estimators=160, max_depth=8, random_state=42, n_jobs=-1
        )
        model_cv.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = model_cv.predict(X.iloc[te_idx])
        maes.append(mean_absolute_error(y.iloc[te_idx], p))

    model = RandomForestRegressor(
        n_estimators=260, max_depth=9, random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    work = model_hist.copy()
    predictions = []
    last_date = work.index[-1]
    current_volume = float(work["Volume"].tail(20).median())

    for _ in range(days):
        tmp = add_features(work)
        row = tmp.iloc[-1]
        vals = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
        pred = float(model.predict(vals)[0])
        next_date = last_date + pd.tseries.offsets.BDay(1)
        synthetic = pd.DataFrame(
            {
                "Open": [pred], "High": [pred], "Low": [pred],
                "Close": [pred], "Volume": [current_volume],
            },
            index=[next_date],
        )
        work = pd.concat([work, synthetic])
        predictions.append((next_date, pred))
        last_date = next_date

    pred_df = pd.DataFrame(
        predictions, columns=["Date", "Predicted Close"]
    ).set_index("Date")
    validation_mae = float(np.mean(maes))
    return pred_df, validation_mae


if not run:
    st.info("Enter a ticker above and tap **Analyze Stock**.")
    st.stop()

if not ticker_input:
    st.error("Enter a ticker symbol.")
    st.stop()

with st.spinner(f"Loading {ticker_input}..."):
    try:
        hist, model_hist, info = load_stock(ticker_input, period)
    except Exception as e:
        st.error(f"Could not load {ticker_input}: {e}")
        st.stop()

if hist.empty or model_hist.empty:
    st.error("No market data was found. Check the ticker symbol and try again.")
    st.stop()

company = info.get("longName") or info.get("shortName") or ticker_input
price = float(hist["Close"].iloc[-1])
prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
change_pct = ((price / prev) - 1) * 100 if prev else 0

st.subheader(f"{company} ({ticker_input})")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Price", fmt_money(price), f"{change_pct:+.2f}%")
m2.metric("Market Cap", fmt_big(info.get("marketCap")))
m3.metric("P/E", f"{info.get('trailingPE'):.2f}" if info.get("trailingPE") else "—")
m4.metric("52W High", fmt_money(info.get("fiftyTwoWeekHigh")))
m5.metric("52W Low", fmt_money(info.get("fiftyTwoWeekLow")))

st.markdown("### Price chart")
chart_df = add_features(hist)
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
        name="Price",
    )
)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA20"], mode="lines", name="20-day MA"))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA50"], mode="lines", name="50-day MA"))
fig.update_layout(
    height=470,
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=1.03, x=0),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

left, right = st.columns([1.15, 1])
with left:
    st.markdown("### Company / stock information")
    description = info.get("longBusinessSummary")
    if description:
        with st.expander("About the company", expanded=False):
            st.write(description)
    details = {
        "Sector": info.get("sector", "—"),
        "Industry": info.get("industry", "—"),
        "Exchange": info.get("exchange", "—"),
        "Beta": info.get("beta", "—"),
        "Dividend Yield": f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "—",
        "EPS": fmt_money(info.get("trailingEps")),
        "Forward P/E": f"{info.get('forwardPE'):.2f}" if info.get("forwardPE") else "—",
        "Average Volume": fmt_big(info.get("averageVolume")),
    }
    st.dataframe(
        pd.DataFrame(details.items(), columns=["Metric", "Value"]),
        hide_index=True,
        use_container_width=True,
    )

with right:
    st.markdown("### Technical snapshot")
    f = add_features(model_hist).iloc[-1]
    rsi_val = float(f["RSI14"]) if not pd.isna(f["RSI14"]) else np.nan
    ma20 = float(f["MA20"]) if not pd.isna(f["MA20"]) else np.nan
    ma50 = float(f["MA50"]) if not pd.isna(f["MA50"]) else np.nan
    momentum20 = float(f["Momentum20"] * 100) if not pd.isna(f["Momentum20"]) else np.nan

    trend = "Bullish" if price > ma20 > ma50 else "Bearish" if price < ma20 < ma50 else "Mixed"
    rsi_signal = "Overbought" if rsi_val >= 70 else "Oversold" if rsi_val <= 30 else "Neutral"
    st.metric("Trend", trend)
    st.metric("RSI (14)", f"{rsi_val:.1f}" if not np.isnan(rsi_val) else "—", rsi_signal)
    st.metric("20-day momentum", f"{momentum20:+.2f}%" if not np.isnan(momentum20) else "—")
    st.write(f"20-day moving average: **{fmt_money(ma20)}**")
    st.write(f"50-day moving average: **{fmt_money(ma50)}**")

st.divider()
st.markdown("## 🔮 Experimental forecast")

with st.spinner("Building forecast..."):
    try:
        pred_df, validation_mae = make_forecast_cached(model_hist, forecast_days)
        ending = float(pred_df["Predicted Close"].iloc[-1])
        forecast_change = (ending / price - 1) * 100

        daily_vol = model_hist["Close"].pct_change().tail(60).std()
        uncertainty = price * daily_vol * math.sqrt(forecast_days) if not pd.isna(daily_vol) else validation_mae
        low_est = max(0, ending - 1.28 * uncertainty)
        high_est = ending + 1.28 * uncertainty

        mae_pct = validation_mae / price * 100 if price else 100
        confidence = "Moderate" if mae_pct < 2 else "Low–moderate" if mae_pct < 4 else "Low"

        p1, p2 = st.columns(2)
        p1.metric(
            f"Predicted price in {forecast_days} trading days",
            fmt_money(ending),
            f"{forecast_change:+.2f}%",
        )
        p2.metric("Estimated range", f"{fmt_money(low_est)} – {fmt_money(high_est)}")
        p3, p4 = st.columns(2)
        p3.metric("Backtest avg. error", fmt_money(validation_mae))
        p4.metric("Forecast confidence", confidence)

        forecast_fig = go.Figure()
        recent = model_hist.tail(120)
        forecast_fig.add_trace(
            go.Scatter(x=recent.index, y=recent["Close"], mode="lines", name="Historical close")
        )
        forecast_fig.add_trace(
            go.Scatter(
                x=pred_df.index,
                y=pred_df["Predicted Close"],
                mode="lines+markers",
                name="Model forecast",
            )
        )
        forecast_fig.update_layout(
            height=390,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.03, x=0),
        )
        st.plotly_chart(
            forecast_fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        if forecast_change > 5:
            outlook = "The model currently leans **bullish** over the selected forecast window."
        elif forecast_change < -5:
            outlook = "The model currently leans **bearish** over the selected forecast window."
        else:
            outlook = "The model currently sees a **mostly sideways / neutral** outlook over the selected forecast window."
        st.write(outlook)

        st.warning(
            "This forecast is experimental and can be wrong. It uses historical price/volume patterns only and does not know future news, earnings surprises, economic changes, or other events. It is not investment advice."
        )
        with st.expander("How the forecast works"):
            st.write(
                "A random-forest model uses price, moving averages, volatility, RSI, momentum, and volume changes from historical data. "
                "The future path is generated recursively from next-day model estimates. The displayed range is an uncertainty estimate, not a guaranteed interval."
            )
    except Exception as e:
        st.error(f"Forecast unavailable: {e}")
