"""Stock Market Analytics: historical market analysis and CAPM scenarios.

Data comes from Yahoo Finance and publicly available index-constituent CSVs.
Expected returns are editable assumptions, not forecasts or investment advice.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# Market identifiers and data-source URLs are configuration, not stock data.
MARKETS = {
    "S&P 500": {
        "benchmark": "^GSPC",
        "symbol": "$",
        "companies_url": (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/main/data/constituents.csv"
        ),
        "company_column": "Security",
        "ticker_column": "Symbol",
        "ticker_suffix": "",
        "minimum_constituents": 450,
    },
    "NIFTY 50": {
        "benchmark": "^NSEI",
        "symbol": "₹",
        "companies_url": (
            "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
        ),
        "company_column": "Company Name",
        "ticker_column": "Symbol",
        "ticker_suffix": ".NS",
        "minimum_constituents": 45,
    },
}

HISTORICAL_PERIODS = ("6mo", "1y", "2y", "3y", "5y")
TRADING_DAYS_PER_YEAR = 252  # Common annualization convention, not a return estimate.
MIN_DAILY_RETURNS = 30       # Minimum paired observations used for risk estimates.
PRICE_CACHE_SECONDS = 3600
COMPANY_CACHE_SECONDS = 86400


st.set_page_config(
    page_title="Stock Market Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp { background: #0b1220; color: #e8edf5; }
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] { display: none; }
        [data-testid="stMetric"] {
            background: #17263f;
            border: 1px solid #293b58;
            border-radius: 10px;
            padding: 14px;
        }
        .block-container { max-width: 1450px; padding-top: 2rem; }
        .dashboard-header {
            background: linear-gradient(110deg, #152b48, #0c1628);
            padding: 30px 32px;
            border: 1px solid #2b3b55;
            border-radius: 14px;
            margin-bottom: 22px;
        }
        .dashboard-header h1 { color: #f5f8ff; margin: 0 0 10px; }
        .dashboard-header p { color: #b8c9dd; margin: 0; }
    </style>
    <div class="dashboard-header">
        <h1>Stock Market Analytics</h1>
        <p>Market Overview | CAPM and Risk Analysis | Stock Deep Dive</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=COMPANY_CACHE_SECONDS)
def get_companies(market_name: str) -> pd.DataFrame:
    """Fetch current index constituents; never maintain a manual stock list."""
    config = MARKETS[market_name]
    response = requests.get(
        config["companies_url"],
        headers={"User-Agent": "Mozilla/5.0 (Stock Market Analytics research)"},
        timeout=20,
    )
    response.raise_for_status()

    source = pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")
    required = [config["company_column"], config["ticker_column"]]
    if any(column not in source.columns for column in required):
        raise ValueError(f"Constituent file does not contain columns: {required}")

    companies = source[required].copy()
    companies.columns = ["Company", "Ticker"]
    companies = companies.dropna(subset=["Company", "Ticker"])
    companies["Company"] = companies["Company"].astype(str).str.strip()
    companies["Ticker"] = companies["Ticker"].astype(str).str.strip()

    suffix = config["ticker_suffix"]
    if suffix:
        companies["Ticker"] = companies["Ticker"].where(
            companies["Ticker"].str.endswith(suffix),
            companies["Ticker"] + suffix,
        )
    else:
        # Yahoo Finance represents symbols such as BRK.B as BRK-B.
        companies["Ticker"] = companies["Ticker"].str.replace(
            ".", "-", regex=False
        )

    companies = companies[
        companies["Ticker"].ne("") & companies["Company"].ne("")
    ].drop_duplicates(subset="Ticker")
    if len(companies) < config["minimum_constituents"]:
        raise ValueError(
            f"Constituent file looks incomplete ({len(companies)} companies)."
        )

    return companies.sort_values("Company").reset_index(drop=True)


@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def get_prices(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    """Return adjusted closing prices for unique requested tickers."""
    unique_tickers = tuple(dict.fromkeys(tickers))
    if not unique_tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers=list(unique_tickers),
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
        multi_level_index=True,
    )
    if raw.empty or "Close" not in raw.columns.get_level_values(0):
        return pd.DataFrame()

    prices = raw["Close"].copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=unique_tickers[0])

    prices = prices.reindex(columns=list(unique_tickers))
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    return prices.sort_index().loc[~prices.index.duplicated(keep="last")]


def calculate_cagr(prices: pd.Series) -> float:
    """Annualized compound growth across the actual observed date range."""
    valid = prices.dropna()
    valid = valid[valid > 0]
    if len(valid) < 2:
        return np.nan

    years = (valid.index[-1] - valid.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float((valid.iloc[-1] / valid.iloc[0]) ** (1 / years) - 1)


def calculate_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough drawdown in the observed series."""
    valid = prices.dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return np.nan
    return float((valid / valid.cummax() - 1).min())


def calculate_capm(
    beta: float, risk_free_rate: float, expected_market_return: float
) -> float:
    """CAPM: expected asset return = Rf + beta * (E[Rm] - Rf)."""
    return float(risk_free_rate + beta * (expected_market_return - risk_free_rate))


def calculate_metrics(
    stock_prices: pd.Series,
    benchmark_prices: pd.Series,
    risk_free_rate: float,
    expected_market_return: float,
) -> dict[str, float] | None:
    """Compute historical risk metrics plus one assumption-driven CAPM result.

    Stock and benchmark are aligned on dates before daily returns are computed.
    Sharpe and Jensen's alpha are annualized approximations based on 252 sessions.
    """
    if risk_free_rate <= -1:
        return None

    aligned = pd.concat(
        [stock_prices.rename("Stock"), benchmark_prices.rename("Market")],
        axis=1,
        join="inner",
    ).dropna()
    aligned = aligned[(aligned > 0).all(axis=1)]
    daily = aligned.pct_change(fill_method=None).dropna()
    if len(daily) < MIN_DAILY_RETURNS:
        return None

    market_variance = daily["Market"].var()
    stock_volatility = daily["Stock"].std()
    if (
        not np.isfinite(market_variance)
        or market_variance <= 0
        or not np.isfinite(stock_volatility)
    ):
        return None

    beta = float(daily["Stock"].cov(daily["Market"]) / market_variance)
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    sharpe = (
        float((daily["Stock"].mean() - daily_rf) / stock_volatility)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
        if stock_volatility > 0
        else np.nan
    )
    daily_alpha = (
        daily["Stock"] - daily_rf
        - beta * (daily["Market"] - daily_rf)
    ).mean()

    return {
        "Beta": beta,
        "CAGR": calculate_cagr(aligned["Stock"]),
        "Volatility": float(stock_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "Drawdown": calculate_drawdown(aligned["Stock"]),
        "Sharpe": float(sharpe),
        "Alpha": float(daily_alpha * TRADING_DAYS_PER_YEAR),
        "CAPM": calculate_capm(beta, risk_free_rate, expected_market_return),
        "Observations": int(len(daily)),
    }


def normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Start all plotted price series at 100 on the same common date."""
    aligned = prices.dropna(how="any")
    aligned = aligned[(aligned > 0).all(axis=1)]
    if len(aligned) < 2:
        return pd.DataFrame()
    return aligned.div(aligned.iloc[0]).mul(100)


def line_chart(data: pd.DataFrame, title: str, y_axis: str) -> go.Figure:
    fig = go.Figure()
    for ticker in data.columns:
        values = data[ticker].dropna()
        if not values.empty:
            fig.add_trace(go.Scatter(
                x=values.index, y=values, mode="lines", name=ticker
            ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        xaxis_title="Date",
        yaxis_title=y_axis,
        hovermode="x unified",
        height=410,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig


@st.cache_data(ttl=PRICE_CACHE_SECONDS)
def top_performers(market_name: str) -> tuple[pd.DataFrame, int]:
    """Rank one-year total returns among today's index constituents.

    This is NOT a historical index-membership backtest: the current-member
    selection introduces survivorship bias, which should be disclosed.
    """
    companies = get_companies(market_name)
    symbols = companies["Ticker"].tolist()
    records = []

    for offset in range(0, len(symbols), 40):
        batch = tuple(symbols[offset:offset + 40])
        try:
            prices = get_prices(batch, "1y")
        except Exception:
            continue

        for ticker in batch:
            if ticker not in prices:
                continue
            history = prices[ticker].dropna()
            history = history[history > 0]
            if (
                len(history) >= 180
                and (history.index[-1] - history.index[0]).days >= 300
            ):
                records.append({
                    "Ticker": ticker,
                    "1Y Return (%)": (history.iloc[-1] / history.iloc[0] - 1) * 100,
                })

    if not records:
        return pd.DataFrame(), 0

    table = pd.DataFrame(records).merge(companies, on="Ticker", how="left")
    table = table.sort_values("1Y Return (%)", ascending=False)
    return table.head(10), len(records)


def load_company_lists() -> dict[str, pd.DataFrame]:
    available = {}
    for market_name in MARKETS:
        try:
            available[market_name] = get_companies(market_name)
        except Exception as exc:
            st.warning(f"Could not load {market_name} constituents: {exc}")
    return available


def show_metrics_table(result: pd.DataFrame) -> None:
    formats = {
        "Beta": "{:.2f}",
        "CAGR": "{:.2%}",
        "Volatility": "{:.2%}",
        "Drawdown": "{:.2%}",
        "Sharpe": "{:.2f}",
        "Alpha": "{:.2%}",
        "CAPM": "{:.2%}",
        "Observations": "{:.0f}",
    }
    st.dataframe(
        result.style.format(formats),
        hide_index=True,
        use_container_width=True,
    )


def show_projection(
    result: pd.DataFrame,
    expected_market_return: float,
) -> None:
    """Show a mechanical compound-growth scenario, never historical prices."""
    st.subheader("Illustrative CAPM Growth Scenario")
    years_ahead = st.slider(
        "Projection horizon (years)", 1, 10, 5, key="projection_horizon"
    )
    years = np.arange(years_ahead + 1)
    fig = go.Figure()

    for _, item in result.iterrows():
        assumed_return = item["CAPM"]
        if not np.isfinite(assumed_return) or assumed_return <= -1:
            st.warning(
                f"Cannot compound {item['Ticker']}: the CAPM return "
                "is at or below -100%."
            )
            continue
        fig.add_trace(go.Scatter(
            x=years,
            y=100 * (1 + assumed_return) ** years,
            name=item["Ticker"],
            mode="lines+markers",
        ))

    if expected_market_return > -1:
        fig.add_trace(go.Scatter(
            x=years,
            y=100 * (1 + expected_market_return) ** years,
            name="Assumed market",
            mode="lines+markers",
            line=dict(dash="dash"),
        ))
    else:
        st.warning("The assumed market return cannot be compounded below -100%.")

    if fig.data:
        fig.update_layout(
            title="CAPM-based scenario vs. assumed market",
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            xaxis_title="Years from today",
            yaxis_title="Illustrative index (starts at 100)",
            hovermode="x unified",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Scenario only: assumes the same annual CAPM return every year. "
        "It does not predict future stock prices, dividends, or volatility."
    )


def show_deep_dive(
    market_name: str,
    companies: pd.DataFrame | None,
    period: str,
    risk_free_rate: float,
) -> None:
    """Render either country's deep dive from the same code path."""
    st.subheader(market_name)
    if companies is None or companies.empty:
        st.info("Company list unavailable; try Refresh Data in the CAPM tab.")
        return

    choices = dict(zip(companies["Company"], companies["Ticker"]))
    selection = st.selectbox(
        "Select Company",
        options=list(choices),
        index=None,
        placeholder="Choose a company",
        key=f"deep_dive_{market_name}",
    )
    if not selection:
        return

    ticker = choices[selection]
    benchmark = MARKETS[market_name]["benchmark"]
    try:
        prices = get_prices((ticker, benchmark), period)
    except Exception as exc:
        st.error(f"Unable to download {ticker}: {exc}")
        return

    if (
        prices.empty
        or ticker not in prices
        or benchmark not in prices
        or prices[ticker].dropna().empty
        or prices[benchmark].dropna().empty
    ):
        st.warning("Stock or benchmark historical prices are unavailable.")
        return

    stock_history = prices[ticker].dropna()
    benchmark_history = prices[benchmark].dropna()
    historical_market_cagr = calculate_cagr(benchmark_history)

    # The CAPM tab stores each market/period's editable expected-return input.
    # For a market not edited there, use its observed historical index CAGR
    # as an explicitly labeled illustrative default, not a true forecast.
    assumption_key = f"market_return_{market_name}_{period}"
    if assumption_key in st.session_state:
        expected_market_return = st.session_state[assumption_key] / 100
        assumption_source = "the editable CAPM assumption"
    else:
        expected_market_return = historical_market_cagr
        assumption_source = "historical benchmark CAGR (illustrative default)"

    if not np.isfinite(expected_market_return):
        st.warning("Not enough benchmark history to estimate a market return.")
        return

    metrics = calculate_metrics(
        stock_history, benchmark_history, risk_free_rate, expected_market_return
    )
    if metrics is None:
        st.warning("At least 30 valid paired daily returns are needed.")
        return

    st.caption(f"{selection} ({ticker}); data through {stock_history.index[-1]:%d %b %Y}")
    a, b = st.columns(2)
    a.metric(
        "Latest adjusted close",
        f"{MARKETS[market_name]['symbol']}{stock_history.iloc[-1]:,.2f}",
    )
    b.metric("Historical CAGR", f"{metrics['CAGR']:.2%}")
    a, b = st.columns(2)
    a.metric("Beta", f"{metrics['Beta']:.2f}")
    b.metric("Annualized volatility", f"{metrics['Volatility']:.2%}")

    st.dataframe(pd.DataFrame({
        "Metric": [
            "Maximum drawdown", "Sharpe ratio", "Historical Jensen's alpha",
            "CAPM expected return", "Paired daily returns",
        ],
        "Value": [
            f"{metrics['Drawdown']:.2%}",
            f"{metrics['Sharpe']:.2f}",
            f"{metrics['Alpha']:.2%}",
            f"{metrics['CAPM']:.2%}",
            f"{metrics['Observations']:,}",
        ],
    }), hide_index=True, use_container_width=True)
    st.caption(
        f"CAPM assumes {expected_market_return:.2%} annual market return "
        f"using {assumption_source}; risk-free rate {risk_free_rate:.2%}."
    )

    st.plotly_chart(
        line_chart(prices[[ticker]], f"Adjusted close: {ticker}", "Adjusted price"),
        use_container_width=True,
    )
    comparison = normalized_prices(prices[[ticker, benchmark]])
    if not comparison.empty:
        st.plotly_chart(
            line_chart(
                comparison,
                f"Historical growth: {ticker} vs. {market_name}",
                "Index (starts at 100)",
            ),
            use_container_width=True,
        )
    else:
        st.info("Not enough common dates for a normalized growth chart.")

    daily_percent = stock_history.pct_change(fill_method=None).dropna() * 100
    if not daily_percent.empty:
        st.plotly_chart(
            line_chart(
                daily_percent.to_frame(name=ticker),
                f"Daily returns: {ticker}",
                "Daily return (%)",
            ),
            use_container_width=True,
        )


company_lists = load_company_lists()

tab_overview, tab_capm, tab_deep_dive = st.tabs([
    "Market Overview", "CAPM and Risk Analysis", "Stock Deep Dive"
])

with tab_overview:
    st.header("Market Overview")
    for container, market_name in zip(st.columns(len(MARKETS)), MARKETS):
        with container:
            st.subheader(market_name)
            benchmark = MARKETS[market_name]["benchmark"]
            try:
                history = get_prices((benchmark,), "1y")
                series = history[benchmark].dropna() if benchmark in history else pd.Series(dtype=float)
            except Exception as exc:
                st.warning(f"Benchmark unavailable: {exc}")
                series = pd.Series(dtype=float)

            if len(series) >= 2 and series.iloc[0] > 0:
                st.metric("Latest index level", f"{series.iloc[-1]:,.2f}")
                st.metric(
                    "Historical 1Y return",
                    f"{(series.iloc[-1] / series.iloc[0] - 1):.2%}",
                )
                st.caption(f"Last historical observation: {series.index[-1]:%d %b %Y}")
                st.plotly_chart(
                    line_chart(
                        series.to_frame(name=market_name),
                        f"{market_name}: 1-year index history", "Index level"
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Index history unavailable at the moment.")

            if market_name in company_lists:
                if st.button(
                    "Calculate Top 10 (1Y)", key=f"top_button_{market_name}"
                ):
                    try:
                        with st.spinner(f"Analyzing {market_name} constituents..."):
                            st.session_state[f"top_{market_name}"] = top_performers(market_name)
                    except Exception as exc:
                        st.error(f"Could not rank constituents: {exc}")

                if f"top_{market_name}" in st.session_state:
                    top, analyzed_count = st.session_state[f"top_{market_name}"]
                    if not top.empty:
                        st.dataframe(
                            top[["Company", "Ticker", "1Y Return (%)"]].style.format(
                                {"1Y Return (%)": "{:.2f}%"}
                            ),
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info("No companies have enough usable price history.")
                    st.caption(
                        f"Usable histories: {analyzed_count} / "
                        f"{len(company_lists[market_name])}. "
                        "Current constituents only; survivorship bias applies."
                    )

with tab_capm:
    st.header("CAPM and Risk Analysis")
    st.write(
        "Compare historical stock risk against a market benchmark and explore "
        "how assumed returns affect CAPM expectations."
    )

    st.subheader("Research Settings")
    market_col, period_col, risk_col = st.columns(3)
    with market_col:
        market_name = st.selectbox("Select Market", list(MARKETS), key="capm_market")
    with period_col:
        period = st.selectbox(
            "Historical Period", HISTORICAL_PERIODS,
            index=2, key="research_period"
        )
    with risk_col:
        risk_free_percent = st.number_input(
            "Annual Risk-Free Rate (%)", min_value=-99.0,
            value=0.0, step=0.25, key="research_risk_free"
        )
    risk_free_rate = risk_free_percent / 100
    st.caption(
        "The 0% risk-free rate is a placeholder, not a live rate. "
        "Enter an appropriate annual government-security yield for the market."
    )

    if st.button("Refresh Data", key="refresh_research_data"):
        get_companies.clear()
        get_prices.clear()
        top_performers.clear()
        for name in MARKETS:
            st.session_state.pop(f"top_{name}", None)
        st.rerun()

    st.divider()
    company_table = company_lists.get(market_name)
    if company_table is None or company_table.empty:
        st.warning("Company list unavailable. Refresh Data and try again.")
    else:
        benchmark = MARKETS[market_name]["benchmark"]
        choices = dict(zip(company_table["Company"], company_table["Ticker"]))
        selected_names = st.multiselect(
            "Select Companies", options=list(choices), key=f"capm_{market_name}"
        )
        selected_tickers = [choices[name] for name in selected_names]

        try:
            benchmark_data = get_prices((benchmark,), period)
            benchmark_series = (
                benchmark_data[benchmark].dropna()
                if benchmark in benchmark_data else pd.Series(dtype=float)
            )
        except Exception as exc:
            st.warning(f"Unable to load benchmark: {exc}")
            benchmark_series = pd.Series(dtype=float)

        historical_market_cagr = calculate_cagr(benchmark_series)
        if np.isfinite(historical_market_cagr):
            st.caption(
                f"Historical benchmark CAGR for the selected period: "
                f"{historical_market_cagr:.2%}."
            )
            default_expected_return = float(round(historical_market_cagr * 100, 2))
        else:
            st.warning("Historical benchmark CAGR unavailable; 0% is a placeholder.")
            default_expected_return = 0.0

        market_return_percent = st.number_input(
            "Expected Annual Market Return (%)",
            value=default_expected_return,
            step=0.5,
            key=f"market_return_{market_name}_{period}",
        )
        expected_market_return = market_return_percent / 100
        st.caption(
            "This is an editable scenario assumption. The initial value uses "
            "historical benchmark CAGR; past returns do not predict future returns."
        )

        if not selected_tickers:
            st.info("Choose one or more companies to calculate CAPM and risk metrics.")
        elif benchmark_series.empty:
            st.error("Cannot calculate beta without historical benchmark data.")
        else:
            try:
                all_prices = get_prices(tuple(selected_tickers + [benchmark]), period)
            except Exception as exc:
                st.error(f"Unable to download historical prices: {exc}")
                all_prices = pd.DataFrame()

            if all_prices.empty or benchmark not in all_prices:
                st.warning("Historical stock or benchmark data is unavailable.")
            else:
                records = []
                used_tickers = []
                for company in selected_names:
                    ticker = choices[company]
                    if ticker not in all_prices:
                        continue
                    metrics = calculate_metrics(
                        all_prices[ticker], all_prices[benchmark],
                        risk_free_rate, expected_market_return
                    )
                    if metrics is None:
                        continue
                    records.append({"Company": company, "Ticker": ticker, **metrics})
                    used_tickers.append(ticker)

                if not records:
                    st.warning(
                        "No selected companies have at least 30 paired daily returns "
                        "with a usable market benchmark."
                    )
                else:
                    result = pd.DataFrame(records)
                    missing = len(selected_names) - len(result)
                    if missing:
                        st.warning(f"Skipped {missing} selected company/companies with insufficient data.")

                    st.subheader("Calculated CAPM and Risk Metrics")
                    show_metrics_table(result)

                    st.subheader("Expected Return Analysis")
                    a, b, c = st.columns(3)
                    a.metric("Risk-free rate", f"{risk_free_rate:.2%}")
                    b.metric("Expected market return", f"{expected_market_return:.2%}")
                    c.metric(
                        "Assumed market risk premium",
                        f"{(expected_market_return - risk_free_rate):.2%}",
                    )

                    premiums = result[["Company", "Ticker", "CAPM"]].copy()
                    premiums["Stock risk premium"] = (
                        result["Beta"] * (expected_market_return - risk_free_rate)
                    )
                    premiums = premiums.rename(columns={"CAPM": "CAPM expected return"})
                    st.dataframe(
                        premiums.style.format({
                            "CAPM expected return": "{:.2%}",
                            "Stock risk premium": "{:.2%}",
                        }),
                        hide_index=True,
                        use_container_width=True,
                    )

                    st.subheader("Historical Stock Growth Comparison")
                    normalized = normalized_prices(
                        all_prices[used_tickers + [benchmark]]
                    )
                    if not normalized.empty:
                        st.plotly_chart(
                            line_chart(
                                normalized,
                                "Historical stock growth vs market benchmark",
                                "Index (starts at 100)",
                            ),
                            use_container_width=True,
                        )
                        st.caption(
                            "Each series starts at 100 on the first common date. "
                            "Changing expected returns does not change historical prices."
                        )
                    else:
                        st.info("Not enough overlapping dates for a fair growth comparison.")

                    show_projection(result, expected_market_return)

with tab_deep_dive:
    st.header("Stock Deep Dive")
    st.write(
        "Explore individual companies from either index using historical "
        "prices and market-relative risk metrics."
    )
    st.caption(
        f"Shared settings: {period} historical period, "
        f"{risk_free_rate:.2%} annual risk-free rate. "
        "Change these under CAPM and Risk Analysis."
    )
    for column, name in zip(st.columns(len(MARKETS)), MARKETS):
        with column:
            show_deep_dive(name, company_lists.get(name), period, risk_free_rate)

st.divider()
st.caption(
    "Sources: Yahoo Finance and public index constituent lists. "
    "Adjusted close may differ from a live market quote. "
    "Historical results, CAPM assumptions, and illustrative projections "
    "do not guarantee future performance."
)
