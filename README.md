# Stock Market Analytics Dashboard

A stock market analysis dashboard built with Python and Streamlit to explore historical stock performance, understand market risk, and calculate expected returns using the Capital Asset Pricing Model (CAPM).

The dashboard covers two major markets: the **S&P 500 (US)** and **NIFTY 50 (India)**. It allows users to select companies, compare their performance against market benchmarks, and explore how different financial assumptions affect expected returns.

## Features

### 1. Market Overview
- View historical performance of the S&P 500 and NIFTY 50 indices.
- Track the latest available index level and historical one-year return.
- Explore interactive index performance charts.
- Identify the top 10 performing stocks over the past year based on available historical data.

### 2. CAPM and Risk Analysis
- Select multiple companies from either the US or Indian market.
- Choose historical periods ranging from six months to five years.
- Adjust the annual risk-free rate and expected market return.
- Calculate and compare:
  - Beta
  - Compound Annual Growth Rate (CAGR)
  - Annualized volatility
  - Maximum drawdown
  - Sharpe ratio
  - Historical Jensen's alpha
  - CAPM expected return
- Compare historical stock growth against the selected market benchmark.
- Explore illustrative CAPM growth scenarios over a selected number of years.

### 3. Stock Deep Dive
- Analyze individual companies from the S&P 500 or NIFTY 50.
- View the latest available adjusted closing price.
- Examine historical returns and risk metrics.
- Compare a company's historical growth against its market index.
- Visualize daily percentage returns using interactive charts.

## Technologies Used

- **Python:** Core application and financial calculations
- **Streamlit:** Interactive dashboard and user interface
- **Pandas:** Data cleaning, processing, and analysis
- **NumPy:** Numerical and statistical calculations
- **Plotly:** Interactive financial charts
- **yfinance:** Historical stock market data
- **Requests:** Retrieving company lists from public CSV sources

## Project Structure

```text
stock-market-analytics/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```


## Data Sources

The application uses publicly available market data.

- **S&P 500 company list:** Public GitHub CSV dataset maintained by the `datasets` project.
- **NIFTY 50 company list:** NSE constituent CSV.
- **Historical stock prices:** Yahoo Finance through the `yfinance` Python library.
- **Market benchmarks:** `^GSPC` for the S&P 500 and `^NSEI` for NIFTY 50.

Company lists are cached for 24 hours, while historical prices are cached for one hour to reduce repeated requests.

The dashboard uses adjusted historical closing prices rather than a real-time streaming market feed. Data availability depends on the external providers.

## How CAPM Works

The Capital Asset Pricing Model estimates the expected return of a stock based on its sensitivity to the overall market.

**CAPM Formula:**

Expected Return = Risk-Free Rate + Beta × (Expected Market Return − Risk-Free Rate)

Users can change the risk-free rate and expected market return to explore different scenarios.

Beta is calculated from historical daily stock and benchmark returns. Other historical metrics, such as CAGR and volatility, are calculated independently of the expected market return.

The dashboard also includes an illustrative growth projection based on CAPM expected returns. This assumes a constant annual return and should not be treated as a prediction of future stock prices.

## What I Learned

Building this project helped me improve my understanding of financial data analysis and Python application development.

I worked with external data sources, cleaned and aligned historical stock prices, implemented financial calculations, and built interactive charts to make the results easier to understand.

One important part of the project was separating historical performance from expected-return assumptions. Changing an expected market return should affect CAPM estimates, but it should not change actual historical stock prices.

I also focused on making the code easier to maintain by using reusable functions, caching downloaded data, and handling missing or incomplete data.

## Limitations

- Market data is retrieved from external sources and may occasionally be unavailable or delayed.
- CAPM expected returns depend on the assumptions entered by the user.
- Historical performance does not guarantee future results.
- The growth projection assumes constant annual returns and does not account for future market volatility.
- The top-performing stock rankings use current index constituents, which introduces survivorship bias.
- The dashboard is intended for financial analysis and learning, not automated trading.

## Live Dashboard

**Live Application:** Add your Streamlit deployment URL here.

## Future Improvements

I plan to explore additional features such as portfolio analysis, sector-level comparisons, CSV exports, and more advanced risk visualizations.

## Disclaimer

This project is built for educational and research purposes. The information presented should not be considered financial or investment advice.