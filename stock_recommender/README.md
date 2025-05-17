# Weekly Stock Recommender

This module provides a script (`weekly_stock_recommender.py`) to identify the
five best performing US stocks or ETFs in AI, machine learning, quantum
computing, software, data and healthcare on a weekly basis.

The script downloads historical data from Yahoo Finance via `yfinance`,
calculates several features (PE ratio, moving averages, volatility and volume),
and trains a `RandomForestRegressor` model to predict next week's return. A
simple backtest computes the Sharpe ratio of the strategy. Results are printed
in a table and saved as `weekly_recommendations.csv`.

## Requirements

```
pandas
numpy
scikit-learn
yfinance
```

Install the requirements with:

```
pip install pandas numpy scikit-learn yfinance
```

## Usage

Run the script manually:

```
python weekly_stock_recommender.py
```

To schedule automatic execution every Monday, run the script with a scheduling
tool such as `cron` or Python's `schedule` library. See the code comments for
details.

Extend the `TICKERS` list in the script to include all relevant US stocks or
ETFs you want considered.
