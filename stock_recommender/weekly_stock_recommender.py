# -*- coding: utf-8 -*-
"""Weekly Stock Recommender

This script fetches weekly data for a universe of US-listed stocks and ETFs
related to AI, machine learning, quantum computing, software, data and
healthcare. It calculates several features including PE ratio, moving averages,
volatility and volume, then trains a machine learning model to predict the next
week's return. A backtest module evaluates the strategy using the Sharpe ratio.

The script is designed to run automatically every Monday to update
recommendations. Dependencies include ``yfinance``, ``pandas``, ``numpy`` and
``scikit-learn``.

Note: The provided universe of tickers is a small example. Extend
``TICKERS`` with a more comprehensive list to cover all eligible assets.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Example universe of tickers (extend as needed)
TICKERS: List[str] = [
    "AAPL",  # Apple - software/data
    "MSFT",  # Microsoft - software/AI
    "GOOGL",  # Alphabet - AI/data
    "AMZN",  # Amazon - AI/data
    "IBM",  # IBM - quantum computing/AI
    "NVDA",  # NVIDIA - AI hardware
    "XLV",  # Healthcare ETF
]

START_DATE = "2020-01-01"


@dataclass
class FeatureSet:
    pe_ratio: float
    sma20: float
    sma50: float
    sma200: float
    volatility: float
    volume: float


@dataclass
class PredictionResult:
    ticker: str
    prediction: float
    features: FeatureSet


def download_data(ticker: str) -> pd.DataFrame:
    """Download historical daily data for ``ticker`` and compute features."""
    df = yf.download(ticker, start=START_DATE, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    # Calculate daily returns
    df["Return"] = df["Adj Close"].pct_change()

    # Moving averages
    df["SMA20"] = df["Adj Close"].rolling(window=20).mean()
    df["SMA50"] = df["Adj Close"].rolling(window=50).mean()
    df["SMA200"] = df["Adj Close"].rolling(window=200).mean()

    # Volatility (30-day standard deviation of returns annualised)
    df["Volatility"] = df["Return"].rolling(window=30).std() * np.sqrt(252)

    # Weekly resample
    weekly = df.resample("W-FRI").last()
    weekly["Volume"] = df["Volume"].resample("W-FRI").sum()
    weekly["Return"] = weekly["Adj Close"].pct_change()
    weekly.dropna(inplace=True)

    return weekly


def fetch_pe_ratio(ticker: str) -> float:
    """Fetch trailing PE ratio using yfinance."""
    info = yf.Ticker(ticker).info
    return info.get("trailingPE", np.nan)


def build_feature_matrix() -> pd.DataFrame:
    """Build a single DataFrame containing features for all tickers."""
    frames = []
    for ticker in TICKERS:
        data = download_data(ticker)
        data["Ticker"] = ticker
        data["PE"] = fetch_pe_ratio(ticker)
        frames.append(data)
    return pd.concat(frames)


def prepare_ml_data(df: pd.DataFrame):
    """Prepare features ``X`` and target ``y``."""
    features = ["PE", "SMA20", "SMA50", "SMA200", "Volatility", "Volume"]
    X = df[features]
    y = df["Return"].shift(-1)  # next week's return
    df = df.dropna(subset=features + ["Return"])
    X = df[features]
    y = df["Return"].shift(-1).dropna()
    X = X.loc[y.index]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """Train a RandomForestRegressor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Model MSE: {mse:.6f}")
    return model


def predict_next_week(df: pd.DataFrame, model: RandomForestRegressor) -> List[PredictionResult]:
    """Predict next week's return for each ticker."""
    results = []
    for ticker in TICKERS:
        subset = df[df["Ticker"] == ticker].iloc[-1]
        features = FeatureSet(
            pe_ratio=subset["PE"],
            sma20=subset["SMA20"],
            sma50=subset["SMA50"],
            sma200=subset["SMA200"],
            volatility=subset["Volatility"],
            volume=subset["Volume"],
        )
        X_new = pd.DataFrame(
            [[
                features.pe_ratio,
                features.sma20,
                features.sma50,
                features.sma200,
                features.volatility,
                features.volume,
            ]],
            columns=["PE", "SMA20", "SMA50", "SMA200", "Volatility", "Volume"],
        )
        prediction = float(model.predict(X_new)[0])
        results.append(PredictionResult(ticker, prediction, features))
    return sorted(results, key=lambda r: r.prediction, reverse=True)


def print_top_recommendations(results: List[PredictionResult], top_n: int = 5) -> None:
    """Print a table of top predicted tickers."""
    table_data = [
        [
            r.ticker,
            f"{r.prediction:.4f}",
            f"{r.features.pe_ratio:.2f}",
            f"{r.features.sma20:.2f}",
            f"{r.features.sma50:.2f}",
            f"{r.features.sma200:.2f}",
            f"{r.features.volatility:.4f}",
        ]
        for r in results[:top_n]
    ]
    columns = [
        "Ticker",
        "PredictedReturn",
        "PE",
        "SMA20",
        "SMA50",
        "SMA200",
        "Volatility",
    ]
    df = pd.DataFrame(table_data, columns=columns)
    print("Top Recommendations:\n", df.to_string(index=False))
    output_path = os.path.join(os.path.dirname(__file__), "weekly_recommendations.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def backtest(df: pd.DataFrame, model: RandomForestRegressor) -> None:
    """Run a simple backtest and compute the Sharpe ratio."""
    df = df.copy()
    df["Prediction"] = model.predict(df[["PE", "SMA20", "SMA50", "SMA200", "Volatility", "Volume"]])

    df.sort_index(inplace=True)
    weekly_dates = sorted(set(df.index))

    returns = []
    for date in weekly_dates:
        week_data = df.loc[date]
        top = week_data.sort_values("Prediction", ascending=False).head(5)
        actual_return = top["Return"].mean()
        returns.append(actual_return)

    returns = pd.Series(returns)
    sharpe = returns.mean() / returns.std(ddof=0) * np.sqrt(52)  # weekly Sharpe
    print(f"Backtest Sharpe ratio: {sharpe:.4f}")


def run() -> None:
    """Fetch data, train model, and output recommendations."""
    df = build_feature_matrix()
    X, y = prepare_ml_data(df)
    model = train_model(X, y)
    results = predict_next_week(df, model)
    print_top_recommendations(results)
    backtest(df, model)


if __name__ == "__main__":
    run()
