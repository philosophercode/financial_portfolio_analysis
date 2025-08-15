"""
Data fetcher module for retrieving stock price data from Alpha Vantage API and Yahoo Finance
"""

import pandas as pd
import requests
import time
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import config


class DataFetcher:
    """
    A class to fetch financial data from multiple sources
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DataFetcher

        Args:
            api_key: Alpha Vantage API key. If None, will use environment variable
        """
        self.api_key = api_key or config.ALPHA_VANTAGE_API_KEY
        self.base_url = config.ALPHA_VANTAGE_BASE_URL

    def get_stock_data_alpha_vantage(
        self, symbol: str, outputsize: str = "full"
    ) -> pd.DataFrame:
        """
        Fetch daily stock data from Alpha Vantage API

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            outputsize: 'compact' for last 100 days, 'full' for full history

        Returns:
            DataFrame with stock price data
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            if "Note" in data:
                raise ValueError(f"Alpha Vantage API limit reached: {data['Note']}")

            # Extract time series data
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                raise ValueError(f"No data found for symbol {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Rename columns and convert to numeric
            df.columns = [
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "dividend",
                "split",
            ]
            df = df.astype(float)

            return df

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch data for {symbol}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing data for {symbol}: {e}")

    def get_stock_data_yahoo(self, symbol: str, period: str = None) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance (as backup/alternative)

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Period for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            DataFrame with stock price data
        """
        try:
            if period is None:
                period = config.DEFAULT_PERIOD
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Standardize column names
            df.columns = df.columns.str.lower()
            if "adj close" in df.columns:
                df["adjusted_close"] = df["adj close"]
                df = df.drop("adj close", axis=1)

            return df

        except Exception as e:
            raise RuntimeError(
                f"Error fetching data from Yahoo Finance for {symbol}: {e}"
            )

    def get_multiple_stocks(
        self, symbols: List[str], source: str = "yahoo", **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data for multiple stocks and return adjusted close prices

        Args:
            symbols: List of stock symbols
            source: Data source ('yahoo' or 'alpha_vantage')
            **kwargs: Additional arguments for the data fetcher

        Returns:
            DataFrame with adjusted close prices for all symbols
        """
        price_data = {}
        failed_symbols = []

        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol}...")

                if source == "yahoo":
                    df = self.get_stock_data_yahoo(symbol, **kwargs)
                    price_data[symbol] = (
                        df["adjusted_close"]
                        if "adjusted_close" in df.columns
                        else df["close"]
                    )
                elif source == "alpha_vantage":
                    df = self.get_stock_data_alpha_vantage(symbol, **kwargs)
                    price_data[symbol] = df["adjusted_close"]
                    # Alpha Vantage has rate limits, so we need to wait
                    time.sleep(12)  # Free tier allows 5 calls per minute
                else:
                    raise ValueError(f"Unknown data source: {source}")

            except Exception as e:
                print(f"Failed to fetch data for {symbol}: {e}")
                failed_symbols.append(symbol)
                continue

        if not price_data:
            raise ValueError("No data could be fetched for any symbols")

        if failed_symbols:
            print(f"Warning: Failed to fetch data for symbols: {failed_symbols}")

        # Combine all price data
        combined_df = pd.DataFrame(price_data)

        # Remove rows with any NaN values
        combined_df = combined_df.dropna()

        if combined_df.empty:
            raise ValueError("No overlapping data found for the selected symbols")

        return combined_df

    def get_company_overview(self, symbol: str) -> Dict:
        """
        Get company overview data from Alpha Vantage

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company information
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")

        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": self.api_key}

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")

            return data

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch company overview for {symbol}: {e}")


def validate_symbols(symbols: List[str]) -> List[str]:
    """
    Validate and clean stock symbols

    Args:
        symbols: List of stock symbols

    Returns:
        List of cleaned symbols
    """
    cleaned_symbols = []
    for symbol in symbols:
        # Remove whitespace and convert to uppercase
        clean_symbol = symbol.strip().upper()
        if clean_symbol and len(clean_symbol) <= 10:  # Basic validation
            cleaned_symbols.append(clean_symbol)
        else:
            print(f"Warning: Invalid symbol format: {symbol}")

    return cleaned_symbols
