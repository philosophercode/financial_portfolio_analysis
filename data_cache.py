"""
Data caching system to avoid re-downloading stock data
"""

import os
import pandas as pd
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib


class DataCache:
    """
    Cache system for stock price data to avoid redundant API calls
    """

    def __init__(self, cache_dir: str = "data_cache"):
        """
        Initialize data cache

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, symbol: str, period: str, source: str) -> str:
        """Generate cache key for symbol/period/source combination"""
        key_string = f"{symbol}_{period}_{source}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def _is_cache_valid(self, cache_path: str, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid"""
        if not os.path.exists(cache_path):
            return False

        # Check file age
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        max_age = timedelta(hours=max_age_hours)

        return datetime.now() - file_time < max_age

    def get_cached_data(
        self, symbol: str, period: str, source: str = "yahoo"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and valid

        Args:
            symbol: Stock symbol
            period: Data period
            source: Data source

        Returns:
            Cached DataFrame or None if not available/valid
        """
        cache_key = self._get_cache_key(symbol, period, source)
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                print(f"📦 Using cached data for {symbol}")
                return cached_data
            except Exception as e:
                print(f"⚠️  Warning: Could not load cache for {symbol}: {e}")
                return None

        return None

    def cache_data(
        self, symbol: str, period: str, data: pd.DataFrame, source: str = "yahoo"
    ) -> None:
        """
        Cache data for future use

        Args:
            symbol: Stock symbol
            period: Data period
            data: DataFrame to cache
            source: Data source
        """
        if data is None or data.empty:
            return

        cache_key = self._get_cache_key(symbol, period, source)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"💾 Cached data for {symbol}")
        except Exception as e:
            print(f"⚠️  Warning: Could not cache data for {symbol}: {e}")

    def clear_cache(self, symbol: str = None) -> None:
        """
        Clear cached data

        Args:
            symbol: If provided, clear only this symbol's cache. Otherwise clear all.
        """
        if symbol:
            # Clear specific symbol (all periods/sources)
            for file in os.listdir(self.cache_dir):
                if file.startswith(symbol) and file.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                        print(f"🗑️  Cleared cache for {symbol}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not clear cache for {symbol}: {e}")
        else:
            # Clear all cache
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(self.cache_dir, file))
                    except Exception as e:
                        print(f"⚠️  Warning: Could not clear cache file {file}: {e}")
            print("🗑️  Cleared all cached data")

    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")]

        info = {"total_files": len(cache_files), "total_size_mb": 0, "files": []}

        for file in cache_files:
            file_path = os.path.join(self.cache_dir, file)
            try:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                info["total_size_mb"] += file_size
                info["files"].append(
                    {
                        "file": file,
                        "size_mb": round(file_size, 2),
                        "modified": file_time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            except Exception:
                continue

        info["total_size_mb"] = round(info["total_size_mb"], 2)
        return info


# Symbol mapping for common issues
SYMBOL_CORRECTIONS = {
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
    # Add more corrections as needed
}


def correct_symbol(symbol: str) -> str:
    """Correct common symbol formatting issues"""
    return SYMBOL_CORRECTIONS.get(symbol, symbol)
