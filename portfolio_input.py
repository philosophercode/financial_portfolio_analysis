"""
Portfolio Input Handler - Load portfolio from various input formats
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from data_fetcher import DataFetcher
import config


class PortfolioInputHandler:
    """
    Handle different portfolio input formats and convert to standardized format
    """

    def __init__(self, data_fetcher: DataFetcher = None):
        """
        Initialize the input handler

        Args:
            data_fetcher: DataFetcher instance for getting current prices
        """
        self.data_fetcher = data_fetcher or DataFetcher()
        self.current_prices = {}

    def load_portfolio_from_file(
        self, file_path: str
    ) -> Tuple[List[str], Dict[str, float], float]:
        """
        Load portfolio from CSV file

        Args:
            file_path: Path to CSV file with portfolio data

        Returns:
            Tuple of (symbols, weights, total_value)
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Detect the format based on columns
            if self._is_shares_format(df):
                return self._process_shares_format(df)
            elif self._is_weights_format(df):
                return self._process_weights_format(df)
            else:
                raise ValueError(
                    "Unknown CSV format. Expected columns: 'symbol,shares' or 'symbol,weight'"
                )

        except Exception as e:
            raise RuntimeError(f"Error loading portfolio from {file_path}: {e}")

    def create_portfolio_from_dict(
        self,
        portfolio_dict: Dict[str, Union[int, float]],
        input_type: str = "auto",
        total_value: float = None,
    ) -> Tuple[List[str], Dict[str, float], float]:
        """
        Create portfolio from dictionary input

        Args:
            portfolio_dict: Dictionary with symbol -> value mapping
            input_type: "shares", "weights", or "auto" to detect
            total_value: Total portfolio value (for shares format)

        Returns:
            Tuple of (symbols, weights, total_value)
        """
        symbols = list(portfolio_dict.keys())

        if input_type == "auto":
            input_type = self._detect_input_type(portfolio_dict)

        if input_type == "shares":
            if total_value is None:
                total_value = self._calculate_total_value_from_shares(portfolio_dict)
            weights = self._shares_to_weights(portfolio_dict, total_value)
        elif input_type == "weights":
            weights = self._normalize_weights(portfolio_dict)
            total_value = total_value or config.DEFAULT_PORTFOLIO_VALUE
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        return symbols, weights, total_value

    def _is_shares_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in shares format"""
        required_cols = ["symbol", "shares"]
        return all(
            col.lower() in [c.lower() for c in df.columns] for col in required_cols
        )

    def _is_weights_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in weights format"""
        required_cols = ["symbol", "weight"]
        return all(
            col.lower() in [c.lower() for c in df.columns] for col in required_cols
        )

    def _process_shares_format(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], Dict[str, float], float]:
        """Process shares format: symbol, shares"""
        # Standardize column names
        df.columns = df.columns.str.lower()

        # Clean symbols
        df["symbol"] = df["symbol"].str.upper().str.strip()

        # Convert to dictionary
        shares_dict = dict(zip(df["symbol"], df["shares"]))

        # Calculate total value and weights
        total_value = self._calculate_total_value_from_shares(shares_dict)
        weights = self._shares_to_weights(shares_dict, total_value)
        symbols = list(shares_dict.keys())

        return symbols, weights, total_value

    def _process_weights_format(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], Dict[str, float], float]:
        """Process weights format: symbol, weight"""
        # Standardize column names
        df.columns = df.columns.str.lower()

        # Clean symbols
        df["symbol"] = df["symbol"].str.upper().str.strip()

        # Convert to dictionary
        weights_dict = dict(zip(df["symbol"], df["weight"]))

        # Normalize weights
        weights = self._normalize_weights(weights_dict)
        symbols = list(weights_dict.keys())

        # Use default portfolio value
        total_value = config.DEFAULT_PORTFOLIO_VALUE

        return symbols, weights, total_value

    def _detect_input_type(self, portfolio_dict: Dict[str, Union[int, float]]) -> str:
        """Auto-detect if input is shares or weights"""
        values = list(portfolio_dict.values())

        # If all values are integers and some are > 1, likely shares
        if all(isinstance(v, int) or v.is_integer() for v in values):
            if any(v > 1 for v in values):
                return "shares"

        # If values sum to approximately 1, likely weights
        total = sum(values)
        if 0.95 <= total <= 1.05:
            return "weights"

        # If values are all < 1, likely weights
        if all(v <= 1 for v in values):
            return "weights"

        # Default to shares
        return "shares"

    def _calculate_total_value_from_shares(self, shares_dict: Dict[str, int]) -> float:
        """Calculate total portfolio value from shares"""
        # Get current prices
        symbols = list(shares_dict.keys())
        self._fetch_current_prices(symbols)

        total_value = 0
        for symbol, shares in shares_dict.items():
            if symbol in self.current_prices:
                total_value += shares * self.current_prices[symbol]
            else:
                print(
                    f"Warning: Could not get price for {symbol}, skipping from total calculation"
                )

        return total_value

    def _shares_to_weights(
        self, shares_dict: Dict[str, int], total_value: float
    ) -> Dict[str, float]:
        """Convert shares to portfolio weights"""
        symbols = list(shares_dict.keys())
        self._fetch_current_prices(symbols)

        weights = {}
        for symbol, shares in shares_dict.items():
            if symbol in self.current_prices:
                position_value = shares * self.current_prices[symbol]
                weights[symbol] = position_value / total_value
            else:
                print(f"Warning: Could not get price for {symbol}, setting weight to 0")
                weights[symbol] = 0

        return weights

    def _normalize_weights(self, weights_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1"""
        total = sum(weights_dict.values())
        if total == 0:
            raise ValueError("Total weights cannot be zero")

        return {symbol: weight / total for symbol, weight in weights_dict.items()}

    def _fetch_current_prices(self, symbols: List[str]) -> None:
        """Fetch current prices for symbols"""
        for symbol in symbols:
            if symbol not in self.current_prices:
                try:
                    # Get recent price data (just need latest price)
                    data = self.data_fetcher.get_stock_data_yahoo(symbol, period="5d")
                    if not data.empty:
                        latest_price = data["close"].iloc[-1]
                        self.current_prices[symbol] = latest_price
                        print(f"📈 {symbol}: ${latest_price:.2f}")
                    else:
                        print(f"❌ Could not fetch price for {symbol}")
                except Exception as e:
                    print(f"❌ Error fetching price for {symbol}: {e}")

    def save_example_files(self):
        """Create example input files for reference"""
        # Example shares format
        shares_example = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "shares": [50, 30, 10, 15, 25],
            }
        )
        shares_example.to_csv("example_portfolio_shares.csv", index=False)

        # Example weights format
        weights_example = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "weight": [0.25, 0.20, 0.20, 0.15, 0.20],
            }
        )
        weights_example.to_csv("example_portfolio_weights.csv", index=False)

        print("📁 Created example files:")
        print("  - example_portfolio_shares.csv (symbol, shares)")
        print("  - example_portfolio_weights.csv (symbol, weight)")


def load_and_analyze_portfolio(
    file_path: str,
    optimization_method: str = "max_sharpe",
    show_plots: bool = True,
    save_results: bool = True,
) -> Dict:
    """
    Load portfolio from file and run complete analysis

    Args:
        file_path: Path to portfolio CSV file
        optimization_method: Optimization method to use
        show_plots: Whether to show plots
        save_results: Whether to save results

    Returns:
        Dictionary with analysis results
    """
    from main import analyze_portfolio

    print("📁 Loading portfolio from file...")

    # Load portfolio
    handler = PortfolioInputHandler()
    symbols, weights, total_value = handler.load_portfolio_from_file(file_path)

    print(f"📊 Loaded portfolio:")
    print(f"   Symbols: {symbols}")
    print(f"   Total Value: ${total_value:,.2f}")
    print(f"   Current Allocation:")
    for symbol, weight in weights.items():
        print(f"     {symbol}: {weight:.1%}")

    # Run analysis
    results = analyze_portfolio(
        symbols=symbols,
        portfolio_value=total_value,
        optimization_method=optimization_method,
        show_plots=show_plots,
        save_results=save_results,
        original_weights=weights,
    )

    # Add original portfolio info to results
    results["original_portfolio"] = {"weights": weights, "total_value": total_value}

    return results


def compare_portfolios(
    current_file: str, optimization_method: str = "max_sharpe", show_plots: bool = True
) -> Dict:
    """
    Compare current portfolio with optimized version

    Args:
        current_file: Path to current portfolio CSV
        optimization_method: Optimization method for comparison
        show_plots: Whether to show comparison plots

    Returns:
        Comparison results
    """
    from portfolio_analyzer import PortfolioAnalyzer
    from visualizer import PortfolioVisualizer
    import matplotlib.pyplot as plt

    print("🔍 Comparing Current vs Optimized Portfolio")
    print("=" * 50)

    # Load current portfolio
    handler = PortfolioInputHandler()
    symbols, current_weights, total_value = handler.load_portfolio_from_file(
        current_file
    )

    # Get price data
    fetcher = DataFetcher()
    price_data = fetcher.get_multiple_stocks(
        symbols, source="yahoo", period=config.DEFAULT_PERIOD
    )

    # Analyze current portfolio
    analyzer = PortfolioAnalyzer(price_data)
    current_performance = analyzer.get_portfolio_performance(current_weights)
    current_risk_metrics = analyzer.get_risk_metrics(current_weights)

    # Get optimized portfolio
    optimized_weights = analyzer.optimize_portfolio(objective=optimization_method)
    optimized_performance = analyzer.get_portfolio_performance(optimized_weights)
    optimized_risk_metrics = analyzer.get_risk_metrics(optimized_weights)

    # Comparison results
    comparison = {
        "current": {
            "weights": current_weights,
            "performance": current_performance,
            "risk_metrics": current_risk_metrics,
        },
        "optimized": {
            "weights": optimized_weights,
            "performance": optimized_performance,
            "risk_metrics": optimized_risk_metrics,
        },
    }

    # Print comparison
    print(f"\n📊 PORTFOLIO COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<25} {'Current':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 70)

    current_ret, current_vol, current_sharpe = current_performance
    opt_ret, opt_vol, opt_sharpe = optimized_performance

    print(
        f"{'Expected Return':<25} {current_ret:<14.2%} {opt_ret:<14.2%} {opt_ret-current_ret:<14.2%}"
    )
    print(
        f"{'Volatility':<25} {current_vol:<14.2%} {opt_vol:<14.2%} {opt_vol-current_vol:<14.2%}"
    )
    print(
        f"{'Sharpe Ratio':<25} {current_sharpe:<14.3f} {opt_sharpe:<14.3f} {opt_sharpe-current_sharpe:<14.3f}"
    )

    if show_plots:
        visualizer = PortfolioVisualizer()

        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Current portfolio weights
        current_clean = analyzer.clean_weights(current_weights)
        if current_clean:
            ax1.pie(
                current_clean.values(), labels=current_clean.keys(), autopct="%1.1f%%"
            )
            ax1.set_title("Current Portfolio")

        # Optimized portfolio weights
        opt_clean = analyzer.clean_weights(optimized_weights)
        if opt_clean:
            ax2.pie(opt_clean.values(), labels=opt_clean.keys(), autopct="%1.1f%%")
            ax2.set_title("Optimized Portfolio")

        # Performance comparison
        metrics = ["Return", "Volatility", "Sharpe"]
        current_vals = [current_ret, current_vol, current_sharpe]
        opt_vals = [opt_ret, opt_vol, opt_sharpe]

        x = range(len(metrics))
        width = 0.35
        ax3.bar(
            [i - width / 2 for i in x], current_vals, width, label="Current", alpha=0.7
        )
        ax3.bar(
            [i + width / 2 for i in x], opt_vals, width, label="Optimized", alpha=0.7
        )
        ax3.set_xlabel("Metrics")
        ax3.set_title("Performance Comparison")
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()

        # Risk comparison
        risk_keys = ["value_at_risk_95", "max_drawdown", "downside_deviation"]
        current_risks = [abs(current_risk_metrics.get(k, 0)) for k in risk_keys]
        opt_risks = [abs(optimized_risk_metrics.get(k, 0)) for k in risk_keys]

        x = range(len(risk_keys))
        ax4.bar(
            [i - width / 2 for i in x], current_risks, width, label="Current", alpha=0.7
        )
        ax4.bar(
            [i + width / 2 for i in x], opt_risks, width, label="Optimized", alpha=0.7
        )
        ax4.set_xlabel("Risk Metrics")
        ax4.set_title("Risk Comparison")
        ax4.set_xticks(x)
        ax4.set_xticklabels(["VaR 95%", "Max Drawdown", "Downside Dev"])
        ax4.legend()

        plt.tight_layout()
        plt.show()

    return comparison
