"""
Portfolio analysis module using PyPortfolioOpt for modern portfolio theory optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pypfopt.expected_returns import (
    mean_historical_return,
    ema_historical_return,
    capm_return,
)
from pypfopt.risk_models import CovarianceShrinkage, sample_cov, semicovariance, exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import objective_functions
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.cla import CLA
import config


class PortfolioAnalyzer:
    """
    A comprehensive portfolio analysis class using modern portfolio theory
    """

    def __init__(self, price_data: pd.DataFrame, risk_free_rate: float = None):
        """
        Initialize the portfolio analyzer

        Args:
            price_data: DataFrame with stock prices (columns = symbols, index = dates)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.price_data = price_data.copy()
        self.risk_free_rate = risk_free_rate or config.DEFAULT_RISK_FREE_RATE
        self.symbols = list(price_data.columns)

        # Calculate returns
        self.returns = price_data.pct_change().dropna()
        
        # Update symbols list to match actual data columns
        self.symbols = list(self.price_data.columns)

        # Initialize expected returns and covariance matrix
        self.expected_returns = None
        self.cov_matrix = None
        self.weights = None
        self.ef = None

    def calculate_expected_returns(self, method: str = None) -> pd.Series:
        """
        Calculate expected returns using various methods

        Args:
            method: Method to use ('mean_historical', 'ema_historical', 'ewma', 'capm')

        Returns:
            Series of expected returns for each asset
        """
        if method is None:
            method = config.RETURN_MODEL

        if method == "mean_historical" or method == "mean":
            self.expected_returns = mean_historical_return(
                self.price_data, frequency=config.DEFAULT_FREQUENCY
            )
        elif method == "ema_historical" or method == "ewma":
            # Use EWMA with span around 60-90 trading days as recommended
            self.expected_returns = ema_historical_return(
                self.price_data, span=75, frequency=config.DEFAULT_FREQUENCY
            )
        elif method == "capm":
            # For CAPM, we need a market return (using SPY as proxy)
            try:
                from data_fetcher import DataFetcher

                fetcher = DataFetcher()
                spy_data = fetcher.get_stock_data_yahoo("SPY", period="2y")
                market_prices = (
                    spy_data["adjusted_close"]
                    if "adjusted_close" in spy_data.columns
                    else spy_data["close"]
                )
                # Align dates with our price data
                market_prices = market_prices.reindex(self.price_data.index).dropna()
                common_dates = self.price_data.index.intersection(market_prices.index)
                aligned_prices = self.price_data.loc[common_dates]
                aligned_market = market_prices.loc[common_dates]
                self.expected_returns = capm_return(
                    aligned_prices,
                    market_prices=aligned_market,
                    risk_free_rate=self.risk_free_rate,
                )
            except Exception as e:
                print(
                    f"Error calculating CAPM returns, falling back to mean historical: {e}"
                )
                self.expected_returns = mean_historical_return(self.price_data)
        else:
            raise ValueError(f"Unknown expected returns method: {method}")

        return self.expected_returns

    def calculate_risk_model(self, method: str = None) -> pd.DataFrame:
        """
        Calculate risk model (covariance matrix) using various methods

        Args:
            method: Method to use ('sample', 'ledoit_wolf', 'oas', 'lw_constant_variance', 'semicovariance', 'semicov', 'exponential')

        Returns:
            Covariance matrix
        """
        if method is None:
            method = config.COVARIANCE_MODEL

        if method == "sample":
            self.cov_matrix = sample_cov(
                self.price_data, frequency=config.DEFAULT_FREQUENCY
            )
        elif method == "ledoit_wolf":
            shrinkage = CovarianceShrinkage(
                self.price_data, frequency=config.DEFAULT_FREQUENCY
            )
            self.cov_matrix = shrinkage.ledoit_wolf()
        elif method == "oas":
            shrinkage = CovarianceShrinkage(
                self.price_data, frequency=config.DEFAULT_FREQUENCY
            )
            self.cov_matrix = shrinkage.oracle_approximating()
        elif method == "lw_constant_variance":
            shrinkage = CovarianceShrinkage(
                self.price_data, frequency=config.DEFAULT_FREQUENCY
            )
            self.cov_matrix = shrinkage.ledoit_wolf(
                shrinkage_target="constant_variance"
            )
        elif method == "semicovariance" or method == "semicov":
            self.cov_matrix = semicovariance(
                self.price_data, frequency=config.DEFAULT_FREQUENCY
            )
        elif method == "exponential":
            self.cov_matrix = exp_cov(
                self.price_data, frequency=config.DEFAULT_FREQUENCY
            )
        else:
            raise ValueError(f"Unknown risk model method: {method}")

        return self.cov_matrix

    def optimize_portfolio(
        self,
        objective: str = "max_sharpe",
        weight_bounds: Tuple[float, float] = None,
        target_return: float = None,
        target_volatility: float = None,
        l2_gamma: float = None,
        market_neutral: bool = False,
    ) -> Dict[str, float]:
        """
        Optimize portfolio using mean-variance optimization

        Args:
            objective: Optimization objective ('max_sharpe', 'min_volatility', 'efficient_return', 'efficient_risk')
            weight_bounds: Tuple of (min_weight, max_weight)
            target_return: Target return for efficient_return objective
            target_volatility: Target volatility for efficient_risk objective
            l2_gamma: L2 regularization parameter for diversification
            market_neutral: Whether to create market neutral portfolio (weights sum to 0)

        Returns:
            Dictionary of optimized weights
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.cov_matrix is None:
            self.calculate_risk_model()

        # Set default weight bounds
        if weight_bounds is None:
            weight_bounds = config.DEFAULT_WEIGHT_BOUNDS

        # Initialize efficient frontier
        self.ef = EfficientFrontier(
            self.expected_returns, self.cov_matrix, weight_bounds=weight_bounds
        )

        # Add L2 regularization (use default if not specified)
        if l2_gamma is None:
            l2_gamma = config.DEFAULT_L2_REG
        if l2_gamma > 0:
            self.ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)

        # Add maximum weight constraint if specified in config
        if config.MAX_WEIGHT < 1.0:
            # Use weight bounds instead of constraints for maximum weight
            max_bound = min(weight_bounds[1], config.MAX_WEIGHT)
            weight_bounds = (weight_bounds[0], max_bound)
            # Reinitialize with updated bounds
            self.ef = EfficientFrontier(
                self.expected_returns, self.cov_matrix, weight_bounds=weight_bounds
            )
            # Re-add L2 regularization after reinitialization
            if l2_gamma > 0:
                self.ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)

        # Optimize based on objective
        if objective == "max_sharpe":
            self.weights = self.ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        elif objective == "min_volatility":
            self.weights = self.ef.min_volatility()
        elif objective == "efficient_return":
            if target_return is None:
                raise ValueError(
                    "target_return must be specified for efficient_return objective"
                )
            self.weights = self.ef.efficient_return(
                target_return, market_neutral=market_neutral
            )
        elif objective == "efficient_risk":
            if target_volatility is None:
                raise ValueError(
                    "target_volatility must be specified for efficient_risk objective"
                )
            self.weights = self.ef.efficient_risk(
                target_volatility, market_neutral=market_neutral
            )
        else:
            raise ValueError(f"Unknown optimization objective: {objective}")

        return self.weights

    def hierarchical_risk_parity(self) -> Dict[str, float]:
        """
        Optimize portfolio using Hierarchical Risk Parity (HRP)

        Returns:
            Dictionary of HRP weights
        """
        hrp = HRPOpt(self.returns)
        self.weights = hrp.optimize()
        return self.weights

    def critical_line_algorithm(self, target_return: float = None) -> Dict[str, float]:
        """
        Optimize portfolio using Critical Line Algorithm (CLA)

        Args:
            target_return: Target return for optimization

        Returns:
            Dictionary of CLA weights
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.cov_matrix is None:
            self.calculate_risk_model()

        cla = CLA(self.expected_returns, self.cov_matrix)

        if target_return is not None:
            self.weights = cla.efficient_frontier(target_return)
        else:
            self.weights = cla.max_sharpe()

        return self.weights

    def get_portfolio_performance(
        self, weights: Dict[str, float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics

        Args:
            weights: Portfolio weights (if None, uses last optimized weights)

        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("No weights available. Run optimization first.")

        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.cov_matrix is None:
            self.calculate_risk_model()

        # Convert weights to numpy array in correct order
        weight_array = np.array([weights.get(symbol, 0) for symbol in self.symbols])

        # Calculate performance
        expected_return = np.dot(weight_array, self.expected_returns)
        portfolio_variance = np.dot(weight_array, np.dot(self.cov_matrix, weight_array))
        volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility

        return expected_return, volatility, sharpe_ratio

    def get_discrete_allocation(
        self, weights: Dict[str, float] = None, total_portfolio_value: float = None
    ) -> Tuple[Dict[str, int], float]:
        """
        Convert portfolio weights to discrete share allocation

        Args:
            weights: Portfolio weights (if None, uses last optimized weights)
            total_portfolio_value: Total value of portfolio in dollars

        Returns:
            Tuple of (allocation_dict, leftover_cash)
        """
        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("No weights available. Run optimization first.")

        if total_portfolio_value is None:
            total_portfolio_value = config.DEFAULT_PORTFOLIO_VALUE

        # Get latest prices
        latest_prices = get_latest_prices(self.price_data)

        # Calculate discrete allocation
        da = DiscreteAllocation(
            weights, latest_prices, total_portfolio_value=total_portfolio_value
        )
        allocation, leftover = da.lp_portfolio()

        return allocation, leftover

    def clean_weights(
        self, weights: Dict[str, float] = None, cutoff: float = 0.0001
    ) -> Dict[str, float]:
        """
        Clean weights by removing very small allocations

        Args:
            weights: Portfolio weights (if None, uses last optimized weights)
            cutoff: Minimum weight threshold

        Returns:
            Dictionary of cleaned weights
        """
        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("No weights available. Run optimization first.")

        cleaned = {}
        for symbol, weight in weights.items():
            if abs(weight) >= cutoff:
                cleaned[symbol] = round(weight, 5)

        return cleaned

    def calculate_efficient_frontier(
        self, num_portfolios: int = 100, ret_range: Tuple[float, float] = None
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate the efficient frontier

        Args:
            num_portfolios: Number of portfolios to calculate
            ret_range: Tuple of (min_return, max_return) for frontier

        Returns:
            Tuple of (returns, volatilities, sharpe_ratios)
        """
        if self.expected_returns is None:
            self.calculate_expected_returns()
        if self.cov_matrix is None:
            self.calculate_risk_model()

        # Calculate return range if not provided
        if ret_range is None:
            min_ret = self.expected_returns.min()
            max_ret = self.expected_returns.max()
            ret_range = (min_ret, max_ret)

        returns = np.linspace(ret_range[0], ret_range[1], num_portfolios)
        volatilities = []
        sharpe_ratios = []

        for target_ret in returns:
            try:
                ef_temp = EfficientFrontier(self.expected_returns, self.cov_matrix)
                ef_temp.efficient_return(target_ret)
                ret, vol, sharpe = ef_temp.portfolio_performance(
                    risk_free_rate=self.risk_free_rate
                )
                volatilities.append(vol)
                sharpe_ratios.append(sharpe)
            except:
                # Skip infeasible portfolios
                volatilities.append(np.nan)
                sharpe_ratios.append(np.nan)

        # Remove NaN values
        valid_indices = ~np.isnan(volatilities)
        returns = returns[valid_indices].tolist()
        volatilities = np.array(volatilities)[valid_indices].tolist()
        sharpe_ratios = np.array(sharpe_ratios)[valid_indices].tolist()

        return returns, volatilities, sharpe_ratios

    def backtest_portfolio(
        self, weights: Dict[str, float] = None, rebalance_frequency: str = "monthly"
    ) -> pd.DataFrame:
        """
        Simple backtest of portfolio performance

        Args:
            weights: Portfolio weights (if None, uses last optimized weights)
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly', 'quarterly')

        Returns:
            DataFrame with portfolio performance over time
        """
        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("No weights available. Run optimization first.")

        # Convert weights to pandas Series aligned with price data columns
        weight_series = pd.Series(
            [weights.get(symbol, 0) for symbol in self.symbols], index=self.symbols
        )

        # Calculate portfolio returns
        portfolio_returns = (self.returns * weight_series).sum(axis=1)

        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Create results DataFrame
        results = pd.DataFrame(
            {
                "portfolio_return": portfolio_returns,
                "cumulative_return": cumulative_returns,
                "portfolio_value": cumulative_returns * config.DEFAULT_PORTFOLIO_VALUE,
            }
        )

        # Add benchmark (equal-weight portfolio)
        equal_weights = pd.Series(
            [1 / len(self.symbols)] * len(self.symbols), index=self.symbols
        )
        benchmark_returns = (self.returns * equal_weights).sum(axis=1)
        results["benchmark_cumulative"] = (1 + benchmark_returns).cumprod()
        results["benchmark_value"] = (
            results["benchmark_cumulative"] * config.DEFAULT_PORTFOLIO_VALUE
        )

        return results

    def get_risk_metrics(self, weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate additional risk metrics

        Args:
            weights: Portfolio weights (if None, uses last optimized weights)

        Returns:
            Dictionary of risk metrics
        """
        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("No weights available. Run optimization first.")

        # Convert weights to pandas Series
        weight_series = pd.Series(
            [weights.get(symbol, 0) for symbol in self.symbols], index=self.symbols
        )

        # Calculate portfolio returns
        portfolio_returns = (self.returns * weight_series).sum(axis=1)

        # Calculate risk metrics
        var_95 = np.percentile(portfolio_returns, 5)  # Value at Risk (95%)
        cvar_95 = portfolio_returns[
            portfolio_returns <= var_95
        ].mean()  # Conditional VaR
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        downside_deviation = self._calculate_downside_deviation(portfolio_returns)

        metrics = {
            "value_at_risk_95": var_95,
            "conditional_var_95": cvar_95,
            "max_drawdown": max_drawdown,
            "downside_deviation": downside_deviation,
            "skewness": portfolio_returns.skew(),
            "kurtosis": portfolio_returns.kurtosis(),
        }

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_downside_deviation(
        self, returns: pd.Series, target: float = 0
    ) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target]
        return np.sqrt((downside_returns**2).mean())
