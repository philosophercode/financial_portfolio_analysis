"""
Visualization module for portfolio analysis results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import config


class PortfolioVisualizer:
    """
    A class for creating various portfolio analysis visualizations
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize the visualizer

        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.figure_size = config.FIGURE_SIZE
        self.dpi = config.DPI

    def plot_price_history(
        self,
        price_data: pd.DataFrame,
        symbols: List[str] = None,
        title: str = "Stock Price History",
    ) -> plt.Figure:
        """
        Plot normalized price history for selected stocks

        Args:
            price_data: DataFrame with stock prices
            symbols: List of symbols to plot (if None, plots all)
            title: Chart title

        Returns:
            Matplotlib figure
        """
        if symbols is None:
            symbols = list(price_data.columns)

        # Normalize prices to start at 100
        normalized_prices = (price_data[symbols] / price_data[symbols].iloc[0]) * 100

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        for symbol in symbols:
            ax.plot(
                normalized_prices.index,
                normalized_prices[symbol],
                label=symbol,
                linewidth=2,
                alpha=0.8,
            )

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Price (Base = 100)", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(
        self, returns: pd.DataFrame, title: str = "Asset Correlation Matrix"
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap

        Args:
            returns: DataFrame of asset returns
            title: Chart title

        Returns:
            Matplotlib figure
        """
        correlation_matrix = returns.corr()

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="RdYlBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_efficient_frontier(
        self,
        returns: List[float],
        volatilities: List[float],
        sharpe_ratios: List[float],
        optimal_portfolio: Tuple[float, float] = None,
        title: str = "Efficient Frontier",
    ) -> plt.Figure:
        """
        Plot the efficient frontier

        Args:
            returns: List of portfolio returns
            volatilities: List of portfolio volatilities
            sharpe_ratios: List of Sharpe ratios
            optimal_portfolio: Tuple of (volatility, return) for optimal portfolio
            title: Chart title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Create scatter plot colored by Sharpe ratio
        scatter = ax.scatter(
            volatilities,
            returns,
            c=sharpe_ratios,
            cmap="viridis",
            s=50,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Sharpe Ratio", rotation=270, labelpad=20)

        # Highlight optimal portfolio if provided
        if optimal_portfolio:
            ax.scatter(
                optimal_portfolio[0],
                optimal_portfolio[1],
                marker="*",
                s=500,
                c="red",
                edgecolors="black",
                linewidth=2,
                label="Optimal Portfolio",
                zorder=5,
            )
            ax.legend()

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Volatility (Risk)", fontsize=12)
        ax.set_ylabel("Expected Return", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_portfolio_weights(
        self, weights: Dict[str, float], title: str = "Portfolio Allocation"
    ) -> plt.Figure:
        """
        Plot portfolio weights as a pie chart

        Args:
            weights: Dictionary of portfolio weights
            title: Chart title

        Returns:
            Matplotlib figure
        """
        # Filter out zero weights
        non_zero_weights = {k: v for k, v in weights.items() if abs(v) > 0.001}

        if not non_zero_weights:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            ax.text(
                0.5,
                0.5,
                "No significant weights to display",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(title, fontsize=16, fontweight="bold")
            return fig

        symbols = list(non_zero_weights.keys())
        values = list(non_zero_weights.values())

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=symbols,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )

        # Beautify the pie chart
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax.set_title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_portfolio_performance(
        self, backtest_results: pd.DataFrame, title: str = "Portfolio Performance"
    ) -> plt.Figure:
        """
        Plot portfolio performance over time

        Args:
            backtest_results: DataFrame with portfolio performance data
            title: Chart title

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(self.figure_size[0], self.figure_size[1] * 1.2),
            dpi=self.dpi,
            sharex=True,
        )

        # Plot cumulative returns
        ax1.plot(
            backtest_results.index,
            backtest_results["portfolio_value"],
            label="Portfolio",
            linewidth=2,
            color="blue",
        )
        ax1.plot(
            backtest_results.index,
            backtest_results["benchmark_value"],
            label="Equal Weight Benchmark",
            linewidth=2,
            color="red",
            alpha=0.7,
        )

        ax1.set_title(title, fontsize=16, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot daily returns
        ax2.plot(
            backtest_results.index,
            backtest_results["portfolio_return"] * 100,
            linewidth=1,
            alpha=0.7,
            color="blue",
        )
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        ax2.set_ylabel("Daily Return (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_risk_return_scatter(
        self,
        symbols: List[str],
        returns: pd.DataFrame,
        weights: Dict[str, float] = None,
        title: str = "Risk-Return Characteristics",
    ) -> plt.Figure:
        """
        Plot risk-return scatter for individual assets and portfolio

        Args:
            symbols: List of asset symbols
            returns: DataFrame of asset returns
            weights: Portfolio weights (optional)
            title: Chart title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Calculate risk and return for each asset
        asset_returns = returns.mean() * 252  # Annualized
        asset_risks = returns.std() * np.sqrt(252)  # Annualized

        # Plot individual assets
        for symbol in symbols:
            ax.scatter(
                asset_risks[symbol],
                asset_returns[symbol],
                s=100,
                alpha=0.7,
                label=symbol,
            )
            ax.annotate(
                symbol,
                (asset_risks[symbol], asset_returns[symbol]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        # Plot portfolio if weights provided
        if weights:
            weight_array = np.array([weights.get(symbol, 0) for symbol in symbols])
            portfolio_return = np.dot(weight_array, asset_returns)
            portfolio_risk = np.sqrt(
                np.dot(weight_array, np.dot(returns.cov() * 252, weight_array))
            )

            ax.scatter(
                portfolio_risk,
                portfolio_return,
                marker="*",
                s=500,
                c="red",
                edgecolors="black",
                linewidth=2,
                label="Portfolio",
                zorder=5,
            )

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Annualized Volatility", fontsize=12)
        ax.set_ylabel("Annualized Return", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        return fig

    def create_interactive_efficient_frontier(
        self,
        returns: List[float],
        volatilities: List[float],
        sharpe_ratios: List[float],
        optimal_portfolio: Tuple[float, float] = None,
    ) -> go.Figure:
        """
        Create interactive efficient frontier using Plotly

        Args:
            returns: List of portfolio returns
            volatilities: List of portfolio volatilities
            sharpe_ratios: List of Sharpe ratios
            optimal_portfolio: Tuple of (volatility, return) for optimal portfolio

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Add efficient frontier
        fig.add_trace(
            go.Scatter(
                x=volatilities,
                y=returns,
                mode="markers",
                marker=dict(
                    size=8,
                    color=sharpe_ratios,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                ),
                text=[
                    f"Return: {r:.3f}<br>Volatility: {v:.3f}<br>Sharpe: {s:.3f}"
                    for r, v, s in zip(returns, volatilities, sharpe_ratios)
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Efficient Frontier",
            )
        )

        # Add optimal portfolio if provided
        if optimal_portfolio:
            fig.add_trace(
                go.Scatter(
                    x=[optimal_portfolio[0]],
                    y=[optimal_portfolio[1]],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color="red",
                        symbol="star",
                        line=dict(width=2, color="black"),
                    ),
                    name="Optimal Portfolio",
                    text=f"Optimal: Return={optimal_portfolio[1]:.3f}, Risk={optimal_portfolio[0]:.3f}",
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Interactive Efficient Frontier",
            xaxis_title="Volatility (Risk)",
            yaxis_title="Expected Return",
            hovermode="closest",
        )

        return fig

    def create_performance_dashboard(
        self,
        backtest_results: pd.DataFrame,
        weights: Dict[str, float],
        risk_metrics: Dict[str, float],
    ) -> go.Figure:
        """
        Create comprehensive performance dashboard

        Args:
            backtest_results: DataFrame with portfolio performance data
            weights: Portfolio weights
            risk_metrics: Dictionary of risk metrics

        Returns:
            Plotly figure with subplots
        """
        # Create subplot structure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Portfolio Value Over Time",
                "Daily Returns Distribution",
                "Portfolio Allocation",
                "Risk Metrics",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "table"}],
            ],
        )

        # Portfolio value over time
        fig.add_trace(
            go.Scatter(
                x=backtest_results.index,
                y=backtest_results["portfolio_value"],
                name="Portfolio",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=backtest_results.index,
                y=backtest_results["benchmark_value"],
                name="Benchmark",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Daily returns distribution
        fig.add_trace(
            go.Histogram(
                x=backtest_results["portfolio_return"] * 100,
                name="Daily Returns (%)",
                nbinsx=50,
            ),
            row=1,
            col=2,
        )

        # Portfolio allocation pie chart
        non_zero_weights = {k: v for k, v in weights.items() if abs(v) > 0.001}
        if non_zero_weights:
            fig.add_trace(
                go.Pie(
                    labels=list(non_zero_weights.keys()),
                    values=list(non_zero_weights.values()),
                    name="Portfolio Allocation",
                ),
                row=2,
                col=1,
            )

        # Risk metrics table
        metrics_df = pd.DataFrame(
            list(risk_metrics.items()), columns=["Metric", "Value"]
        )
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Risk Metric", "Value"],
                    fill_color="lightblue",
                    align="left",
                ),
                cells=dict(
                    values=[
                        metrics_df["Metric"],
                        [f"{v:.4f}" for v in metrics_df["Value"]],
                    ],
                    fill_color="lightcyan",
                    align="left",
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800, showlegend=False, title_text="Portfolio Performance Dashboard"
        )

        return fig

    def save_all_plots(
        self,
        figures: List[plt.Figure],
        output_dir: str = None,
        prefix: str = "portfolio_analysis",
    ):
        """
        Save all matplotlib figures to files in organized directory structure

        Args:
            figures: List of matplotlib figures
            output_dir: Directory to save plots (if None, creates timestamped folder)
            prefix: Prefix for file names
        """
        import os
        from datetime import datetime

        # Create timestamped directory if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"portfolio_analysis/portfolio_run_{timestamp}"

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define plot names for better organization
        plot_names = [
            "price_history",
            "correlation_matrix",
            "portfolio_weights",
            "risk_return_scatter",
            "efficient_frontier",
            "performance_backtest",
        ]

        saved_files = []
        for i, fig in enumerate(figures):
            # Use descriptive name if available, otherwise use index
            if i < len(plot_names):
                filename = f"{prefix}_{plot_names[i]}.png"
            else:
                filename = f"{prefix}_{i+1}.png"

            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
            saved_files.append(filepath)
            print(f"Saved plot: {filepath}")

        print(f"\n📁 All plots saved to directory: {output_dir}")
        return output_dir, saved_files

    def display_summary_table(
        self,
        portfolio_performance: Tuple[float, float, float],
        risk_metrics: Dict[str, float],
        weights: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Create a summary table of portfolio analysis results

        Args:
            portfolio_performance: Tuple of (return, volatility, sharpe_ratio)
            risk_metrics: Dictionary of risk metrics
            weights: Portfolio weights

        Returns:
            DataFrame with summary statistics
        """
        expected_return, volatility, sharpe_ratio = portfolio_performance

        # Portfolio performance metrics
        performance_data = {
            "Expected Annual Return": f"{expected_return:.2%}",
            "Annual Volatility": f"{volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.3f}",
            "Value at Risk (95%)": f"{risk_metrics.get('value_at_risk_95', 0):.2%}",
            "Conditional VaR (95%)": f"{risk_metrics.get('conditional_var_95', 0):.2%}",
            "Maximum Drawdown": f"{risk_metrics.get('max_drawdown', 0):.2%}",
            "Downside Deviation": f"{risk_metrics.get('downside_deviation', 0):.2%}",
            "Skewness": f"{risk_metrics.get('skewness', 0):.3f}",
            "Kurtosis": f"{risk_metrics.get('kurtosis', 0):.3f}",
        }

        # Top holdings
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        top_holdings = {
            f"Top {i+1} Holding": f"{symbol} ({weight:.1%})"
            for i, (symbol, weight) in enumerate(sorted_weights[:5])
        }

        # Combine all data
        summary_data = {**performance_data, **top_holdings}

        summary_df = pd.DataFrame(
            list(summary_data.items()), columns=["Metric", "Value"]
        )

        return summary_df
