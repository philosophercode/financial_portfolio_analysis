"""
Example Jupyter Notebook Script for Portfolio Analysis
This script demonstrates various portfolio analysis scenarios
"""

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_fetcher import DataFetcher, validate_symbols
from portfolio_analyzer import PortfolioAnalyzer
from visualizer import PortfolioVisualizer
import warnings

warnings.filterwarnings("ignore")

print("📚 Portfolio Analysis Examples")
print("=" * 40)

# %%
# Example 1: Basic Portfolio Analysis
print("\n🔬 Example 1: Basic Technology Portfolio")

# Define technology stocks
tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
print(f"Stocks: {tech_stocks}")

# Fetch data using Yahoo Finance
fetcher = DataFetcher()
price_data = fetcher.get_multiple_stocks(tech_stocks, source="yahoo", period="1y")

print(f"Data shape: {price_data.shape}")
print(f"Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}")

# %%
# Initialize portfolio analyzer
analyzer = PortfolioAnalyzer(price_data)

# Calculate expected returns and risk model
expected_returns = analyzer.calculate_expected_returns("mean_historical")
cov_matrix = analyzer.calculate_risk_model("ledoit_wolf")

print("\nExpected Returns (Annualized):")
print((expected_returns * 100).round(2))

# %%
# Portfolio Optimization - Max Sharpe Ratio
weights_max_sharpe = analyzer.optimize_portfolio("max_sharpe")
cleaned_weights = analyzer.clean_weights(weights_max_sharpe)

print("\nOptimal Portfolio Weights (Max Sharpe):")
for stock, weight in cleaned_weights.items():
    print(f"{stock}: {weight:.1%}")

# Get portfolio performance
performance = analyzer.get_portfolio_performance()
print(f"\nExpected Return: {performance[0]:.2%}")
print(f"Volatility: {performance[1]:.2%}")
print(f"Sharpe Ratio: {performance[2]:.3f}")

# %%
# Example 2: Compare Different Optimization Methods
print("\n⚖️  Example 2: Comparing Optimization Methods")

methods = ["max_sharpe", "min_volatility", "hrp"]
results_comparison = {}

for method in methods:
    if method == "hrp":
        weights = analyzer.hierarchical_risk_parity()
    else:
        weights = analyzer.optimize_portfolio(method)

    performance = analyzer.get_portfolio_performance(weights)
    results_comparison[method] = {
        "Return": f"{performance[0]:.2%}",
        "Volatility": f"{performance[1]:.2%}",
        "Sharpe": f"{performance[2]:.3f}",
    }

comparison_df = pd.DataFrame(results_comparison).T
print("\nOptimization Methods Comparison:")
print(comparison_df)

# %%
# Example 3: Efficient Frontier Analysis
print("\n📈 Example 3: Efficient Frontier")

# Calculate efficient frontier
ef_returns, ef_volatilities, ef_sharpe_ratios = analyzer.calculate_efficient_frontier(
    num_portfolios=50
)

print(f"Efficient frontier calculated with {len(ef_returns)} portfolios")
print(f"Return range: {min(ef_returns):.2%} to {max(ef_returns):.2%}")
print(f"Risk range: {min(ef_volatilities):.2%} to {max(ef_volatilities):.2%}")

# %%
# Example 4: Visualization
print("\n📊 Example 4: Creating Visualizations")

visualizer = PortfolioVisualizer()

# Plot 1: Price history
fig1 = visualizer.plot_price_history(price_data, tech_stocks)
plt.title("Normalized Price History - Technology Stocks")
plt.show()

# %%
# Plot 2: Correlation matrix
fig2 = visualizer.plot_correlation_matrix(analyzer.returns)
plt.title("Asset Correlation Matrix")
plt.show()

# %%
# Plot 3: Portfolio weights
fig3 = visualizer.plot_portfolio_weights(
    cleaned_weights, "Optimal Portfolio Allocation"
)
plt.show()

# %%
# Plot 4: Efficient frontier
optimal_point = (performance[1], performance[0])  # (volatility, return)
fig4 = visualizer.plot_efficient_frontier(
    ef_returns, ef_volatilities, ef_sharpe_ratios, optimal_point
)
plt.title("Efficient Frontier with Optimal Portfolio")
plt.show()

# %%
# Example 5: Risk Analysis
print("\n⚠️  Example 5: Risk Metrics Analysis")

risk_metrics = analyzer.get_risk_metrics(weights_max_sharpe)

print("Risk Metrics:")
for metric, value in risk_metrics.items():
    if (
        "var" in metric.lower()
        or "drawdown" in metric.lower()
        or "deviation" in metric.lower()
    ):
        print(f"{metric}: {value:.2%}")
    else:
        print(f"{metric}: {value:.3f}")

# %%
# Example 6: Discrete Allocation
print("\n💰 Example 6: Portfolio Allocation for $25,000")

allocation, leftover = analyzer.get_discrete_allocation(weights_max_sharpe, 25000)

print("Share Allocation:")
total_allocated = 0
for stock, shares in allocation.items():
    latest_price = price_data[stock].iloc[-1]
    value = shares * latest_price
    total_allocated += value
    print(f"{stock}: {shares} shares @ ${latest_price:.2f} = ${value:.2f}")

print(f"\nTotal allocated: ${total_allocated:.2f}")
print(f"Leftover cash: ${leftover:.2f}")

# %%
# Example 7: Backtesting
print("\n🔄 Example 7: Portfolio Backtesting")

backtest_results = analyzer.backtest_portfolio(weights_max_sharpe)

# Calculate performance metrics
total_return = (
    backtest_results["portfolio_value"].iloc[-1]
    / backtest_results["portfolio_value"].iloc[0]
    - 1
)
benchmark_return = (
    backtest_results["benchmark_value"].iloc[-1]
    / backtest_results["benchmark_value"].iloc[0]
    - 1
)

print(f"Portfolio Total Return: {total_return:.2%}")
print(f"Benchmark Total Return: {benchmark_return:.2%}")
print(f"Excess Return: {(total_return - benchmark_return):.2%}")

# Plot backtest results
fig5 = visualizer.plot_portfolio_performance(backtest_results)
plt.show()

# %%
# Example 8: Sector Diversification Analysis
print("\n🏭 Example 8: Adding Sector Diversification")

# Diversified portfolio across sectors
diversified_stocks = [
    "AAPL",  # Technology
    "JNJ",  # Healthcare
    "JPM",  # Financials
    "XOM",  # Energy
    "PG",  # Consumer Goods
    "DIS",  # Entertainment
    "BA",  # Industrials
    "GLD",  # Commodities ETF
]

print(f"Diversified portfolio: {diversified_stocks}")

# Fetch data for diversified portfolio
div_price_data = fetcher.get_multiple_stocks(
    diversified_stocks, source="yahoo", period="2y"
)
div_analyzer = PortfolioAnalyzer(div_price_data)

# Optimize diversified portfolio
div_weights = div_analyzer.optimize_portfolio(
    "max_sharpe", l2_gamma=0.2
)  # Higher regularization
div_cleaned = div_analyzer.clean_weights(div_weights)
div_performance = div_analyzer.get_portfolio_performance()

print("\nDiversified Portfolio Weights:")
for stock, weight in div_cleaned.items():
    print(f"{stock}: {weight:.1%}")

print(f"\nDiversified Portfolio Performance:")
print(f"Expected Return: {div_performance[0]:.2%}")
print(f"Volatility: {div_performance[1]:.2%}")
print(f"Sharpe Ratio: {div_performance[2]:.3f}")

# %%
# Example 9: Monte Carlo Simulation
print("\n🎲 Example 9: Monte Carlo Simulation")


def monte_carlo_simulation(analyzer, weights, num_simulations=1000, time_horizon=252):
    """Simple Monte Carlo simulation for portfolio returns"""

    # Get portfolio parameters
    portfolio_return = np.dot(weights, analyzer.expected_returns)
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(analyzer.cov_matrix, weights)))

    # Generate random scenarios
    random_returns = np.random.normal(
        portfolio_return / 252,  # Daily return
        portfolio_vol / np.sqrt(252),  # Daily volatility
        (num_simulations, time_horizon),
    )

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + random_returns, axis=1)
    final_values = cumulative_returns[:, -1]

    return final_values, cumulative_returns


# Run Monte Carlo simulation
weight_array = np.array([weights_max_sharpe.get(stock, 0) for stock in tech_stocks])
final_values, paths = monte_carlo_simulation(
    analyzer, weight_array, num_simulations=500
)

# Calculate confidence intervals
percentiles = [5, 25, 50, 75, 95]
confidence_levels = np.percentile(final_values, percentiles)

print("Monte Carlo Results (1-year horizon):")
print("Portfolio Value Percentiles:")
for i, p in enumerate(percentiles):
    value = confidence_levels[i] * 10000  # Assuming $10k initial investment
    print(f"{p}th percentile: ${value:.0f}")

# %%
# Example 10: Summary Report
print("\n📋 Example 10: Comprehensive Summary Report")

summary_table = visualizer.display_summary_table(
    performance, risk_metrics, cleaned_weights
)

print("\nPortfolio Analysis Summary:")
print("=" * 50)
print(summary_table.to_string(index=False))
print("=" * 50)

# %%
print("\n🎉 Portfolio Analysis Examples Completed!")
print("All examples have been executed successfully.")
print("\nKey Takeaways:")
print("• Different optimization methods yield different risk-return profiles")
print("• Diversification across sectors can improve risk-adjusted returns")
print("• Regular backtesting helps validate optimization strategies")
print("• Risk metrics provide important insights beyond return and volatility")
print("• Monte Carlo simulation helps understand potential outcome ranges")
