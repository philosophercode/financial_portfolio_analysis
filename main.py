"""
Main script for comprehensive portfolio analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from data_fetcher import DataFetcher, validate_symbols
from portfolio_analyzer import PortfolioAnalyzer
from visualizer import PortfolioVisualizer
import config


def analyze_portfolio(
    symbols: List[str],
    api_key: str = None,
    portfolio_value: float = None,
    optimization_method: str = "max_sharpe",
    data_source: str = "yahoo",
    show_plots: bool = True,
    save_results: bool = True,
) -> Dict:
    """
    Comprehensive portfolio analysis workflow

    Args:
        symbols: List of stock symbols to analyze
        api_key: Alpha Vantage API key (optional if using Yahoo Finance)
        portfolio_value: Total portfolio value for discrete allocation
        optimization_method: Portfolio optimization method
        data_source: 'yahoo' or 'alpha_vantage'
        show_plots: Whether to display plots
        save_results: Whether to save results to files

    Returns:
        Dictionary containing all analysis results
    """
    print("🚀 Starting Portfolio Analysis")
    print("=" * 50)

    # Validate and clean symbols
    symbols = validate_symbols(symbols)
    print(f"📊 Analyzing portfolio with symbols: {symbols}")
    print(f"💰 Portfolio value: ${portfolio_value:,}")
    print(f"📅 Data period: {config.DEFAULT_PERIOD}")
    print(f"🎯 Max weight per asset: {config.MAX_WEIGHT:.1%}")
    print(f"⚙️  L2 regularization: {config.DEFAULT_L2_REG}")

    if portfolio_value is None:
        portfolio_value = config.DEFAULT_PORTFOLIO_VALUE

    # Step 1: Fetch data
    print("\n📈 Fetching stock data...")
    fetcher = DataFetcher(api_key)

    try:
        if data_source == "yahoo":
            price_data = fetcher.get_multiple_stocks(
                symbols, source="yahoo", period=config.DEFAULT_PERIOD
            )
        else:
            price_data = fetcher.get_multiple_stocks(
                symbols, source="alpha_vantage", outputsize="full"
            )

        print(f"✅ Successfully fetched data for {len(price_data.columns)} symbols")
        print(
            f"📅 Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}"
        )
        print(f"📏 Total observations: {len(price_data)}")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return {}

    # Step 2: Initialize analyzer
    print("\n🔬 Initializing portfolio analyzer...")
    analyzer = PortfolioAnalyzer(price_data)

    # Step 3: Calculate expected returns and risk model
    print("📊 Calculating expected returns and risk model...")
    print(f"📈 Using return model: {config.RETURN_MODEL}")
    print(f"📉 Using covariance model: {config.COVARIANCE_MODEL}")
    expected_returns = analyzer.calculate_expected_returns()  # Uses config defaults
    cov_matrix = analyzer.calculate_risk_model()  # Uses config defaults

    print(f"📈 Expected returns calculated for {len(expected_returns)} assets")
    print(f"📉 Risk model dimensions: {cov_matrix.shape}")

    # Step 4: Portfolio optimization
    print(f"\n⚡ Optimizing portfolio using {optimization_method}...")

    try:
        if optimization_method == "hrp":
            weights = analyzer.hierarchical_risk_parity()
        elif optimization_method == "cla":
            weights = analyzer.critical_line_algorithm()
        else:
            weights = analyzer.optimize_portfolio(
                objective=optimization_method  # Uses config.DEFAULT_L2_REG automatically
            )

        cleaned_weights = analyzer.clean_weights(weights)
        print("✅ Portfolio optimization completed")

    except Exception as e:
        print(f"❌ Error in portfolio optimization: {e}")
        return {}

    # Step 5: Performance analysis
    print("\n📊 Analyzing portfolio performance...")

    try:
        performance = analyzer.get_portfolio_performance(weights)
        risk_metrics = analyzer.get_risk_metrics(weights)
        allocation, leftover = analyzer.get_discrete_allocation(
            weights, portfolio_value
        )

        expected_return, volatility, sharpe_ratio = performance

        print(f"📈 Expected Annual Return: {expected_return:.2%}")
        print(f"📉 Annual Volatility: {volatility:.2%}")
        print(f"⚡ Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"💰 Leftover cash from allocation: ${leftover:.2f}")

    except Exception as e:
        print(f"❌ Error in performance analysis: {e}")
        return {}

    # Step 6: Backtesting
    print("\n🔄 Running portfolio backtest...")

    try:
        backtest_results = analyzer.backtest_portfolio(weights)

        # Calculate additional metrics
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

        print(f"📊 Portfolio Total Return: {total_return:.2%}")
        print(f"📊 Benchmark Total Return: {benchmark_return:.2%}")
        print(f"📊 Excess Return: {(total_return - benchmark_return):.2%}")

    except Exception as e:
        print(f"❌ Error in backtesting: {e}")
        backtest_results = pd.DataFrame()

    # Step 7: Efficient frontier calculation
    print("\n📈 Calculating efficient frontier...")

    try:
        ef_returns, ef_volatilities, ef_sharpe_ratios = (
            analyzer.calculate_efficient_frontier()
        )
        print(f"✅ Efficient frontier calculated with {len(ef_returns)} portfolios")

    except Exception as e:
        print(f"❌ Error calculating efficient frontier: {e}")
        ef_returns, ef_volatilities, ef_sharpe_ratios = [], [], []

    # Step 8: Visualization
    if show_plots:
        print("\n📊 Creating visualizations...")

        try:
            visualizer = PortfolioVisualizer()
            figures = []

            # Price history
            fig1 = visualizer.plot_price_history(price_data, symbols)
            figures.append(fig1)

            # Correlation matrix
            fig2 = visualizer.plot_correlation_matrix(analyzer.returns)
            figures.append(fig2)

            # Portfolio weights
            fig3 = visualizer.plot_portfolio_weights(cleaned_weights)
            figures.append(fig3)

            # Risk-return scatter
            fig4 = visualizer.plot_risk_return_scatter(
                symbols, analyzer.returns, cleaned_weights
            )
            figures.append(fig4)

            # Efficient frontier (if calculated)
            if ef_returns:
                optimal_point = (volatility, expected_return)
                fig5 = visualizer.plot_efficient_frontier(
                    ef_returns, ef_volatilities, ef_sharpe_ratios, optimal_point
                )
                figures.append(fig5)

            # Portfolio performance (if backtest successful)
            if not backtest_results.empty:
                fig6 = visualizer.plot_portfolio_performance(backtest_results)
                figures.append(fig6)

            print(f"✅ Created {len(figures)} visualizations")

            if save_results:
                output_dir, saved_plots = visualizer.save_all_plots(
                    figures, prefix="portfolio_analysis"
                )
            else:
                output_dir = None

            plt.show()

        except Exception as e:
            print(f"❌ Error in visualization: {e}")

    # Step 9: Summary report
    print("\n📋 Generating summary report...")

    try:
        visualizer = PortfolioVisualizer()
        summary_table = visualizer.display_summary_table(
            performance, risk_metrics, cleaned_weights
        )

        print("\n" + "=" * 60)
        print("📊 PORTFOLIO ANALYSIS SUMMARY")
        print("=" * 60)
        print(summary_table.to_string(index=False))
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        summary_table = pd.DataFrame()

    # Compile results
    results = {
        "symbols": symbols,
        "price_data": price_data,
        "expected_returns": expected_returns,
        "cov_matrix": cov_matrix,
        "weights": weights,
        "cleaned_weights": cleaned_weights,
        "performance": performance,
        "risk_metrics": risk_metrics,
        "allocation": allocation,
        "leftover_cash": leftover,
        "backtest_results": backtest_results,
        "efficient_frontier": {
            "returns": ef_returns,
            "volatilities": ef_volatilities,
            "sharpe_ratios": ef_sharpe_ratios,
        },
        "summary_table": summary_table,
    }

    if save_results:
        print(f"\n💾 Saving results to CSV files...")
        try:
            # Create timestamped directory if not already created from plots
            if "output_dir" not in locals() or output_dir is None:
                import os
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = f"portfolio_analysis/portfolio_run_{timestamp}"
                os.makedirs(output_dir, exist_ok=True)

            # Save weights
            weights_df = pd.DataFrame(
                list(cleaned_weights.items()), columns=["Symbol", "Weight"]
            )
            weights_path = f"{output_dir}/portfolio_weights.csv"
            weights_df.to_csv(weights_path, index=False)

            # Save allocation
            allocation_df = pd.DataFrame(
                list(allocation.items()), columns=["Symbol", "Shares"]
            )
            allocation_path = f"{output_dir}/portfolio_allocation.csv"
            allocation_df.to_csv(allocation_path, index=False)

            # Save summary
            summary_path = f"{output_dir}/portfolio_summary.csv"
            summary_table.to_csv(summary_path, index=False)

            # Save backtest results
            if not backtest_results.empty:
                backtest_path = f"{output_dir}/portfolio_backtest.csv"
                backtest_results.to_csv(backtest_path)

            print("✅ Results saved successfully")
            print(f"📁 CSV files saved to: {output_dir}")

        except Exception as e:
            print(f"❌ Error saving results: {e}")

    print(f"\n🎉 Portfolio analysis completed successfully!")
    return results


def run_example_analysis():
    """
    Run an example portfolio analysis with technology stocks
    """
    # Example portfolio of technology stocks
    tech_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

    print("🔬 Running example analysis with technology stocks")
    print(f"Symbols: {tech_symbols}")

    # Run analysis
    results = analyze_portfolio(
        symbols=tech_symbols,
        portfolio_value=50000,  # $50,000 portfolio
        optimization_method="max_sharpe",
        data_source="yahoo",
        show_plots=True,
        save_results=True,
    )

    return results


def run_conservative_analysis():
    """
    Run analysis with conservative dividend-paying stocks
    """
    conservative_symbols = ["JNJ", "PG", "KO", "PFE", "VZ", "T", "XOM", "CVX"]

    print("🔬 Running conservative portfolio analysis")
    print(f"Symbols: {conservative_symbols}")

    results = analyze_portfolio(
        symbols=conservative_symbols,
        portfolio_value=100000,  # $100,000 portfolio
        optimization_method="min_volatility",
        data_source="yahoo",
        show_plots=True,
        save_results=True,
    )

    return results


def compare_optimization_methods(symbols: List[str]):
    """
    Compare different optimization methods on the same portfolio
    """
    print("⚖️  Comparing optimization methods...")

    methods = ["max_sharpe", "min_volatility", "hrp"]
    results_comparison = {}

    for method in methods:
        print(f"\n🔍 Analyzing with {method} method...")

        try:
            results = analyze_portfolio(
                symbols=symbols,
                optimization_method=method,
                show_plots=False,
                save_results=False,
            )

            if results:
                performance = results["performance"]
                results_comparison[method] = {
                    "Expected Return": f"{performance[0]:.2%}",
                    "Volatility": f"{performance[1]:.2%}",
                    "Sharpe Ratio": f"{performance[2]:.3f}",
                }

        except Exception as e:
            print(f"❌ Error with {method}: {e}")

    # Display comparison
    if results_comparison:
        comparison_df = pd.DataFrame(results_comparison).T
        print("\n" + "=" * 60)
        print("⚖️  OPTIMIZATION METHODS COMPARISON")
        print("=" * 60)
        print(comparison_df.to_string())
        print("=" * 60)

    return results_comparison


if __name__ == "__main__":
    print("🏦 Financial Portfolio Analysis Tool")
    print("=" * 50)

    # You can choose which analysis to run:

    # 1. Run example tech portfolio analysis
    results = run_example_analysis()

    # 2. Uncomment to run conservative portfolio analysis
    # results = run_conservative_analysis()

    # 3. Uncomment to compare optimization methods
    # comparison = compare_optimization_methods(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])

    # 4. Custom analysis - modify symbols as needed
    # custom_symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD', 'VNQ']
    # results = analyze_portfolio(custom_symbols, portfolio_value=25000)

    print("\n🏁 Analysis complete! Check the generated files and plots.")
    print(
        "📁 All output files are organized in timestamped directories under ./portfolio_analysis/"
    )
    print("📊 Each run creates a new folder: portfolio_run_YYYYMMDD_HHMMSS")
    print("📄 Contains: CSV files + descriptively named PNG charts")
