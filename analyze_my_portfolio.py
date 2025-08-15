"""
Simple Portfolio Analysis Script
Load portfolio from CSV file and analyze it
"""

from portfolio_input import (
    PortfolioInputHandler,
    load_and_analyze_portfolio,
    compare_portfolios,
)


def create_example_files():
    """Create example portfolio input files"""
    print("📁 Creating example portfolio files...")

    handler = PortfolioInputHandler()
    handler.save_example_files()

    print("\n📋 Example file formats:")
    print("\n1. Shares Format (example_portfolio_shares.csv):")
    print("   symbol,shares")
    print("   AAPL,50")
    print("   MSFT,30")
    print("   GOOGL,10")

    print("\n2. Weights Format (example_portfolio_weights.csv):")
    print("   symbol,weight")
    print("   AAPL,0.25")
    print("   MSFT,0.20")
    print("   GOOGL,0.20")


def analyze_portfolio_from_file(file_path: str = "my_portfolio.csv"):
    """
    Analyze portfolio from CSV file

    Args:
        file_path: Path to your portfolio CSV file
    """
    try:
        print(f"🚀 Analyzing portfolio from: {file_path}")

        # Load and analyze portfolio
        results = load_and_analyze_portfolio(
            file_path=file_path,
            optimization_method="max_sharpe",
            show_plots=True,
            save_results=True,
        )

        print("\n✅ Analysis complete!")
        return results

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("💡 Create your portfolio file or run create_example_files() first")
        return None
    except Exception as e:
        print(f"❌ Error analyzing portfolio: {e}")
        return None


def compare_my_portfolio(file_path: str = "my_portfolio.csv"):
    """
    Compare your current portfolio with optimized version

    Args:
        file_path: Path to your current portfolio CSV file
    """
    try:
        print(f"🔍 Comparing your portfolio: {file_path}")

        comparison = compare_portfolios(
            current_file=file_path, optimization_method="max_sharpe", show_plots=True
        )

        print("\n✅ Comparison complete!")
        return comparison

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("💡 Create your portfolio file first")
        return None
    except Exception as e:
        print(f"❌ Error comparing portfolio: {e}")
        return None


def quick_portfolio_analysis(
    portfolio_dict: dict, input_type: str = "auto", total_value: float = None
):
    """
    Quick analysis from dictionary input

    Args:
        portfolio_dict: Dictionary with symbol -> value mapping
        input_type: "shares", "weights", or "auto"
        total_value: Total portfolio value (for shares)

    Example:
        # Using shares
        portfolio = {"AAPL": 50, "MSFT": 30, "GOOGL": 10}
        quick_portfolio_analysis(portfolio, "shares")

        # Using weights
        portfolio = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        quick_portfolio_analysis(portfolio, "weights", total_value=25000)
    """
    from main import analyze_portfolio

    print("🚀 Quick Portfolio Analysis")
    print("=" * 30)

    # Convert dictionary to standard format
    handler = PortfolioInputHandler()
    symbols, weights, portfolio_value = handler.create_portfolio_from_dict(
        portfolio_dict, input_type, total_value
    )

    print(f"📊 Portfolio loaded:")
    print(f"   Symbols: {symbols}")
    print(f"   Total Value: ${portfolio_value:,.2f}")

    # Run analysis
    results = analyze_portfolio(
        symbols=symbols,
        portfolio_value=portfolio_value,
        optimization_method="max_sharpe",
        show_plots=True,
        save_results=True,
    )

    return results


if __name__ == "__main__":
    print("🏦 Portfolio Analysis Tool")
    print("=" * 50)
    print("Choose an option:")
    print("1. Create example files")
    print("2. Analyze portfolio from file")
    print("3. Compare current vs optimized portfolio")
    print("4. Quick analysis from code")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        create_example_files()

    elif choice == "2":
        file_path = input(
            "Enter portfolio file path (default: my_portfolio.csv): "
        ).strip()
        if not file_path:
            file_path = "my_portfolio.csv"
        analyze_portfolio_from_file(file_path)

    elif choice == "3":
        file_path = input(
            "Enter portfolio file path (default: my_portfolio.csv): "
        ).strip()
        if not file_path:
            file_path = "my_portfolio.csv"
        compare_my_portfolio(file_path)

    elif choice == "4":
        print("\nExample quick analysis:")
        print("Running analysis with: AAPL=50 shares, MSFT=30 shares, GOOGL=10 shares")

        example_portfolio = {"AAPL": 50, "MSFT": 30, "GOOGL": 10}

        quick_portfolio_analysis(example_portfolio, "shares")

    else:
        print("Invalid choice. Please run the script again.")

    print("\n🎉 Done!")
