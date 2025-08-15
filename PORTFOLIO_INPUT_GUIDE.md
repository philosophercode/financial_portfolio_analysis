# Portfolio Input Guide

This guide shows you how to input your portfolio holdings for analysis.

## Quick Start

### Method 1: Use the Simple Script
```bash
python analyze_my_portfolio.py
```

### Method 2: Create Your Portfolio File
Create a CSV file with your holdings and run:
```python
from portfolio_input import load_and_analyze_portfolio

results = load_and_analyze_portfolio("my_portfolio.csv")
```

## Input Formats

### Format 1: Shares (Recommended)
If you know how many shares you own:

**File: `my_portfolio.csv`**
```csv
symbol,shares
AAPL,50
MSFT,30
GOOGL,10
AMZN,15
TSLA,25
```

The system will:
- Fetch current market prices
- Calculate your total portfolio value
- Convert to percentage weights

### Format 2: Percentage Weights
If you want to specify target allocations:

**File: `my_portfolio.csv`**
```csv
symbol,weight
AAPL,0.25
MSFT,0.30
GOOGL,0.20
AMZN,0.15
TSLA,0.10
```

**Note:** Weights should sum to 1.0 (100%)

## Usage Examples

### Example 1: Analyze From File
```python
from portfolio_input import load_and_analyze_portfolio

# Analyze your portfolio
results = load_and_analyze_portfolio(
    file_path="my_portfolio.csv",
    optimization_method="max_sharpe",  # or "min_volatility", "hrp"
    show_plots=True,
    save_results=True
)
```

### Example 2: Compare Current vs Optimized
```python
from portfolio_input import compare_portfolios

# Compare your current allocation with optimized version
comparison = compare_portfolios(
    current_file="my_portfolio.csv",
    optimization_method="max_sharpe",
    show_plots=True
)
```

### Example 3: Quick Analysis from Code
```python
from portfolio_input import quick_portfolio_analysis

# Using shares
my_shares = {
    "AAPL": 50,
    "MSFT": 30,
    "GOOGL": 10
}
results = quick_portfolio_analysis(my_shares, "shares")

# Using weights
my_weights = {
    "AAPL": 0.4,
    "MSFT": 0.3,
    "GOOGL": 0.3
}
results = quick_portfolio_analysis(my_weights, "weights", total_value=25000)
```

## What You Get

### 1. Current Portfolio Analysis
- Total portfolio value (for shares format)
- Current allocation percentages
- Risk and return metrics

### 2. Optimized Portfolio
- Mathematically optimal weights
- Expected performance improvements
- Detailed risk analysis

### 3. Comparison Report
- Side-by-side comparison
- Performance metrics
- Allocation changes needed

### 4. Visual Analysis
- Portfolio allocation pie charts
- Efficient frontier plot
- Risk-return scatter plot
- Performance over time

## Output Files

All results are saved in timestamped directories:
```
portfolio_analysis/portfolio_run_YYYYMMDD_HHMMSS/
├── portfolio_weights.csv              # Optimized weights
├── portfolio_allocation.csv           # Share quantities to buy
├── portfolio_summary.csv              # Performance summary
├── portfolio_backtest.csv             # Historical performance
└── [visualization charts].png
```

## Interactive Mode

Run the interactive script:
```bash
python analyze_my_portfolio.py
```

Choose from:
1. **Create example files** - Generate sample CSV templates
2. **Analyze portfolio from file** - Load and optimize your portfolio
3. **Compare portfolios** - Current vs optimized comparison
4. **Quick analysis** - Fast analysis with built-in example

## Tips

### For Shares Format:
- Use actual share quantities you own
- System fetches current market prices automatically
- More accurate for existing portfolios

### For Weights Format:
- Good for target allocations
- Weights must sum to 1.0
- Specify total_value parameter if different from default

### Symbol Format:
- Use standard ticker symbols (AAPL, MSFT, GOOGL)
- Supports stocks and ETFs
- International symbols may work but not guaranteed

## Troubleshooting

**File not found error:**
```bash
# Create example files first
python -c "from portfolio_input import PortfolioInputHandler; PortfolioInputHandler().save_example_files()"
```

**Price fetch errors:**
- Check ticker symbols are correct
- Some international symbols may not work
- System will skip symbols it can't find

**Weight errors:**
- Ensure weights sum to 1.0 for weights format
- Use positive numbers only
- At least 2 symbols required

## Advanced Usage

### Custom Analysis
```python
from portfolio_input import PortfolioInputHandler
from main import analyze_portfolio

# Load portfolio
handler = PortfolioInputHandler()
symbols, weights, total_value = handler.load_portfolio_from_file("my_portfolio.csv")

# Custom analysis
results = analyze_portfolio(
    symbols=symbols,
    portfolio_value=total_value,
    optimization_method="min_volatility",  # Lower risk
    data_source="yahoo",
    show_plots=True,
    save_results=True
)
```

### Batch Analysis
```python
portfolios = ["portfolio1.csv", "portfolio2.csv", "portfolio3.csv"]

for portfolio_file in portfolios:
    print(f"Analyzing {portfolio_file}...")
    results = load_and_analyze_portfolio(portfolio_file, save_results=True)
```

Happy investing! 📈🎯
