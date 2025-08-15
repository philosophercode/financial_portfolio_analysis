# Financial Portfolio Analysis Tool

A comprehensive Python-based portfolio analysis tool that leverages modern portfolio theory for optimal asset allocation. This tool combines data from Alpha Vantage API and Yahoo Finance with PyPortfolioOpt's optimization algorithms to provide detailed portfolio insights.

## Features

🚀 **Data Collection**
- Fetch stock data from Alpha Vantage API or Yahoo Finance
- Support for multiple symbols and time periods
- Automatic data validation and cleaning

📊 **Portfolio Optimization**
- Mean-variance optimization (Markowitz)
- Hierarchical Risk Parity (HRP)
- Critical Line Algorithm (CLA)
- Multiple objective functions (Max Sharpe, Min Volatility, etc.)

📈 **Analysis & Metrics**
- Expected returns and risk models
- Efficient frontier calculation
- Portfolio performance metrics
- Risk metrics (VaR, CVaR, Max Drawdown, etc.)
- Discrete allocation for real-world trading

📋 **Visualization**
- Interactive and static plots
- Efficient frontier visualization
- Portfolio composition charts
- Performance dashboards
- Risk-return analysis

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd financial_portfolio_analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Alpha Vantage API:
   - Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Create a `.env` file based on `.env.example`
   - Add your API key: `ALPHA_VANTAGE_API_KEY=your_api_key_here`

## Quick Start

### Basic Usage

```python
from main import analyze_portfolio

# Define your portfolio
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Run comprehensive analysis
results = analyze_portfolio(
    symbols=symbols,
    portfolio_value=10000,
    optimization_method='max_sharpe',
    data_source='yahoo'
)
```

### Run Example Analyses

```bash
# Run the main script with example portfolios
python main.py
```

This will run a technology portfolio analysis with visualization and save results to CSV files.

## Usage Examples

### 1. Technology Portfolio Analysis

```python
tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']

results = analyze_portfolio(
    symbols=tech_symbols,
    portfolio_value=50000,
    optimization_method='max_sharpe',
    show_plots=True,
    save_results=True
)
```

### 2. Conservative Portfolio with Dividend Stocks

```python
conservative_symbols = ['JNJ', 'PG', 'KO', 'PFE', 'VZ', 'T', 'XOM', 'CVX']

results = analyze_portfolio(
    symbols=conservative_symbols,
    optimization_method='min_volatility',
    portfolio_value=100000
)
```

### 3. Using Alpha Vantage API

```python
from data_fetcher import DataFetcher

fetcher = DataFetcher(api_key='your_api_key')
price_data = fetcher.get_multiple_stocks(
    symbols=['AAPL', 'MSFT', 'GOOGL'], 
    source='alpha_vantage'
)
```

### 4. Custom Analysis

```python
from portfolio_analyzer import PortfolioAnalyzer
from visualizer import PortfolioVisualizer

# Initialize analyzer with your data
analyzer = PortfolioAnalyzer(price_data)

# Calculate expected returns and risk model
analyzer.calculate_expected_returns(method='mean_historical')
analyzer.calculate_risk_model(method='ledoit_wolf')

# Optimize portfolio
weights = analyzer.optimize_portfolio(
    objective='max_sharpe',
    l2_gamma=0.1  # Add regularization for diversification
)

# Get performance metrics
performance = analyzer.get_portfolio_performance()
risk_metrics = analyzer.get_risk_metrics()

# Create visualizations
visualizer = PortfolioVisualizer()
fig = visualizer.plot_efficient_frontier(returns, volatilities, sharpe_ratios)
```

## Optimization Methods

### 1. Mean-Variance Optimization
- **Max Sharpe**: Maximizes risk-adjusted returns
- **Min Volatility**: Minimizes portfolio risk
- **Efficient Return**: Targets specific return level
- **Efficient Risk**: Targets specific risk level

### 2. Alternative Methods
- **Hierarchical Risk Parity (HRP)**: Robust to estimation errors
- **Critical Line Algorithm (CLA)**: Exact solution for mean-variance

### 3. Advanced Features
- **L2 Regularization**: Promotes diversification
- **Market Neutral**: Zero-beta portfolios
- **Custom Constraints**: Weight bounds and sector limits

## Configuration

Modify `config.py` to customize the analysis:

```python
# Portfolio & data
DEFAULT_PORTFOLIO_VALUE = 10_000
DEFAULT_PERIOD = "5y"            # more history for stabler covariances
DEFAULT_INTERVAL = "1d"
DEFAULT_FREQUENCY = 252          # use consistently in PPO calls

# Return & risk models (pick in code)
RETURN_MODEL = "ewma"            # or "mean"; for ewma use span ~60–90 trading days
COVARIANCE_MODEL = "ledoit_wolf" # shrinkage; or "oas"; or "semicov" for downside risk

# Optimization
DEFAULT_L2_REG = 1e-3            # PPO's L2 regularization (gamma-like)
DEFAULT_WEIGHT_BOUNDS = (0, 1)
MAX_WEIGHT = 0.60                # optional cap to avoid concentration
SECTOR_NEUTRAL = False           # if you have sector labels, can add constraints

# Risk-free & scaling
DEFAULT_RISK_FREE_RATE = 0.02    # annual; keep synced to T-bills
```

### Key Configuration Benefits:
- **5-year data period**: Provides more stable covariance estimates
- **EWMA returns**: Exponentially weighted moving average gives more weight to recent data
- **Ledoit-Wolf shrinkage**: Improves covariance matrix estimation
- **Weight constraints**: Prevents over-concentration in single assets
- **L2 regularization**: Promotes diversification

## Output Files

The analysis generates organized output files in timestamped directories:

```
portfolio_analysis/
└── portfolio_run_YYYYMMDD_HHMMSS/
    ├── portfolio_weights.csv              # Optimized portfolio weights
    ├── portfolio_allocation.csv           # Discrete share allocation  
    ├── portfolio_summary.csv              # Performance summary
    ├── portfolio_backtest.csv             # Historical performance
    ├── portfolio_analysis_price_history.png
    ├── portfolio_analysis_correlation_matrix.png
    ├── portfolio_analysis_portfolio_weights.png
    ├── portfolio_analysis_risk_return_scatter.png
    ├── portfolio_analysis_efficient_frontier.png
    └── portfolio_analysis_performance_backtest.png
```

Each analysis run creates a new timestamped folder, keeping all results organized and preventing overwrites.

## API Documentation

### DataFetcher Class

```python
fetcher = DataFetcher(api_key='optional')

# Fetch single stock
data = fetcher.get_stock_data_yahoo('AAPL', period='1y')

# Fetch multiple stocks
portfolio_data = fetcher.get_multiple_stocks(['AAPL', 'MSFT'], source='yahoo')

# Get company information (Alpha Vantage only)
info = fetcher.get_company_overview('AAPL')
```

### PortfolioAnalyzer Class

```python
analyzer = PortfolioAnalyzer(price_data, risk_free_rate=0.02)

# Expected returns methods
analyzer.calculate_expected_returns(method='mean_historical')  # or 'ema_historical', 'capm'

# Risk model methods  
analyzer.calculate_risk_model(method='ledoit_wolf')  # or 'sample', 'oas', 'semicovariance'

# Optimization
weights = analyzer.optimize_portfolio(objective='max_sharpe')
weights = analyzer.hierarchical_risk_parity()

# Analysis
performance = analyzer.get_portfolio_performance()
allocation, leftover = analyzer.get_discrete_allocation(total_portfolio_value=10000)
backtest = analyzer.backtest_portfolio()
```

### PortfolioVisualizer Class

```python
visualizer = PortfolioVisualizer()

# Static plots
fig1 = visualizer.plot_price_history(price_data)
fig2 = visualizer.plot_correlation_matrix(returns)
fig3 = visualizer.plot_portfolio_weights(weights)
fig4 = visualizer.plot_efficient_frontier(returns, volatilities, sharpe_ratios)

# Interactive plots (Plotly)
interactive_fig = visualizer.create_interactive_efficient_frontier(returns, vols, sharpes)
dashboard = visualizer.create_performance_dashboard(backtest_results, weights, risk_metrics)
```

## Performance Metrics

The tool calculates comprehensive performance and risk metrics:

### Return Metrics
- Expected Annual Return
- Realized Returns
- Excess Returns vs Benchmark

### Risk Metrics
- Annual Volatility
- Sharpe Ratio
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum Drawdown
- Downside Deviation
- Skewness and Kurtosis

### Portfolio Characteristics
- Number of Holdings
- Concentration Metrics
- Turnover Analysis

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **PyPortfolioOpt**: Portfolio optimization
- **matplotlib/seaborn**: Static visualization
- **plotly**: Interactive visualization
- **requests**: API data fetching
- **yfinance**: Yahoo Finance data
- **python-dotenv**: Environment management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This tool is for educational and research purposes only. It should not be considered as financial advice. Always consult with qualified financial professionals before making investment decisions.

## References

- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)
- [Alpha Vantage API Documentation](https://www.alphavantage.co/documentation/)
- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.

## Support

For questions, issues, or feature requests, please open an issue on the GitHub repository.
