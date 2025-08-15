"""
Configuration file for portfolio analysis
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Portfolio & data
DEFAULT_PORTFOLIO_VALUE = 10_000
DEFAULT_PERIOD = "5y"  # more history for stabler covariances
DEFAULT_INTERVAL = "1d"
DEFAULT_FREQUENCY = 252  # use consistently in PPO calls

# Return & risk models (pick in code)
RETURN_MODEL = "ewma"  # or "mean"; for ewma use span ~60–90 trading days
COVARIANCE_MODEL = "ledoit_wolf"  # shrinkage; or "oas"; or "semicov" for downside risk

# Optimization
DEFAULT_L2_REG = 1e-3  # PPO's L2 regularization (gamma-like)
DEFAULT_WEIGHT_BOUNDS = (0, 1)
MAX_WEIGHT = 0.60  # optional cap to avoid concentration
SECTOR_NEUTRAL = False  # if you have sector labels, can add constraints

# Risk-free & scaling
DEFAULT_RISK_FREE_RATE = 0.02  # annual; keep synced to T-bills

# Plotting settings
FIGURE_SIZE = (12, 8)
DPI = 300
