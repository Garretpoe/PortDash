import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, expected_returns, risk_models


def run_optimizer(TICKERS, weight_constraints, risk_free_rate=0.03, period="5y"):

    # Download prices
    prices = yf.download(TICKERS, period=period, auto_adjust=True, progress=False)["Close"].dropna()

    # Define asset types 

    # Expected returns & covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    weight_bounds = [(weight_constraints[ticker][0], weight_constraints[ticker][1]) for ticker in TICKERS]


    # Efficient Frontier
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    # Optimize for max Sharpe
    ef.max_sharpe(risk_free_rate=risk_free_rate)

    weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)


    return {
        "weights": weights,
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "prices": prices
    }

def monte_carlo_simulation(prices,weights,initial_investment, n_sims= 100, n_days=252):
    returns = prices.pct_change().dropna()
    weights = np.array(list(weights.values()))
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    #Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    n_assets = len(prices.columns)
    sims = np.zeros((n_days,n_sims))
    sims[0] = initial_investment 

    for s in range(n_sims):
        portfolio_paths = np.zeros(n_days)
        portfolio_paths[0] = initial_investment

        for t in range(1, n_days):
            z = np.random.normal(size=(n_assets))
            shocks = z @ L.T + mean_returns
            portfolio_returns = shocks @ weights
            portfolio_paths[t] = portfolio_paths[t-1] * (1 + portfolio_returns)
        sims[:,s] = portfolio_paths
    return sims















