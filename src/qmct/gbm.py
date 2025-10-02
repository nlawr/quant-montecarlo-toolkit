import numpy as np
import matplotlib.pyplot as plt
from rng import set_seed

def gbm_paths(S0: float, mu: float, sigma: float, n_paths: int, n_steps: int, T: float = 1.0) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion (GBM) stock price paths.
    dS = mu*S*dt + sigma*S*dW
    """

    dt= T/n_steps

    # draw random increments ~ Normal(O, sqrt(dt))
    Z = np.random.randn(n_paths, n_steps)
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # cumulative sum of log returns
    log_S = np.cumsum(increments, axis=1)

    # prepend zeros (so we start at log(S0))
    log_S = np.concatenate([np.zeros((n_paths, 1)), log_S], axis=1)

    # exponentiate to get prices
    
    S = S0 * np.exp(log_S)

    return S

def gbm_apaths(S0: float, mu: float, sigma: float, n_paths: int, n_steps: int, T: float = 1.0) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion (GBM) stock price paths, with antithetic pairing.
    dS = mu*S*dt + sigma*S*dW
    """

    dt= T/n_steps

    Z = np.random.randn(n_paths, n_steps)
    Z_full = np.vstack([Z, -Z])1

    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_full
    log_S = np.cumsum(increments, axis=1)
    log_S = np.concatenate([np.zeros((Z_full.shape[0], 1)), log_S], axis=1)
    S = S0 * np.exp(log_S)
    ST = S[:, -1]
    return ST

if __name__ == "__main__":
    set_seed(42)

""" LINE GRAPH
paths = gbm_paths(S0=100, mu=0.05, sigma=0.2, n_paths=10, n_steps=100, T=1.0)
plt.plot(paths.T)
plt.xlabel("Step")
plt.ylabel("Stock Price")
plt.title("GBM Stock Price Simulation")
plt.show()
"""


""" HISTOGRAM
    paths = gbm_paths(S0=100, mu=0.05, sigma=0.2, n_paths=10000, n_steps=100, T=1.0)

    final_prices = paths[:, -1]

    plt.hist(final_prices, bins=50, density=True, alpha=0.7, color="blue")
    plt.xlabel("Final Stock Price")
    plt.ylabel("Density")
    plt.title("Distribuition of GBM FInal Prices (T=1)")
    plt.show()
"""
