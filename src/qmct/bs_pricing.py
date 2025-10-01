import numpy as np
from math import log, sqrt, exp
from rng import set_seed
from gbm import gbm_paths

# ----- Black-Scholes closed-form (risk-neutral) -----
# Assumes continuous compounding and no dividends.

def bs_call_closed(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """European call price by Black-Scholes."""
    from math import log, sqrt, exp
    from scipy.stats import norm

    d1 = (log(S0/K)+ (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

# ----- Monte Carlo pricing using GBM simulator -----

def mc_call_price(
        S0: float, K: float, r: float, sigma: float, T: float,
        n_paths: int = 100_000, n_steps: int=100, seed: int | None = 42
):
    """
    Monte Carlo price for a European call using GBM
    Returns (price, 95% CI half-width)
    """
    if seed is not None: 
        set_seed(seed)
    
    #simulate terminal prices
    paths = gbm_paths(S0=S0, mu=r, sigma=sigma, n_paths=n_paths, n_steps=n_steps, T=T)
    ST = paths[:,-1]

    # payoff at maturity
    payoff = np.maximum(ST - K, 0.0)

    # risk-neutral discounting
    disc = np.exp(-r * T)
    price = disc * np.mean(payoff)

    # 95% confidence interval half-width (1.96 * stderr)
    stderr = disc * np.std(payoff, ddof=1) / np.sqrt(n_paths)
    ci95 = 1.96 * stderr
    return price, ci95

if __name__ == "__main__":
    # Example: at the money 1Y call
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0

    mc_price, mc_ci = mc_call_price(S0, K, r, sigma, T, n_paths=200_000, n_steps=200, seed=42)
    print (f"MC call: {mc_price:.4f} +- {mc_ci:.4f} (95% CI)")

    try:
        bs=bs_call_closed(S0, K, r, sigma, T)
        print(f"BS closed-form: {bs:.4f}")
    except Exception as e:
        print("Install scipy to compute closed form (pip install scipy.")