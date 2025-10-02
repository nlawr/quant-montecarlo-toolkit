import numpy as np
from math import log, sqrt, exp
from rng import set_seed
from gbm import gbm_paths, gbm_apaths
from scipy.stats import norm

# ----- Black-Scholes closed-form (risk-neutral) -----
# Assumes continuous compounding and no dividends.

def bs_call_closed(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """European call price by Black-Scholes."""

    d1 = (log(S0/K)+ (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def bs_put_closed(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """European put price by Black-Scholes."""

    d1 = (log(S0/K)+ (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T ) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

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

def mc_put_price(
        S0: float, K: float, r: float, sigma: float, T: float,
        n_paths: int = 100_000, n_steps: int=100, seed: int | None = 42
):
    """
    Monte Carlo price for a European put using GBM
    Returns (price, 95% CI half-width)
    """
    if seed is not None:
        set_seed(seed)

    #simulate terminal prices
    paths = gbm_paths(S0=S0, mu=r, sigma=sigma, n_paths=n_paths, n_steps=n_steps, T=T)
    ST = paths[:,-1]

    # payoff at maturity
    payoff = np.maximum(K - ST, 0.0)

    # risk-neutral discounting
    disc = np.exp(-r * T)
    price = disc * np.mean(payoff)

    # 95% confidence interval half-width (1.96 * stderr)
    stderr = disc * np.std(payoff, ddof=1) / np.sqrt(n_paths)
    ci95 = 1.96 * stderr
    return price, ci95

def mc_call_aprice(
        S0: float, K: float, r: float, sigma: float, T: float,
        n_paths: int = 100_000, n_steps: int=100, seed: int | None = 42
):
    """
    Monte Carlo price for a European call using GBM
    Returns (price, 95% CI half-width, Antithetic)
    """
    if seed is not None: 
        set_seed(seed)
    
    #simulate terminal prices
    ST = gbm_apaths(S0=S0, mu=r, sigma=sigma, n_paths=n_paths, n_steps=n_steps, T=T)

    # payoff at maturity
    payoff = np.maximum(ST - K, 0.0)

    # risk-neutral discounting
    disc = np.exp(-r * T)
    price = disc * np.mean(payoff)

    # 95% confidence interval half-width (1.96 * stderr)
    stderr = disc * np.std(payoff, ddof=1) / np.sqrt(2*n_paths)
    ci95 = 1.96 * stderr
    return price, ci95

def mc_put_aprice(
        S0: float, K: float, r: float, sigma: float, T: float,
        n_paths: int = 100_000, n_steps: int=100, seed: int | None = 42
):
    """
    Monte Carlo price for a European put using GBM
    Returns (price, 95% CI half-width, Antithetic)
    """
    if seed is not None:
        set_seed(seed)

    #simulate terminal prices
    ST = gbm_apaths(S0=S0, mu=r, sigma=sigma, n_paths=n_paths, n_steps=n_steps, T=T)

    # payoff at maturity
    payoff = np.maximum(K - ST, 0.0)

    # risk-neutral discounting
    disc = np.exp(-r * T)
    price = disc * np.mean(payoff)

    # 95% confidence interval half-width (1.96 * stderr)
    stderr = disc * np.std(payoff, ddof=1) / np.sqrt(2*n_paths)
    ci95 = 1.96 * stderr
    return price, ci95

if __name__ == "__main__":
# Ex: At 1 Year
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0

# Black-Scholes
    bs_call=bs_call_closed(S0, K, r, sigma, T)
    bs_put = bs_put_closed(S0, K, r, sigma, T)

    print (f"BS call: {bs_call:.4f}")
    print (f"BS put: {bs_put:.4f}")

# Monte-Carlo
    mc_c_price, mc_c_ci = mc_call_price(S0, K, r, sigma, T, n_paths=200_000, n_steps=200, seed=42)
    mc_p_price, mc_p_ci = mc_put_price(S0, K, r, sigma, T, n_paths=200_000, n_steps=200, seed=42)

    print (f"MC call: {mc_c_price:.4f} +- {mc_c_ci:.4f} (95% CI)")
    print (f"MC put: {mc_p_price:.4f} +- {mc_p_ci:.4f} (95% CI)")

# Anthithetic Monte-Carlo
    mc_c_aprice, mc_c_ci = mc_call_aprice(S0, K, r, sigma, T, n_paths=200_000, n_steps=200, seed=42)
    mc_p_aprice, mc_p_ci = mc_put_aprice(S0, K, r, sigma, T, n_paths=200_000, n_steps=200, seed=42)

    print (f"MC anthithetic call: {mc_c_aprice:.4f} +- {mc_c_ci:.4f} (95% CI)")
    print (f"MC antithetic put: {mc_p_aprice:.4f} +- {mc_p_ci:.4f} (95% CI)")

# Put-Call Parity
lhs = bs_call - bs_put
rhs = S0 - K * np.exp(-r*T)
print(f"Parity check: LHS={lhs:.4f}, RHS={rhs:.4f}")