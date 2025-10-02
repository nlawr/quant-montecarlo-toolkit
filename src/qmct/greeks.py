import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
from rng import set_seed

# --- reference (closed-form) delta for a European call ---
def bs_call_delta_closed(S0, K, r, sigma, T):
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return norm.cdf(d1)

# --- helper: build terminal prices with antithetic Z and reuse them across bumps ---
def _gbm_ST_from_Z(S0, r, sigma, T, Z_full):
    n_steps = Z_full.shape[1]
    dt = T / n_steps
    inc = (r-0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_full
    log_S = np.cumsum(inc, axis=1)
    log_S = np.concatenate([np.zeros((Z_full.shape[0], 1)), log_S], axis=1)
    S = S0 * np.exp(log_S)
    return S[:, -1]

def mc_call_delta_bump(
    S0, K, r, sigma, T, 
    n_paths=100_000, n_steps=100, eps=1e-2, seed=42
):
    """
    Monte Carlo Delta (European call) via central difference with
    - common random numbers
    antithetic variates
    Returns (delta_estimate, 95% CI half-width)
    """
    if seed is not None: 
        set_seed(seed)
    
    # common random numbers + antithetics
    Z = np.random.randn(n_paths, n_steps)
    Z_full = np.vstack([Z, -Z])
    disc = np.exp(-r * T)

    # terminal prices for S0+eps and S0-eps using the SAME Z_full
    ST_up = _gbm_ST_from_Z(S0 + eps, r, sigma, T, Z_full)
    ST_dn = _gbm_ST_from_Z(S0 - eps, r, sigma, T, Z_full)

    # discounted payoffs for each path
    pay_up = disc * np.maximum(ST_up - K, 0.0)
    pay_dn = disc * np.maximum(ST_dn - K, 0.0)

    # per-path delta estimates (paired difference)
    delta_samples = (pay_up - pay_dn) / (2.0 * eps)

    # MC estimate + CI
    delta = np.mean(delta_samples)
    stderr = np.std(delta_samples, ddof=1) / np.sqrt(delta_samples.shape[0])
    ci95 = 1.96 * stderr
    return delta, ci95

if __name__ == "__main__":
    # Quick self-check
    S0, K, r, sigma, T = 100.0, 100.0, 0.02, 0.20, 1.0
    delta_mc, delta_ci = mc_call_delta_bump(S0, K, r, sigma, T, n_paths=100_000, n_steps=200, eps=0.01, seed=42)
    delta_bs = bs_call_delta_closed(S0, K, r, sigma, T)

    print(f"Delta (MC): {delta_mc:.4f} ± {delta_ci:.4f}")
    print(f"Delta (BS): {delta_bs:.4f}")