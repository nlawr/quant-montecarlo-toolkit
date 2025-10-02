# greeks.py
# Clean, extensible Greeks utilities for Black-Scholes
# - Vectorized via NumPy
# - Supports dividend yield q
# - Common helpers for d1, d2, and payoffs
# - MC Delta via antithetic bump w/ common random numbers

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from typing import Literal, Tuple
from rng import set_seed

OptionType = Literal["call", "put"]

# ----------------------------------------------
# Helpers
# ----------------------------------------------
def _d1_d2(S0: float | np.ndarray,
           K: float | np.ndarray,
           r: float | np.ndarray,
           sigma: float | np.ndarray,
           T: float | np.ndarray,
           q: float | np.ndarray = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute d1, d2 for Black-Scholes with dividend yield q.
    Works on scalars or arrays; returns arrays.
    """
    S0 = np.asarray(S0, dtype=float)
    K = np.asarray(K, dtype=float)
    r = np.asarray(r, dtype=float)
    s = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    q = np.asarray(q, dtype=float)

    eps = 1e-12
    s = np.maximum(s, eps)
    T = np.maximum(T, eps)

    num = np.log(np.maximum(S0, eps) / np.maximum(K, eps))+ (r - q + 0.5 * s**2) * T
    den = s * np.sqrt(T)
    d1 = num/den
    d2 = d1- s * np.sqrt(T)
    return d1, d2

def _payoff(ST: np.ndarray, K: float, option: OptionType) -> np.ndarray:
    if option == "call":
        return np.maximum (ST - K, 0.0)
    else:
        return np.maximum (K - ST, 0.0)

def _gbm_ST_from_Z(S0: float,
                   r: float,
                   sigma: float,
                   T: float,
                   Z_full: np.ndarray,
                   q: float = 0.0) -> np.ndarray:
    """
    Multi-step GBM terminal prices using provided normal shocks (including anthitetics).
    Uses risk-neutral drift (r - q)
    """
    n_steps = Z_full.shape[1]
    dt = T / n_steps
    inc = (r - q -0.5 * sigma**2) *dt + sigma * np.sqrt(dt) * Z_full
    log_S = np.cumsum(inc, axis=1)
    log_S = np.concatenate([np.zeros((Z_full.shape[0],1)), log_S], axis=1)
    S = S0 * np.exp(log_S)
    return S[:, -1]

# ----------------------------------------------
# Closed-form Greeks
# ----------------------------------------------
def bs_delta_closed(S0: float, K: float, r: float, sigma: float, T: float,
                    option: OptionType = "call", q: float = 0.0) -> float:
    """
    Black-Scholes Delta with divident yield q.
    Returns a scalar
    """
    d1, _ = _d1_d2(S0, K, r, sigma, T, q)
    if option == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    else:
        return float(np.exp(-q * T) * (norm.cdf(d1)- 1.0))

def bs_gamma_closed(S0, K, r, sigma, T, q=0.0) -> float:
    """
    Black–Scholes Gamma (same for call & put).
    Γ = e^{-qT} * φ(d1) / (S0 * σ * sqrt(T))
    """
    d1, _ = _d1_d2(S0, K, r, sigma, T, q)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S0 * sigma * np.sqrt(T)))

def bs_vega_closed(S0, K, r, sigma, T, q=0.0) -> float:
    """
    Black–Scholes Vega (same for call & put), per 1.00 volatility (i.e., 100% change).
    For per 1% vol point, use bs_vega_closed(...)/100.
    Vega = S0 * e^{-qT} * φ(d1) * sqrt(T)
    """
    d1, _ = _d1_d2(S0, K, r, sigma, T, q)
    return float (S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))

def bs_theta_closed(S0, K, r, sigma, T, option: OptionType = "call", q=0.0) -> float:
    """
    Black–Scholes Theta (time decay) with dividend yield q.
    """
    d1, d2 = _d1_d2(S0, K, r, sigma, T, q)
    term1 = - (S0 * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S0 * np.exp(-q * T) * norm.cdf(d1)
        return float(term1 + term2 + term3)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S0 * np.exp(-q * T) * norm.cdf(-d1)
        return float(term1 + term2 + term3)

def bs_rho_closed(S0, K, r, sigma, T, option: OptionType = "call", q = 0.0) -> float:
    """
    Black-Scholes Rho (sensitivity to interest rate).
    """
    _, d2 = _d1_d2(S0, K, r, sigma, T, q)
    if option == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(-K * T * np.exp(-r * T) * norm.cdf(-d2))

def mc_delta_bump(S0: float,
           K: float, r: float, sigma: float, T: float,
           option: OptionType = "call",
           n_paths: int = 100_000, n_steps: int = 100, eps: float = 1e-2, seed: int | None = 42,
           q: float = 0.0
           ) -> tuple[float, float]:
    """
    Monte Carlo Delta via central difference on S0 with:
    - common random numbers
    - antithetic variates
    Returns (delta_estimate, 95% CI half-width)
    """
    if seed is not None:
        set_seed(seed)
    
    # common normals + antithetics
    Z = np.random.randn(n_paths, n_steps)
    Z_full = np.vstack([Z, -Z])

    disc = np.exp(-r * T)

    # Terminal prices under same Z_full for S0±eps
    ST_up = _gbm_ST_from_Z(S0 + eps, r, sigma, T, Z_full, q=q)
    ST_dn = _gbm_ST_from_Z(S0 - eps, r, sigma, T, Z_full, q=q)

    # Discounted payoffs
    pay_up = disc * _payoff(ST_up, K, option)
    pay_dn = disc * _payoff(ST_dn, K, option)

    # Per-path delta estimates (paired)
    delta_samples = (pay_up - pay_dn) / (2.0 * eps)

    # MC estimate + CI
    delta = float(np.mean(delta_samples))
    stderr = np.std(delta_samples, ddof=1) / np.sqrt(delta_samples.shape[0])
    ci95 = 1.96 * stderr
    return delta, ci95
# -----------------------------------
# Self Check
# -----------------------------------
if __name__ == "__main__":
    S0, K, r, sigma, T, q = 100.0, 100.0, 0.02, 0.20, 1.0, 0.0

    # closed-form
    d_call = bs_delta_closed(S0, K, r, sigma, T, option="call", q=q)
    d_put  = bs_delta_closed(S0, K, r, sigma, T, option="put",  q=q)
    g_bs = bs_gamma_closed(S0, K, r, sigma, T, q)
    v_bs = bs_vega_closed(S0, K, r, sigma, T, q)
    theta_c = bs_theta_closed(S0, K, r, sigma, T, option="call", q=q)
    theta_p = bs_theta_closed(S0, K, r, sigma, T, option="put", q=q)
    rho_c   = bs_rho_closed(S0, K, r, sigma, T, option="call", q=q)
    rho_p   = bs_rho_closed(S0, K, r, sigma, T, option="put", q=q)

    print(f"Delta (BS) call: {d_call:.4f} | put: {d_put:.4f}")
    print(f"Gamma (BS): {g_bs:.6f}")
    print(f"Vega  (BS, per 1.00 vol): {v_bs:.4f}  | per 1% vol: {v_bs/100:.4f}")
    print(f"Theta (call): {theta_c:.4f} | Theta (put): {theta_p:.4f}")
    print(f"Rho (call): {rho_c:.4f} | Rho (put): {rho_p:.4f}")

    # MC delta (bump) – call
    d_mc, ci = mc_delta_bump(S0, K, r, sigma, T, option="call",
                             n_paths=100_000, n_steps=200, eps=0.01, seed=42, q=q)
    print(f"Delta (MC bump, call): {d_mc:.4f} ± {ci:.4f}")