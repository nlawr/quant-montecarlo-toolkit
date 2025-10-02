# quant-montecarlo-toolkit
Learning Python + Monte Carlo Simulation for Quant Finance

**Day 1**
Set up project repository and file structure (src/qmct/)
Implemented reproducible random number generator (rng.py) using NumPy’s default_rng with seed control
Generated and verified random samples for consistency across runs
Reviewed purpose of random seeds in simulations (ensuring reproducibility for testing/validation)

**Day 2**
Implemented Geometric Brownian Motion (GBM) simulator (gbm.py) for stock price paths
Visualized sample GBM trajectories and verified lognormal distribution of terminal prices
Built Monte Carlo European call option pricer (bs_pricing.py)
Compared Monte Carlo results to Black–Scholes closed-form solution — confirmed agreement within 95% confidence interval

**Day 3**
Implemented Monte Carlo European push option pricer
Implemented antithetics for the Monte Carlo Models
Began work on Greeks section of code to explore how varying values change the simulation

**Day 4**
Added Delta, Gamma, Vega, Theta, and Rho Greeks to code
Restructured greeks.py
Added Monte Carlo delta bump

