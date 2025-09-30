import numpy as np

def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    np.random.seed(seed)

def randn(n: int):
    """Generate n standard normal random numbers."""
    return np.random.standard_normal(size=n)

if __name__ == "__main__":
    set_seed(42)
    samples = randn(5)
    print("Random samples:", samples)