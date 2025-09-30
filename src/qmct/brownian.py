import numpy as np
from rng import set_seed

def brownian_paths(n_paths: int, n_steps: int, T: float = 1.0) -> np.ndarray:
    """
    Simulate n_paths of standard brownian motion up to time T
    Returns an array of shape (n_paths,  n_steps+1)
    """
    dt = T / n_steps #time step
    dW = np.sqrt(dt) *np.random.randn(n_paths, n_steps)

    #cumulative sum across steps
    W = np.cumsum(dW, axis=1)

    #prepend 0 at t=0
    zeros = np.zeros((n_paths, 1))
    W = np.concatenate([zeros, W], axis=1)

    return W
if __name__ == "__main__":
    set_seed(42)
    
    paths = brownian_paths(n_paths=3, n_steps=10, T=1.0)
    print("Brownian motion paths (3x11):")
    print(paths)
