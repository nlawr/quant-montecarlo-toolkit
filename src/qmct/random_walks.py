import numpy as np
from rng import set_seed

def random_walk(n_steps: int, step_size: float = 1.0) -> np.ndarray:
    """
    Simple 1D coin-flip random walk starting at 0.
    Each step is +step_size or -step_size with equal probability
    Returns an array of length n_steps+1 including the starting 0.
    """
    #1) draw steps: -1 or +1 with 50/50 probability
    steps = np.random.choice([-1.0, 1.0], size=n_steps)

    #2) scale by step_size
    steps *= step_size

    #3) cumulative sum gives positions; include starting position 0
    positions = np.cumsum(steps)
    positions = np.concatenate([[0.0], positions])

    return positions

def random_walks(n_paths: int, n_steps: int, step_size: float = 1.0) -> np.ndarray:
    """
    Generate multiple 1D coin-flip random walks.
    Returns an array of shape (n_paths, n_steps+1)
    """
    # draw steps for all paths at once: -1 or +1
    steps = np.random.choice([-1.0, 1.0], size=(n_paths, n_steps))
    steps *= step_size

    #cumulative sum across steps -> axis=1
    positions = np.cumsum(steps, axis=1)

    #prepend 0 for the start
    zeros = np.zeros((n_paths, 1))
    positions = np.concatenate([zeros,positions], axis=1)

    return positions

if __name__ == "__main__":
    # demon: reproducible walk
    set_seed(42)

    # single path
    path = random_walk(n_steps=10, step_size=1.0)
    print("Random-walk positions (first 11):", path)
    
    #multiple paths
    paths = random_walks(n_paths=3, n_steps=10)
    print("Many paths shape:", paths.shape)
    print(paths)