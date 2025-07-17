import gymnasium as gym
from gymnasium import spaces
import numpy as np

def calculate_conjecture_score(adj_matrix: np.ndarray) -> float:
    """
    Calculates a score for a given graph. The score is designed to guide the
    agent towards a valid counterexample by applying penalties for
    constraint violations and rewarding progress towards the goal.
    """
    A = adj_matrix
    n = A.shape[0]
    if n == 0:
        return -500000.0 # Penalty for an empty graph

    # Start with a neutral score and apply penalties/rewards.
    total_score = 0.0

    # --- Constraint Penalties ---

    # 1. Graded penalty for each 2-cycle.
    # The number of 2-cycles is the sum of the diagonal of A^2.
    num_2_cycles = np.trace(A @ A)
    total_score -= num_2_cycles * 5000.0

    ## CHANGED: Reinstated the hard penalty for the minimum out-degree constraint.
    # This treats the minimum out-degree as a strict, non-negotiable rule.
    out_degrees = A.sum(axis=1)
    MIN_OUT_DEGREE = 7
    if np.any(out_degrees < MIN_OUT_DEGREE):
        return -1000000.0 # Heavy penalty for not meeting the minimum out-degree

    # --- Conjecture Scoring ---

    # A_squared[i, j] counts walks of length 2 from j to i.
    A_squared = A @ A
    A_reach_in_2 = (A_squared > 0).astype(int)
    N2_matrix = np.clip(A_reach_in_2 - A, 0, 1)
    second_neighborhood_sizes = N2_matrix.sum(axis=1)

    # diffs < 0 means |N2| < |N|, which is a "violating" vertex (good for us).
    diffs = second_neighborhood_sizes - out_degrees

    # Penalize each vertex that satisfies the conjecture (|N2| >= |N|).
    num_satisfying_vertices = np.sum(diffs >= 0)
    total_score -= num_satisfying_vertices * 100.0

    # Reward each vertex that violates the conjecture (|N2| < |N|).
    num_violating_vertices = np.sum(diffs < 0)
    total_score += num_violating_vertices

    return float(total_score)

class SecondNeighborhoodEnv(gym.Env):
    """
    RL environment to find counterexamples to the Second Neighborhood Conjecture
    using a DISCRETE action space (PPO) and enforcing a minimum out-degree of 7.
    """
    metadata = {"render_modes": []}

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.observation_space = spaces.MultiBinary(num_nodes * num_nodes)
        self.action_space = spaces.MultiBinary(num_nodes * num_nodes)
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.uint8)

    def step(self, action):
        self.adj_matrix = action.reshape(self.num_nodes, self.num_nodes)
        np.fill_diagonal(self.adj_matrix, 0)

        reward = calculate_conjecture_score(self.adj_matrix)
        
        observation = self.adj_matrix.flatten()
        info = {'violation_score': reward}
        terminated = False
        
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.adj_matrix = self.np_random.integers(0, 2, size=(self.num_nodes, self.num_nodes), dtype=np.uint8)
        np.fill_diagonal(self.adj_matrix, 0)
        
        observation = self.adj_matrix.flatten()
        initial_score = calculate_conjecture_score(self.adj_matrix)
        info = {'violation_score': initial_score}
        
        return observation, info