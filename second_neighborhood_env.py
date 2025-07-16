import gymnasium as gym
from gymnasium import spaces
import numpy as np

def calculate_conjecture_score(adj_matrix: np.ndarray) -> float:
    """
    Calculates a score for the second neighborhood conjecture, enforcing that
    all vertices must have an out-degree of at least 7.
    """
    A = adj_matrix
    n = A.shape[0]
    if n == 0:
        return -50000.0

    out_degrees = A.sum(axis=1)

    # --- NEW: Minimum Out-Degree Constraint ---
    # If any vertex has an out-degree less than 7, assign a massive penalty.
    # This guides the agent to only explore valid regions of the search space.
    MIN_OUT_DEGREE = 7
    if np.any(out_degrees < MIN_OUT_DEGREE):
        return -100000.0
    # --- End of New Constraint ---

    # --- Core graph calculations ---
    A_squared = A @ A
    A_reach_in_2 = (A_squared > 0).astype(int)
    N2_matrix = np.clip(A_reach_in_2 - A, 0, 1)

    second_neighborhood_sizes = N2_matrix.sum(axis=1)

    # --- Restored Penalty Logic ---
    diffs = second_neighborhood_sizes - out_degrees

    total_penalty = 0
    penalty_multiplier = 100
    sink_penalty = 5000 # This is now implicitly handled by the min-degree check but kept for consistency

    num_violating_vertices = np.sum(diffs < 0)
    total_penalty -= num_violating_vertices

    satisfying_diffs = diffs[diffs >= 0]
    total_penalty += np.sum(satisfying_diffs + 1) * penalty_multiplier

    num_sinks = np.sum(out_degrees == 0)
    total_penalty += num_sinks * sink_penalty

    return float(-total_penalty)


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