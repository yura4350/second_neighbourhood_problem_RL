import gymnasium as gym
from gymnasium import spaces
import numpy as np

def calculate_conjecture_score(adj_matrix: np.ndarray) -> float:
    A = adj_matrix
    n = A.shape[0]
    if n == 0:
        return -50000.0

    total_score = 0.0

    # Calculate the penalty based on the number of vertices with out-degree <= 6.
    out_degrees = A.sum(axis=1)
    MIN_OUT_DEGREE = 7
    num_low_degree_vertices = np.sum(out_degrees < MIN_OUT_DEGREE)
    # Apply a penalty for each vertex that violates the constraint.
    total_score -= num_low_degree_vertices * 2000.0

    # Conjecture Scoring
    A_squared = A @ A
    A_reach_in_2 = (A_squared > 0).astype(int)
    N2_matrix = np.clip(A_reach_in_2 - A, 0, 1)
    second_neighborhood_sizes = N2_matrix.sum(axis=1)

    diffs = second_neighborhood_sizes - out_degrees

    num_satisfying_vertices = np.sum(diffs >= 0)
    total_score -= num_satisfying_vertices * 50.0

    num_violating_vertices = np.sum(diffs < 0)
    total_score += num_violating_vertices

    return float(total_score)


class SecondNeighborhoodEnv(gym.Env):
    """
    RL environment that finds counterexamples by building a graph that is
    structurally guaranteed to have no 2-cycles.

    - Action Space: For each pair of vertices {i, j}, choose one of three options:
        0: No edge
        1: Edge from i to j
        2: Edge from j to i
    """
    metadata = {"render_modes": []}

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        # The number of unique pairs of vertices is n * (n-1) / 2
        self.num_pairs = self.num_nodes * (self.num_nodes - 1) // 2

        ## CHANGED: The action space is now smarter.
        # It's a vector of length `num_pairs`, where each element can be 0, 1, or 2.
        self.action_space = spaces.MultiDiscrete([3] * self.num_pairs)

        # The observation is still the standard adjacency matrix for consistency.
        self.observation_space = spaces.MultiBinary(num_nodes * num_nodes)
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)

    def _decode_action(self, action_vector: np.ndarray) -> np.ndarray:
        """
        Decodes the compact action vector into a full n x n adjacency matrix.
        """
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)
        action_idx = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                choice = action_vector[action_idx]
                if choice == 1:  # Your definition: 1 = edge from i to j
                    # Our convention: A[row, col]=1 means edge col -> row
                    # So, for an edge i -> j, we set A[j, i] = 1
                    A[j, i] = 1
                elif choice == 2:  # Your definition: 2 = edge from j to i
                    A[i, j] = 1
                # If choice is 0, we do nothing (no edge).
                action_idx += 1
        return A

    def step(self, action):
        ## CHANGED: Decode the action vector to build the matrix.
        self.adj_matrix = self._decode_action(action)
        # NOTE: No need to check for self-loops or 2-cycles. They can't be created.

        reward = calculate_conjecture_score(self.adj_matrix)
        observation = self.adj_matrix.flatten()
        terminated = False
        info = {'violation_score': reward}

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ## CHANGED: Generate a random valid action and decode it.
        random_action = self.action_space.sample()
        self.adj_matrix = self._decode_action(random_action)

        observation = self.adj_matrix.flatten()
        initial_score = calculate_conjecture_score(self.adj_matrix)
        info = {'violation_score': initial_score}

        return observation, info