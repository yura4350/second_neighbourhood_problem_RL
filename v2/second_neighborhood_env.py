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
    """
    metadata = {"render_modes": []}

    def __init__(self, num_nodes, min_out_degree=7):
        super().__init__()
        self.num_nodes = num_nodes
        self.min_out_degree = min_out_degree
        self.num_pairs = self.num_nodes * (self.num_nodes - 1) // 2
        
        self.action_space = spaces.MultiDiscrete([3] * self.num_pairs)
        self.observation_space = spaces.MultiBinary(self.num_nodes * self.num_nodes)
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)

    def _decode_action(self, action_vector: np.ndarray) -> np.ndarray:
        """
        Decodes the compact action vector into a full n x n adjacency matrix.
        Convention: Adjacency matrix A[i,j]=1 means edge from j to i.
        """
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)
        action_idx = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                choice = action_vector[action_idx]
                # Action choice 1 for pair (i,j) means edge i -> j
                if choice == 1:
                    A[j, i] = 1
                # Action choice 2 for pair (i,j) means edge j -> i
                elif choice == 2:
                    A[i, j] = 1
                action_idx += 1
        return A

    def _generate_valid_adj_matrix(self) -> np.ndarray:
        """
        Generates a valid starting adjacency matrix where every vertex
        has an out-degree of at least `min_out_degree`.
        Convention: A.sum(axis=1) is the out-degree. So edge i->j is A[i,j]=1.
        """
        n = self.num_nodes
        k = self.min_out_degree
        A = np.zeros((n, n), dtype=np.uint8)
        
        # We must use a different convention here temporarily for easier logic,
        # then transpose at the end. Here, A[i,j]=1 means edge i->j.
        
        while True:
            fixes_made = 0
            for v in range(n):
                out_degree = A[v, :].sum()
                while out_degree < k:
                    fixes_made += 1
                    
                    # Systematically find a valid neighbor to add an edge to.
                    targets = np.random.permutation(n)
                    found_target = False
                    for u in targets:
                        # Check if target u is valid
                        if u != v and A[v, u] == 0 and A[u, v] == 0:
                            A[v, u] = 1
                            out_degree += 1
                            found_target = True
                            break
                    
                    if not found_target:
                        # This should only happen if graph is nearly complete.
                        break
            
            if fixes_made == 0:
                # Transpose the matrix to match the environment's convention (j -> i)
                return A.T

    def step(self, action):
        self.adj_matrix = self._decode_action(action)
        reward = calculate_conjecture_score(self.adj_matrix)
        observation = self.adj_matrix.flatten()
        terminated = False
        info = {'violation_score': reward}

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Generate an adjacency matrix that is guaranteed to be valid.
        self.adj_matrix = self._generate_valid_adj_matrix()

        # 2. Reset the agent's state based on this valid matrix.
        observation = self.adj_matrix.flatten()
        initial_score = calculate_conjecture_score(self.adj_matrix)
        info = {'violation_score': initial_score}
        
        return observation, info