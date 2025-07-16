import numpy as np
import networkx as nx
from time import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO  # CHANGE: Import PPO instead of SAC
from stable_baselines3.common.callbacks import BaseCallback

from second_neighborhood_env import SecondNeighborhoodEnv

class SaveBestGraphCallback(BaseCallback):
    """Callback to save the graph with the best violation score."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_score = -float('inf')
        self.best_adj_matrix = None

    def _on_step(self) -> bool:
        current_score = self.locals['infos'][0]['violation_score']
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_adj_matrix = self.training_env.get_attr('adj_matrix')[0].copy()
            if self.verbose > 0:
                print(f"\nNew best violation score: {self.best_score:.4f} (at step {self.num_timesteps})")
        return True

# --- Configuration ---
NUM_NODES = 16
TOTAL_TIMESTEPS = 500_000  # PPO often benefits from more steps

# --- Initialization ---
env = SecondNeighborhoodEnv(num_nodes=NUM_NODES)

# CHANGE: Use PPO with a policy designed for discrete actions
model = PPO("MlpPolicy", env, verbose=0)
save_best_callback = SaveBestGraphCallback(verbose=1)

obs, info = env.reset()
save_best_callback.best_score = info['violation_score']
save_best_callback.best_adj_matrix = env.adj_matrix.copy()

print("--- Searching for Counterexamples with PPO (Discrete Actions) ---")
print(f"--- Using digraphs with {NUM_NODES} nodes ---")
print(f"Initial Violation Score: {info['violation_score']:.4f}")

# --- RL Training ---
start_time = time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=save_best_callback)
end_time = time()
print(f"\nTotal search time: {end_time - start_time:.2f} seconds")

# --- Final Results & Analysis ---
if save_best_callback.best_adj_matrix is not None:
    # NOTE: The saved matrix is already the final discrete graph. No thresholding needed.
    A = save_best_callback.best_adj_matrix
    final_score = save_best_callback.best_score
    
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    print("\n--- RL Search Finished ---")
    print("Best discrete adjacency matrix found:")
    print(A)
    print(f"\nHighest Violation Score Found: {final_score:.4f}")
    
    if final_score > 0:
        print("\n\n*** POTENTIAL COUNTEREXAMPLE FOUND! *** 🏆")
        print("This graph has at least one vertex 'v' where |N_out(v)| > |N_out_2(v)|.")
    else:
        print("\nNo counterexample found. The conjecture holds for the best graph discovered.")

    # --- Analysis & Visualization of the Best Graph ---
    print("\nVisualizing the best graph found...")
    
    A_squared = A @ A
    N2_matrix = np.clip((A_squared > 0).astype(int) - A, 0, 1)
    out_degrees = A.sum(axis=1)
    second_neighborhood_sizes = N2_matrix.sum(axis=1)
    violations = out_degrees - second_neighborhood_sizes
    
    violating_node = np.argmax(violations)
    
    n1_nodes = list(G.successors(violating_node))
    n2_nodes = [i for i, val in enumerate(N2_matrix[violating_node]) if val > 0]

    color_map = ['skyblue'] * G.number_of_nodes()
    for node in G:
        if node in n2_nodes:
            color_map[node] = 'yellow'
        if node in n1_nodes:
            color_map[node] = 'orange'
    color_map[violating_node] = 'red'

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=700,
            font_color='black', edge_color='gray', arrows=True, arrowstyle='->',
            connectionstyle='arc3,rad=0.1')
    
    v_degree = out_degrees[violating_node]
    v_n2_degree = second_neighborhood_sizes[violating_node]
    plt.title(f"Best Graph Found (Node {violating_node}: |N_out|={v_degree}, |N_out_2|={v_n2_degree}) -> Violation: {v_degree - v_n2_degree}")
    plt.show()
else:
    print("\nTraining finished, but no improved graph was found.")