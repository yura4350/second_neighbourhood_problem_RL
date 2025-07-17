import numpy as np
import networkx as nx
from time import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Make sure this is the "smarter" version of the environment file
from second_neighborhood_env import SecondNeighborhoodEnv, calculate_conjecture_score


class SaveBestGraphCallback(BaseCallback):
    """
    A callback to save the graph with the best score found during training,
    correctly handling vectorized (parallel) environments.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_score = -float('inf')
        self.best_adj_matrix = None

    def _on_step(self) -> bool:
        # For vectorized environments, self.locals['infos'] is a list of dicts.
        for i, info in enumerate(self.locals['infos']):
            current_score = info.get('violation_score')
            if current_score is not None and current_score > self.best_score:
                self.best_score = current_score
                # Get the adjacency matrix from the specific environment (env i) that got the best score
                self.best_adj_matrix = self.training_env.get_attr('adj_matrix', indices=[i])[0].copy()
                if self.verbose > 0:
                    print(f"\nNew best score: {self.best_score:.2f} (from env {i} at timestep {self.num_timesteps})")
        return True

# --- Main Configuration ---
NUM_NODES = 25
# Use a higher number of timesteps because parallel training is much faster
TOTAL_TIMESTEPS = 200_000
# Set to the number of CPU cores you want to use for parallel training
NUM_CPU = 6

# 1. Create the vectorized environment to run environments in parallel
vec_env = make_vec_env(lambda: SecondNeighborhoodEnv(NUM_NODES), n_envs=NUM_CPU)

# 2. Define the PPO model with tuned hyperparameters for faster learning
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=5e-4,     # Slightly higher learning rate
    n_steps=512,            # More frequent policy updates
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,          # Encourage exploration to escape local optima
    verbose=0               # Set to 1 to see PPO's training logs
)

# 3. Set up the callback to save the best result
save_best_callback = SaveBestGraphCallback(verbose=1)


print("--- Searching for Counterexamples with Parallel PPO ---")
print(f"--- Using {NUM_CPU} parallel environments ---")

# 4. Train the model
start_time = time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=save_best_callback)
end_time = time()
print(f"\nTotal search time: {end_time - start_time:.2f} seconds")

# Final Results & Analysis
if save_best_callback.best_adj_matrix is not None:

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

   # Analysis & Visualization of the Best Graph
    print("\nVisualizing the best graph found...")

    # Recalculate values
    A_squared = A @ A
    N2_matrix = np.clip((A_squared > 0).astype(int) - A, 0, 1)
    out_degrees = A.sum(axis=1)
    second_neighborhood_sizes = N2_matrix.sum(axis=1)
    violations = out_degrees - second_neighborhood_sizes # |N| - |N2|

    color_map = [''] * G.number_of_nodes()
    for node in G:
        # A positive violation score means |N| > |N2|
        if violations[node] > 0:
            color_map[node] = 'yellow'  # Violating node
        else:
            color_map[node] = 'red'     # Satisfying node

    num_satisfying = np.sum(violations <= 0)
    num_violating = np.sum(violations > 0)

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=700,
            font_color='black', edge_color='gray', arrows=True, arrowstyle='->',
            connectionstyle='arc3,rad=0.1')

    plt.title(f"Graph Analysis: {num_satisfying} Satisfying Nodes (Red), {num_violating} Violating Nodes (Yellow)")
    plt.show()