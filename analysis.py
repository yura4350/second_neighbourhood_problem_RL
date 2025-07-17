import numpy as np

def analyze_graph_properties(adj_matrix):
    """
    Calculates various properties of a directed graph from its adjacency matrix.

    For each vertex, it computes:
    - In-degree: Number of incoming edges.
    - Out-degree: Number of outgoing edges.
    - First Neighborhood (N1): Vertices directly reachable in one step.
    - Second Neighborhood (N2): Vertices reachable in two steps, but not one.

    Args:
        adj_matrix (list or np.ndarray): A square matrix representing the
                                         directed graph, where matrix[i][j] = 1
                                         indicates an edge from vertex i to j.

    Returns:
        dict: A dictionary where keys are vertex indices (e.g., 'vertex_0')
              and values are another dictionary containing the calculated
              properties for that vertex.
    """
    # Ensure the matrix is a NumPy array for easier calculations
    matrix = np.array(adj_matrix)
    num_vertices = matrix.shape[0]

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    graph_properties = {}

    for i in range(num_vertices):
        # --- Degree Calculation ---
        # In-degree: Sum of the column for vertex i
        in_degree = int(np.sum(matrix[:, i]))

        # Out-degree: Sum of the row for vertex i
        out_degree = int(np.sum(matrix[i, :]))

        # --- First Neighborhood (N1) ---
        # These are the vertices j where there is an edge from i to j.
        # We find the indices where the row for vertex i has a 1.
        first_neighborhood = np.where(matrix[i, :] == 1)[0].tolist()
        num_in_first_neighborhood = len(first_neighborhood)
        
        # --- Second Neighborhood (N2) ---
        # These are vertices reachable in two steps, but not in one.
        second_neighborhood = set()
        # Iterate through the direct neighbors of vertex i
        for neighbor in first_neighborhood:
            # Find the neighbors of the current neighbor
            two_steps_away = np.where(matrix[neighbor, :] == 1)[0]
            # Add these to our set of potential second neighbors
            second_neighborhood.update(two_steps_away)

        # Exclude the starting vertex itself from its own neighborhood
        second_neighborhood.discard(i)
        
        # Exclude all vertices that are already in the first neighborhood
        final_second_neighborhood = sorted(list(second_neighborhood - set(first_neighborhood)))
        num_in_second_neighborhood = len(final_second_neighborhood)

        # --- Store Results ---
        graph_properties[f'vertex_{i}'] = {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'num_in_first_neighborhood': num_in_first_neighborhood,
            'first_neighborhood_vertices': first_neighborhood,
            'num_in_second_neighborhood': num_in_second_neighborhood,
            'second_neighborhood_vertices': final_second_neighborhood
        }

    return graph_properties

def analyze_second_neighborhood_conjecture(graph_properties):
    """
    Analyzes the Second Neighborhood Conjecture for each vertex.

    The conjecture states that for a directed graph, the size of the second
    neighborhood should be greater than or equal to the size of the first.

    Args:
        graph_properties (dict): The output from analyze_graph_properties.

    Returns:
        dict: A dictionary containing lists of vertices that satisfy or
              violate the conjecture, and their respective counts.
    """
    satisfying_vertices = []
    violating_vertices = []

    for vertex_name, properties in graph_properties.items():
        vertex_index = int(vertex_name.split('_')[1])
        n1_size = properties['num_in_first_neighborhood']
        n2_size = properties['num_in_second_neighborhood']

        if n2_size >= n1_size:
            satisfying_vertices.append(vertex_index)
        else:
            violating_vertices.append(vertex_index)
            
    return {
        'satisfying_count': len(satisfying_vertices),
        'satisfying_vertices': satisfying_vertices,
        'violating_count': len(violating_vertices),
        'violating_vertices': violating_vertices
    }

def print_analysis_results(results):
    """Nicely prints the analysis results for each vertex."""
    for vertex, properties in results.items():
        print("-" * 40)
        print(f"Analysis for {vertex.replace('_', ' ').title()}:")
        print("-" * 40)
        print(f"  In-Degree: {properties['in_degree']}")
        print(f"  Out-Degree: {properties['out_degree']}")
        print("\n  --- First Neighborhood (N1) ---")
        print(f"  Number of vertices: {properties['num_in_first_neighborhood']}")
        print(f"  Vertices: {properties['first_neighborhood_vertices']}")
        print("\n  --- Second Neighborhood (N2) ---")
        print(f"  Number of vertices: {properties['num_in_second_neighborhood']}")
        print(f"  Vertices: {properties['second_neighborhood_vertices']}")
        print("\n")

def print_conjecture_results(conjecture_analysis):
    """Nicely prints the Second Neighborhood Conjecture results."""
    print("=" * 50)
    print("Second Neighborhood Conjecture Analysis")
    print("=" * 50)
    print(f"Conjecture: |N2(v)| >= |N1(v)|")
    print("-" * 50)
    print(f"Number of vertices SATISFYING the conjecture: {conjecture_analysis['satisfying_count']}")
    print(f"Vertices that satisfy: {conjecture_analysis['satisfying_vertices']}")
    print("-" * 50)
    print(f"Number of vertices VIOLATING the conjecture: {conjecture_analysis['violating_count']}")
    print(f"Vertices that violate: {conjecture_analysis['violating_vertices']}")
    print("=" * 50)


# Example adjacency matrix provided by the user
adj_matrix_example = [
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]
]

if __name__ == "__main__":
    # Analyze the basic graph properties
    analysis = analyze_graph_properties(adj_matrix_example)
    
    # Print the detailed results for each vertex
    print_analysis_results(analysis)
    
    # Analyze the second neighborhood conjecture based on the results
    conjecture_results = analyze_second_neighborhood_conjecture(analysis)
    
    # Print the summary of the conjecture analysis
    print_conjecture_results(conjecture_results)
