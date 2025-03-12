import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import heapq
import math
from collections import defaultdict


def compute_rng(points):
    """Compute the Relative Neighborhood Graph for a set of points."""
    n = len(points)
    edges = []

    # Compute all pairwise distances
    dist_matrix = distance.cdist(points, points)

    # For each pair of points
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = dist_matrix[i, j]

            # Check if this edge should be in RNG
            is_rng_edge = True

            # Check the "lune" condition for all other points
            for k in range(n):
                if k != i and k != j:
                    # If any point k is closer to both i and j than they are to each other
                    if dist_matrix[i, k] < dist_ij and dist_matrix[j, k] < dist_ij:
                        is_rng_edge = False
                        break

            if is_rng_edge:
                edges.append((i, j))

    return edges, dist_matrix


def compute_sng(points):
    """Compute the Sparse Neighborhood Graph (SNG) for a set of points."""
    n = len(points)
    directed_edges = []

    # Compute all pairwise distances
    dist_matrix = distance.cdist(points, points)

    # For each point p
    for i in range(n):
        # Initialize set S of remaining points
        S = list(range(n))
        S.remove(i)

        # While S is not empty
        while S:
            # Find closest remaining point to i
            min_dist = float("inf")
            closest_idx = -1

            for j in S:
                if dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
                    closest_idx = j

            # Add directed edge from i to closest_idx
            directed_edges.append((i, closest_idx))

            # Remove from S all points farther from i than from closest_idx
            S_new = []
            for j in S:
                if j == closest_idx:
                    continue  # Skip the point we just connected to
                if dist_matrix[i, j] <= dist_matrix[closest_idx, j]:
                    S_new.append(j)  # Keep points not pruned
            S = S_new

    return directed_edges, dist_matrix


def greedy_search(graph, dist_matrix, s, query_idx, k, L):
    """
    Implementation of Algorithm 1: GreedySearch from Vamana paper
    """
    # Initialize result set and visited set
    result_set = set([s])
    visited = set()

    while result_set - visited:
        # Find closest unvisited point in result_set to query
        p_star = min(result_set - visited, key=lambda p: dist_matrix[p, query_idx])

        # Update sets
        visited.add(p_star)
        result_set.update(graph[p_star])

        # Retain only closest L points
        if len(result_set) > L:
            result_set = set(
                sorted(result_set, key=lambda p: dist_matrix[p, query_idx])[:L]
            )

    # Return closest k points and all visited nodes
    closest_k = sorted(result_set, key=lambda p: dist_matrix[p, query_idx])[:k]
    return closest_k, visited


def robust_prune(graph, dist_matrix, p, candidates, alpha, R):
    """
    Implementation of Algorithm 2: RobustPrune from Vamana paper
    """
    # Combine existing neighbors with candidates and remove p itself
    V = set(candidates).union(set(graph[p])) - {p}

    # Reset out-neighbors of p
    new_neighbors = []

    # Main pruning loop
    while V and len(new_neighbors) < R:
        # Find closest point in V to p
        p_star = min(V, key=lambda p_prime: dist_matrix[p, p_prime])

        # Add to neighbors
        new_neighbors.append(p_star)
        V.remove(p_star)

        # Further pruning based on distance criterion
        V_copy = V.copy()
        for p_prime in V_copy:
            if alpha * dist_matrix[p_star, p_prime] <= dist_matrix[p, p_prime]:
                V.remove(p_prime)

    return new_neighbors


def compute_vamana(points, alpha=1.2, L=32, R=8):
    """
    Implementation of Algorithm 3: Vamana Indexing algorithm
    """
    n = len(points)
    dist_matrix = distance.cdist(points, points)

    # Initialize graph as R-regular random directed graph
    graph = [[] for _ in range(n)]
    for i in range(n):
        # Choose R random neighbors
        potential_neighbors = list(range(n))
        potential_neighbors.remove(i)
        if len(potential_neighbors) > R:
            neighbors = np.random.choice(potential_neighbors, R, replace=False)
            graph[i] = list(neighbors)
        else:
            graph[i] = potential_neighbors

    # Find medoid (closest to center of mass)
    center = np.mean(points, axis=0)
    s = min(range(n), key=lambda i: np.linalg.norm(points[i] - center))

    # Create random permutation
    sigma = np.random.permutation(n)

    # Build the graph incrementally
    for i in range(n):
        node = sigma[i]

        # Use greedy search to find candidates
        _, visited = greedy_search(graph, dist_matrix, s, node, 1, L)

        # Update out-neighbors of current node
        graph[node] = robust_prune(graph, dist_matrix, node, visited, alpha, R)

        # Update in-neighbors
        for j in graph[node]:
            if len(graph[j]) + 1 > R:
                # Too many neighbors, prune
                candidates = graph[j] + [node]
                graph[j] = robust_prune(graph, dist_matrix, j, candidates, alpha, R)
            else:
                # Add new in-neighbor
                if node not in graph[j]:
                    graph[j].append(node)

    # Convert to edge list for visualization
    edges = []
    for i in range(n):
        for j in graph[i]:
            edges.append((i, j))

    return edges, dist_matrix


def compute_hnsw(points, M=16, ef_construction=200, mL=None):
    """
    Implementation of Hierarchical Navigable Small World (HNSW) graph construction.

    Parameters:
    - points: array of data points
    - M: number of established connections per element per layer
    - ef_construction: size of the dynamic candidate list during construction
    - mL: normalization factor for level generation (defaults to 1/ln(M))

    Returns:
    - edges: list of directed edges in the graph
    - layer_info: dictionary mapping node index to its maximum layer
    """
    n = len(points)
    dist_matrix = distance.cdist(points, points)

    # Set default mL if not provided
    if mL is None:
        mL = 1.0 / np.log(M)

    # Maximum connections for ground layer (layer 0)
    Mmax0 = 2 * M
    # Maximum connections for other layers
    Mmax = M

    # Initialize graph: adjacency list for each layer
    # graph[layer][node] = list of neighbors
    graph = defaultdict(lambda: defaultdict(list))

    # Store max layer for each node
    max_layer = {}

    # Algorithm 1: INSERT function from the paper
    def insert_element(q_idx, element_level):
        # If this is the first element
        if not max_layer:
            max_layer[q_idx] = element_level
            return

        # Find entry point - start at max level
        ep = max(max_layer.items(), key=lambda x: x[1])[0]
        L = max_layer[ep]

        # Phase 1: Find neighbors on each level > element_level
        for lc in range(L, element_level, -1):
            # Search layer to find the closest element
            W = search_layer(q_idx, [ep], 1, lc)
            ep = W[0]

        # Phase 2: Insert element into each level <= element_level
        for lc in range(min(L, element_level), -1, -1):
            # Search layer to find ef_construction closest elements
            W = search_layer(q_idx, [ep], ef_construction, lc)

            # Select neighbors for the element using heuristic
            neighbors = select_neighbors_heuristic(q_idx, W, M, lc)

            # Add bidirectional connections
            for neighbor in neighbors:
                if neighbor not in graph[lc][q_idx]:
                    graph[lc][q_idx].append(neighbor)
                if q_idx not in graph[lc][neighbor]:
                    graph[lc][neighbor].append(q_idx)

                    # Check if we need to shrink connections
                    if lc == 0 and len(graph[lc][neighbor]) > Mmax0:
                        graph[lc][neighbor] = select_neighbors_heuristic(
                            neighbor, graph[lc][neighbor], Mmax0, lc
                        )
                    elif lc > 0 and len(graph[lc][neighbor]) > Mmax:
                        graph[lc][neighbor] = select_neighbors_heuristic(
                            neighbor, graph[lc][neighbor], Mmax, lc
                        )

            ep = W[0]  # Set enter point for the next layer

        # Update max layer
        max_layer[q_idx] = element_level

    # Algorithm 2: SEARCH-LAYER function from the paper
    def search_layer(q_idx, ep_idxs, ef, lc):
        visited = set(ep_idxs)
        candidates = [(dist_matrix[q_idx, e], e) for e in ep_idxs]
        heapq.heapify(candidates)

        # Use a min-heap for candidates and a max-heap for results
        results = [(-dist_matrix[q_idx, e], e) for e in ep_idxs]
        heapq.heapify(results)

        # Keep track of furthest distance in results
        if results:
            furthest_dist = -results[0][0]
        else:
            furthest_dist = float("inf")

        # Main search loop
        while candidates:
            _, current = heapq.heappop(candidates)

            # If we've checked all potentially closer elements
            if dist_matrix[q_idx, current] > furthest_dist and len(results) >= ef:
                break

            # Explore neighbors
            for neighbor in graph[lc][current]:
                if neighbor not in visited:
                    visited.add(neighbor)

                    dist_to_neighbor = dist_matrix[q_idx, neighbor]

                    # Add to candidates if potentially better
                    if dist_to_neighbor < furthest_dist or len(results) < ef:
                        heapq.heappush(candidates, (dist_to_neighbor, neighbor))
                        heapq.heappush(results, (-dist_to_neighbor, neighbor))

                        # Remove furthest if we have too many results
                        if len(results) > ef:
                            heapq.heappop(results)
                            furthest_dist = -results[0][0]

        # Return the closest ef elements
        return [item[1] for item in sorted(results, key=lambda x: -x[0])]

    # Algorithm 4: SELECT-NEIGHBORS-HEURISTIC from the paper
    def select_neighbors_heuristic(
        q_idx, candidates, M_target, lc, extend_candidates=False
    ):
        # Working queue of candidates
        W = list(candidates)
        result = []

        # Option to extend candidates by neighbors (rarely used)
        if extend_candidates:
            extended = set(W)
            for e in W:
                for neighbor in graph[lc][e]:
                    if neighbor not in extended:
                        extended.add(neighbor)
                        W.append(neighbor)

        # Sort candidates by distance to q
        W.sort(key=lambda x: dist_matrix[q_idx, x])

        # Select diverse neighbors
        while W and len(result) < M_target:
            candidate = W.pop(0)

            # Add candidate if it's closer to q than to any existing results
            should_add = True
            for existing in result:
                if dist_matrix[candidate, existing] < dist_matrix[q_idx, candidate]:
                    should_add = False
                    break

            if should_add:
                result.append(candidate)

        # If we don't have enough neighbors, add remaining closest ones
        if len(result) < M_target:
            W.sort(key=lambda x: dist_matrix[q_idx, x])
            while W and len(result) < M_target:
                result.append(W.pop(0))

        return result

    # Build the graph by inserting elements one by one
    for i in range(n):
        # Generate random level with exponential distribution
        element_level = int(-math.log(np.random.random()) * mL)
        insert_element(i, element_level)

    # Convert graph to edge list for visualization
    edges = []
    for layer in graph:
        for node in graph[layer]:
            for neighbor in graph[layer][node]:
                edges.append((node, neighbor))

    return edges, dist_matrix, max_layer


def plot_all_graphs(points, rng_edges, sng_edges, vamana_edges, hnsw_edges):
    """Plot the points with RNG, SNG, Vamana, and HNSW graphs side by side."""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

    # Plot RNG
    ax1.scatter(points[:, 0], points[:, 1], c="blue", s=30)
    for i, j in rng_edges:
        ax1.plot(
            [points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], "k-", alpha=0.6
        )
    for i, (x, y) in enumerate(points):
        ax1.text(x + 0.05, y + 0.05, str(i), fontsize=12)
    ax1.set_title("Relative Neighborhood Graph (RNG)")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Plot SNG
    ax2.scatter(points[:, 0], points[:, 1], c="blue", s=30)
    for i, j in sng_edges:
        # Draw directed edges with arrows
        ax2.arrow(
            points[i, 0],
            points[i, 1],
            points[j, 0] - points[i, 0],
            points[j, 1] - points[i, 1],
            head_width=0.15,
            head_length=0.25,
            fc="k",
            ec="k",
            alpha=0.6,
            length_includes_head=True,
        )
    for i, (x, y) in enumerate(points):
        ax2.text(x + 0.05, y + 0.05, str(i), fontsize=12)
    ax2.set_title("Sparse Neighborhood Graph (SNG)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # Plot Vamana
    ax3.scatter(points[:, 0], points[:, 1], c="blue", s=30)
    for i, j in vamana_edges:
        # Draw directed edges with arrows
        ax3.arrow(
            points[i, 0],
            points[i, 1],
            points[j, 0] - points[i, 0],
            points[j, 1] - points[i, 1],
            head_width=0.15,
            head_length=0.25,
            fc="k",
            ec="k",
            alpha=0.6,
            length_includes_head=True,
        )
    for i, (x, y) in enumerate(points):
        ax3.text(x + 0.05, y + 0.05, str(i), fontsize=12)
    ax3.set_title("Vamana Graph (DiskANN)")
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal")

    # Plot HNSW
    ax4.scatter(points[:, 0], points[:, 1], c="blue", s=30)
    for i, j in hnsw_edges:
        # Draw directed edges with arrows
        ax4.arrow(
            points[i, 0],
            points[i, 1],
            points[j, 0] - points[i, 0],
            points[j, 1] - points[i, 1],
            head_width=0.15,
            head_length=0.25,
            fc="k",
            ec="k",
            alpha=0.6,
            length_includes_head=True,
        )
    for i, (x, y) in enumerate(points):
        ax4.text(x + 0.05, y + 0.05, str(i), fontsize=12)
    ax4.set_title("Hierarchical NSW Graph (HNSW)")
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal")

    plt.tight_layout()
    plt.show()


def generate_separated_points(n_points, min_distance=1.5, max_range=15):
    points = []
    while len(points) < n_points:
        candidate = np.random.rand(2) * max_range  # Generate a new random point
        if all(np.linalg.norm(candidate - np.array(p)) > min_distance for p in points):
            points.append(
                candidate
            )  # Add point if it's far enough from existing points

    return np.array(points)


def main():
    # Generate random points
    np.random.seed(42)  # For reproducibility
    n_points = 30  # Using fewer points for clarity of visualization
    min_distance = 2  # Minimum allowed distance between points
    max_range = 15
    
    
    # Generate points
    print ("Generate points")
    points = generate_separated_points(n_points, min_distance, max_range)

    # Compute RNG
    print ("Compute RNG")
    rng_edges, _ = compute_rng(points)

    # Compute SNG
    print ("Compute SNG")
    sng_edges, _ = compute_sng(points)

    # Compute Vamana
    print ("Compute Vamana")
    vamana_edges, _ = compute_vamana(points, alpha=1.2, L=10, R=4)

    # Compute HNSW
    print ("Compute HNSW")
    hnsw_edges, _, _ = compute_hnsw(points, M=4, ef_construction=10)

    # Plot the result
    print ("Plot the result")
    plot_all_graphs(points, rng_edges, sng_edges, vamana_edges, hnsw_edges)

    print(f"RNG has {len(rng_edges)} undirected edges")
    print(f"SNG has {len(sng_edges)} directed edges")
    print(f"Vamana has {len(vamana_edges)} directed edges")
    print(f"HNSW has {len(hnsw_edges)} directed edges")


if __name__ == "__main__":
    main()
