# Graph-Based Nearest Neighbor Structures

This repository contains Python implementations of various **graph-based nearest neighbor structures**, including:

- **Relative Neighborhood Graph (RNG)**
- **Sparse Neighborhood Graph (SNG)**
- **Vamana Graph (DiskANN)**
- **Hierarchical Navigable Small World Graph (HNSW)**

These structures are useful for **approximate nearest neighbor (ANN) search**, a fundamental problem in high-dimensional data retrieval.

## 🚀 Features

- **Computes and visualizes** key graph-based structures used in ANN search.
- **Implements core algorithms** from research papers, including:
  - **Vamana (DiskANN)**
  - **HNSW (Hierarchical NSW)**
- **Efficiently generates points** while ensuring a minimum separation distance.
- **Plots all graphs side-by-side** for easy comparison.

## 📌 Dependencies

Ensure you have the following Python packages installed:

```bash
pip install numpy matplotlib scipy
```

## 🛠️ Usage

Run the script to generate random points, compute graphs, and visualize them:

```bash
python main.py
```

### Example Output

The script will generate **30 random points** and compute the following graphs:

- **RNG (Relative Neighborhood Graph)**
- **SNG (Sparse Neighborhood Graph)**
- **Vamana (DiskANN)**
- **HNSW (Hierarchical NSW)**

It will then plot them side by side for comparison.

## 📊 Visualization

The script generates a **4-panel visualization** of the different graphs:

[Graph Visualization](example_graph.png) *(Replace with actual output image)*

Each panel represents a different graph structure, showing how edges are formed between points.

## 📜 Algorithms Implemented

### 1️⃣ **Relative Neighborhood Graph (RNG)**

- Computes edges based on the "lune" condition.
- Ensures that no other point is closer to both endpoints of an edge.

### 2️⃣ **Sparse Neighborhood Graph (SNG)**

- Iteratively connects each point to its closest yet-unconnected neighbor.
- Removes points that are farther from the source than from the closest neighbor.

### 3️⃣ **Vamana Graph (DiskANN)**

- Implements **Vamana indexing algorithm** for fast ANN search.
- Prunes connections using **greedy search** and **robust pruning**.
- Constructs an **R-regular directed graph**.

### 4️⃣ **Hierarchical NSW (HNSW)**

- Implements **HNSW graph construction** for fast nearest neighbor retrieval.
- Uses **multi-layered indexing** and **heuristic-based neighbor selection**.
- Implements **greedy search** and **hierarchical pruning**.

## 📌 Example Code Execution

```python
# Generate random points
points = generate_separated_points(n_points=30, min_distance=2.0, max_range=15)

# Compute graphs
rng_edges, _ = compute_rng(points)
sng_edges, _ = compute_sng(points)
vamana_edges, _ = compute_vamana(points, alpha=1.2, L=10, R=4)
hnsw_edges, _, _ = compute_hnsw(points, M=4, ef_construction=10)

# Plot results
plot_all_graphs(points, rng_edges, sng_edges, vamana_edges, hnsw_edges)
```

## ✨ Results

At the end of execution, the script prints:

```text
RNG has X undirected edges
SNG has Y directed edges
Vamana has Z directed edges
HNSW has W directed edges
```

*(Exact values depend on random points generated.)*

## 📄 References

- **Vamana (DiskANN)**: [Subramanya et al., 2019](https://arxiv.org/pdf/1907.09991.pdf)
- **HNSW**: [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)

## 🏗️ Future Improvements

- Add **interactive visualization** with `networkx` or `pyvis`.
- Implement **query-based nearest neighbor retrieval**.
- Extend support to **higher-dimensional spaces**.

## 📜 License

This project is licensed under the **MIT License**.

---

🔥 **Star this repository** if you find it useful! 🚀
