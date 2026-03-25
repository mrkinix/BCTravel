# BCTravel: Exact TSP Solver
**By Hédi Ben Chiboub**

BCTravel is a high-performance, exact Branch-and-Bound solver for the Symmetric Traveling Salesman Problem (TSP). It operates by combining a Held-Karp 1-Tree Lagrangian relaxation for sharp lower bounds with custom topological pruning driven by the **BCcarver** Hamiltonian cycle engine.

Built in Rust, the solver is aggressively optimized for zero-heap inner loops, cache-friendly data structures, and highly parallelized tree exploration.

---

## ⚙️ Architecture & Engine

The solver achieves exact optimality through a synthesis of heuristics, graph theory, and parallel computing:

* **Held-Karp 1-Tree Lower Bound**: Utilizes a fixed-iteration Lagrangian relaxation. Computes Minimum Spanning Trees (MST) with penalty adjustments to enforce degree constraints, providing sharp lower bounds to prune the DFS tree early.
* **BCcarver Topological Pruning**: Before branching begins, the graph is structurally preprocessed using the Ben-Chiboub Carver logic. Edges that cannot belong to an optimal tour are permanently pruned from the domain, massively reducing the state space.
* **Heuristic Upper Bounding**: Establishes a strict initial ceiling using a multi-start Nearest-Neighbor heuristic followed by aggressive 2-Opt local search optimizations.
* **Statistical Branch Ordering**: Prioritizes edge traversal using a `μ + 2σ` statistical threshold. Costly/unlikely edges are evaluated last (for ordering only, not unsafe pruning).
* **Adaptive Bitmasking**: State tracking utilizes a highly optimized `u128` fast-path for instances where `N <= 128`. Automatically falls back to a dynamic `Vec<u64>` mask for larger graphs without sacrificing exactness.
* **Rayon Parallel Splitting**: Deploys independent subtasks across a thread pool by enumerating early branch decisions to a configurable depth (`SPLIT_DEPTH`), maximizing multi-core throughput.

---

## 🚀 Usage & Protocol

Run the solver directly via Cargo. The CLI supports built-in test suites, custom TSPLIB parsing, and random instance generation.

### Built-in Test Suite & Benchmarks
Run the default suite (verifies small known instances and benchmarks random Euclidean graphs):
```bash
cargo run --release
```

Benchmark a specific range of N nodes:
```bash
cargo run --release -- --bench 10 50
```

### Solve External TSPLIB Files
Parse and solve standard TSPLIB .tsp files (supports EUC_2D, GEO, and explicit weight matrices):
```bash
cargo run --release -- --tsp data/berlin52.tsp
```

### Random Instance Generation
Generate and solve a random Euclidean or symmetric non-metric instance with a specific seed:
```bash
# random N [seed]
cargo run --release -- --random 40 42

# symmetric N [max_w] [seed]
cargo run --release -- --symmetric 30 100 42
```

### Thread Override
Override the default core count (defaults to system num_cpus):
```bash
cargo run --release -- --threads 4 --bench 10 30
```

---

## 📊 Performance Matrix
Practical limits depend heavily on the graph structure. The solver guarantees exact optimality for all N, but compute time scales exponentially based on the tightness of the upper bounds and edge variance.

* **Worst-Case Topology**: Exact for N≤60.
* **Random Metric / Euclidean TSP**: Exact up to N≈100–200.

### Reference Benchmarks (8-Core CPU)
| N | NN + 2-Opt UB | B&B Optimal | Time (s) | Status |
|---|---|---|---|---|
| 20 | 2384 | 2384 | 0.0649 | ✅ |
| 30 | 2480 | 2473 | 0.3948 | ✅ |
| 40 | 2587 | 2559 | 2.4217 | ✅ |
| 50 | 2922 | 2882 | 11.9314 | ✅ |
| 75 | 3347 | 3317 | 107.3725 | ✅ |