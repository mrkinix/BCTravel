# BCTravel: Exact TSP Solver
**By Hédi Ben Chiboub**

Exact TSP up to ~75 nodes in ~90s using Held-Karp bounds and custom pruning (BCcarver).

BCTravel is a high-performance exact solver for the Symmetric Traveling Salesman Problem (TSP), built on Branch-and-Bound with strong lower bounds and aggressive structural pruning.

It combines Held-Karp 1-tree Lagrangian relaxation with a custom pruning engine (**BCcarver**) and an iterative optimality loop that can prove optimality *without full search*.

Implemented in Rust with focus on low-level efficiency: zero-heap inner loops, cache-friendly data structures, and parallel tree exploration.

---

## ⚙️ Architecture & Engine

BCTravel achieves exact optimality through a layered approach:

- **Held-Karp 1-Tree Lower Bound**  
  Fixed-iteration Lagrangian relaxation using MST with penalties to enforce degree constraints. Provides tight lower bounds for pruning.

- **BCcarver Topological Pruning**  
  Preprocessing phase that removes edges incompatible with any optimal Hamiltonian cycle, reducing branching factor before search.

- **Iterative Optimality Loop (NEW)**  
  Before full Branch-and-Bound, the solver attempts to prove optimality early:
  - Prunes edges where `LB ≥ UB`
  - Removes edges of the current best tour
  - Runs BCcarver to search for alternative Hamiltonian cycles  
    - **UNSAT** → no alternative tour exists → current solution is globally optimal  
    - **SAT** → new tour found → improved via local search → UB tightened  
    - **TIMEOUT** → fallback to B&B with improved UB  
  - Repeats for a fixed number of iterations

  This loop can eliminate the need for full search on structured instances.

- **Heuristic Upper Bound (UB)**  
  Multi-start Nearest Neighbor followed by 2-Opt refinement.

- **Branch Ordering Heuristics**  
  Uses a `μ + 2σ` cost-based ordering to prioritize promising edges.

- **Adaptive Bitmasking**  
  Fast `u128` path for `N ≤ 128`, fallback to dynamic bitsets for larger graphs.

- **Parallel Branching (Rayon)**  
  Early tree splitting across threads for efficient multi-core scaling.

---

## 🚀 Usage

### Default benchmarks
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
| N | NN + 2-Opt UB | B&B Optimal | Time (s) | Gap | Status |
|---|---|---|---|---|---|
| 20 | 2384 | 2384 | 0.0607 | 0.0% | ✅ |
| 30 | 2473 | 2473 | 0.3014 | 0.0% | ✅ |
| 40 | 2587 | 2559 | 1.8621 | 1.1% | ✅ |
| 41 | 2462 | 2462 | 2.0212 | 0.0% | ✅ |
| 50 | 2916 | 2882 | 9.7106 | 1.2% | ✅ |
| 75 | 3317 | 3317 | 94.1843 | 0.0% | ✅ |