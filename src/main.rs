// bctravel.rs — Exact TSP Solver (Ben-Chiboub Travel)

//

// Architecture: Branch-and-Bound with:

//   • Held-Karp 1-Tree lower bound (fixed-iteration Lagrangian relaxation)

//   • Nearest-Neighbor + 2-Opt upper bound (initial ceiling)

//   • Statistical branch ordering (μ + 2σ prioritization — for ordering ONLY, not pruning)

//   • Cache-friendly flat 1D adjacency matrix (u32 weights)

//   • Pre-allocated zero-heap workspace for all inner-loop operations

//   • Rayon parallel tree splitting at configurable depth

//   • Dynamic bitmask: u128 for N≤128, Vec<u64> for N>128

//

// USAGE:

//   cargo run --release                    → built-in test suite

//   cargo run --release -- --tsp file.tsp  → solve a TSPLIB .tsp file

//   cargo run --release -- --random N      → solve a random N-city instance

//   cargo run --release -- --bench N0 N1   → benchmark range [N0..N1]

//

// NOTE on N > 128:

//   For N > 128 the visited-city bitmask is stored as Vec<u64> (dynamic bitmask).

//   This is slower than u128 but correct. The solver remains exact for all N,

//   but practical limits depend on graph structure (typically exact for N ≤ 60 in

//   worst case; random metric TSP up to N ≈ 100–200 with good upper bounds).


use rayon::prelude::*;

use std::env;

use std::fs;

use std::path::Path;

use std::sync::atomic::{AtomicU64, Ordering};

use std::sync::{Arc, Mutex};

use std::time::Instant;


mod bccarver;


// ───────────────────────────── CONFIG ─────────────────────────────


/// Parallel tree-split depth. 4 → up to 16 subtrees.

const SPLIT_DEPTH: usize = 4;


/// Maximum cities for u128 fast path. Above this → Vec<u64> mask.

const U128_LIMIT: usize = 128;


/// Fixed-iteration Held-Karp tuning for the 1-tree lower bound.

const HELD_KARP_MAX_ITERS: usize = 12;

const HELD_KARP_STALL_ITERS: usize = 3;

const HELD_KARP_MIN_LAMBDA: f64 = 0.25;

const HELD_KARP_INF: i64 = i64::MAX / 8;


// ───────────────────────────── DISTANCE MATRIX ─────────────────────────────


/// Flat row-major distance matrix. dist[i*n + j] = cost(i,j).

/// Symmetric: dist[i*n+j] == dist[j*n+i].

#[derive(Clone)]

struct DistMatrix {

    n: usize,

    data: Vec<u32>,

}


impl DistMatrix {

    fn new(n: usize, data: Vec<u32>) -> Self {

        assert_eq!(data.len(), n * n);

        Self { n, data }

    }


    #[inline(always)]

    fn get(&self, i: usize, j: usize) -> u32 {

        // SAFETY: caller guarantees i,j < n

        unsafe { *self.data.get_unchecked(i * self.n + j) }

    }


    #[inline(always)]

    fn is_edge_allowed(&self, i: usize, j: usize) -> bool {

        i != j && self.get(i, j) != u32::MAX

    }


    /// Global mean and std-dev of all off-diagonal edge weights.

    fn mean_stddev(&self) -> (f64, f64) {

        let n = self.n;

        let mut sum = 0u64;

        let mut count = 0u64;

        for i in 0..n {

            for j in 0..n {

                if self.is_edge_allowed(i, j) {

                    sum += self.get(i, j) as u64;

                    count += 1;

                }

            }

        }

        if count == 0 {

            return (0.0, 0.0);

        }

        let mean = sum as f64 / count as f64;

        let mut var = 0f64;

        for i in 0..n {

            for j in 0..n {

                if self.is_edge_allowed(i, j) {

                    let d = self.get(i, j) as f64 - mean;

                    var += d * d;

                }

            }

        }

        let stddev = (var / count as f64).sqrt();

        (mean, stddev)

    }

}


// ───────────────────────────── HELD-KARP 1-TREE WORKSPACE ─────────────────────────────


#[derive(Clone, Copy)]

enum HeldKarpMode {

    Cycle { root: usize },

    Path { start: usize, current: usize },

}


/// Reusable scratch space for the Held-Karp 1-tree lower bound.

struct HeldKarpWorkspace {

    penalties: Vec<i64>,

    key: Vec<i64>,

    parent: Vec<usize>,

    in_tree: Vec<bool>,

    degree: Vec<i32>,

    active_nodes: Vec<usize>,

}


impl HeldKarpWorkspace {

    fn new(n: usize) -> Self {

        Self {

            penalties: vec![0; n],

            key: vec![HELD_KARP_INF; n],

            parent: vec![usize::MAX; n],

            in_tree: vec![false; n],

            degree: vec![0; n],

            active_nodes: Vec::with_capacity(n),

        }

    }


    fn prepare_cycle_u128(&mut self, root: usize, unvisited: u128, n: usize) {

        self.active_nodes.clear();

        self.active_nodes.push(root);

        let mut mask = unvisited;

        while mask != 0 {

            let v = mask.trailing_zeros() as usize;

            mask &= mask - 1;

            if v < n {

                self.active_nodes.push(v);

            }

        }

    }


    fn prepare_path_u128(&mut self, start: usize, current: usize, unvisited: u128, n: usize) {

        self.active_nodes.clear();

        self.active_nodes.push(start);

        if current != start {

            self.active_nodes.push(current);

        }

        let mut mask = unvisited;

        while mask != 0 {

            let v = mask.trailing_zeros() as usize;

            mask &= mask - 1;

            if v < n {

                self.active_nodes.push(v);

            }

        }

    }


    fn prepare_cycle_dyn(&mut self, root: usize, unvisited: &[u64], n: usize) {

        self.active_nodes.clear();

        self.active_nodes.push(root);

        for (wi, &word) in unvisited.iter().enumerate() {

            let mut mask = word;

            while mask != 0 {

                let tz = mask.trailing_zeros() as usize;

                let v = wi * 64 + tz;

                mask &= mask - 1;

                if v < n {

                    self.active_nodes.push(v);

                }

            }

        }

    }


    fn prepare_path_dyn(&mut self, start: usize, current: usize, unvisited: &[u64], n: usize) {

        self.active_nodes.clear();

        self.active_nodes.push(start);

        if current != start {

            self.active_nodes.push(current);

        }

        for (wi, &word) in unvisited.iter().enumerate() {

            let mut mask = word;

            while mask != 0 {

                let tz = mask.trailing_zeros() as usize;

                let v = wi * 64 + tz;

                mask &= mask - 1;

                if v < n {

                    self.active_nodes.push(v);

                }

            }

        }

    }


    fn adjusted_cost(&self, dist: &DistMatrix, u: usize, v: usize) -> i64 {

        let base = dist.get(u, v);

        if base == u32::MAX {

            return HELD_KARP_INF;

        }

        let mut adjusted = base as i64;

        adjusted = adjusted.saturating_add(self.penalties[u]);

        adjusted = adjusted.saturating_add(self.penalties[v]);

        adjusted.clamp(-HELD_KARP_INF, HELD_KARP_INF)

    }


    fn reset_iteration_state(&mut self) {

        for idx in 0..self.active_nodes.len() {

            let node = self.active_nodes[idx];

            self.key[node] = HELD_KARP_INF;

            self.parent[node] = usize::MAX;

            self.in_tree[node] = false;

            self.degree[node] = 0;

        }

    }


    fn mst_adjusted(&mut self, dist: &DistMatrix, skip: Option<usize>) -> Option<i64> {

        let target = self

            .active_nodes

            .iter()

            .filter(|&&node| Some(node) != skip)

            .count();

        if target <= 1 {

            return Some(0);

        }


        let start = self

            .active_nodes

            .iter()

            .copied()

            .find(|&node| Some(node) != skip)?;

        self.key[start] = 0;


        let mut total = 0i64;

        let mut added = 0usize;


        for _ in 0..target {

            let mut best_node = usize::MAX;

            let mut best_key = HELD_KARP_INF;

            for idx in 0..self.active_nodes.len() {

                let node = self.active_nodes[idx];

                if Some(node) == skip || self.in_tree[node] || self.key[node] >= best_key {

                    continue;

                }

                best_key = self.key[node];

                best_node = node;

            }


            if best_node == usize::MAX || best_key == HELD_KARP_INF {

                return None;

            }


            self.in_tree[best_node] = true;

            total = total.saturating_add(best_key);

            added += 1;


            if self.parent[best_node] != usize::MAX {

                let parent = self.parent[best_node];

                self.degree[best_node] += 1;

                self.degree[parent] += 1;

            }


            for idx in 0..self.active_nodes.len() {

                let other = self.active_nodes[idx];

                if Some(other) == skip || self.in_tree[other] {

                    continue;

                }

                let cost = self.adjusted_cost(dist, best_node, other);

                if cost < self.key[other] {

                    self.key[other] = cost;

                    self.parent[other] = best_node;

                }

            }

        }


        if added == target {

            Some(total)

        } else {

            None

        }

    }


    fn one_tree_adjusted_cost(&mut self, dist: &DistMatrix, mode: HeldKarpMode) -> Option<i64> {

        self.reset_iteration_state();


        match mode {

            HeldKarpMode::Cycle { root } => {

                let mst_cost = self.mst_adjusted(dist, Some(root))?;

                let mut best_1 = (HELD_KARP_INF, usize::MAX);

                let mut best_2 = (HELD_KARP_INF, usize::MAX);


                for idx in 0..self.active_nodes.len() {

                    let node = self.active_nodes[idx];

                    if node == root {

                        continue;

                    }

                    let cost = self.adjusted_cost(dist, root, node);

                    if cost < best_1.0 {

                        best_2 = best_1;

                        best_1 = (cost, node);

                    } else if cost < best_2.0 {

                        best_2 = (cost, node);

                    }

                }


                if best_1.1 == usize::MAX

                    || best_2.1 == usize::MAX

                    || best_1.0 == HELD_KARP_INF

                    || best_2.0 == HELD_KARP_INF

                {

                    return None;

                }


                self.degree[root] += 2;

                self.degree[best_1.1] += 1;

                self.degree[best_2.1] += 1;


                Some(mst_cost.saturating_add(best_1.0).saturating_add(best_2.0))

            }

            HeldKarpMode::Path { start, current } => {

                let mst_cost = self.mst_adjusted(dist, None)?;

                self.degree[start] += 1;

                self.degree[current] += 1;

                Some(

                    mst_cost

                        .saturating_add(self.penalties[start])

                        .saturating_add(self.penalties[current]),

                )

            }

        }

    }


    fn held_karp_bound(&mut self, dist: &DistMatrix, mode: HeldKarpMode, remaining_ub: u64) -> u64 {

        if self.active_nodes.len() <= 1 {

            return 0;

        }


        for idx in 0..self.active_nodes.len() {

            let node = self.active_nodes[idx];

            self.penalties[node] = 0;

        }


        let mut best_bound = i64::MIN;

        let mut lambda = 2.0;

        let mut stall_iters = 0usize;


        for _ in 0..HELD_KARP_MAX_ITERS {

            let adjusted_tree = match self.one_tree_adjusted_cost(dist, mode) {

                Some(cost) => cost,

                None => return u64::MAX,

            };


            let mut penalty_sum = 0i64;

            let mut norm_sq = 0i64;

            for idx in 0..self.active_nodes.len() {

                let node = self.active_nodes[idx];

                penalty_sum = penalty_sum.saturating_add(self.penalties[node]);

                let diff = (self.degree[node] - 2) as i64;

                norm_sq = norm_sq.saturating_add(diff.saturating_mul(diff));

            }


            let bound = adjusted_tree.saturating_sub(2 * penalty_sum);

            if bound > best_bound {

                best_bound = bound;

                stall_iters = 0;

            } else {

                stall_iters += 1;

                if stall_iters >= HELD_KARP_STALL_ITERS {

                    lambda *= 0.5;

                    stall_iters = 0;

                    if lambda < HELD_KARP_MIN_LAMBDA {

                        break;

                    }

                }

            }


            if best_bound >= remaining_ub as i64 || norm_sq == 0 {

                break;

            }


            let gap = remaining_ub.saturating_sub(best_bound.max(0) as u64) as f64;

            if gap <= 0.0 {

                break;

            }


            let step = ((lambda * gap / norm_sq as f64).round() as i64).max(1);

            for idx in 0..self.active_nodes.len() {

                let node = self.active_nodes[idx];

                let diff = (self.degree[node] - 2) as i64;

                self.penalties[node] =

                    self.penalties[node].saturating_add(step.saturating_mul(diff));

            }

        }


        best_bound.max(0) as u64

    }

}


// ───────────────────────────── HEURISTIC UPPER BOUND ─────────────────────────────


/// Nearest-Neighbor heuristic: O(N²). Returns (tour_cost, tour_order).

fn nearest_neighbor(dist: &DistMatrix, start: usize) -> (u64, Vec<usize>) {

    let n = dist.n;

    let mut visited = vec![false; n];

    let mut tour = Vec::with_capacity(n);

    let mut current = start;

    visited[current] = true;

    tour.push(current);


    for _ in 1..n {

        let mut best_cost = u32::MAX;

        let mut best_next = usize::MAX;

        for v in 0..n {

            if !visited[v] {

                let c = dist.get(current, v);

                if c != u32::MAX && c < best_cost {

                    best_cost = c;

                    best_next = v;

                }

            }

        }

        if best_next == usize::MAX {

            return (u64::MAX, Vec::new());

        }

        visited[best_next] = true;

        tour.push(best_next);

        current = best_next;

    }


    let cost = tour_cost(dist, &tour);

    (cost, tour)

}


/// Tour cost (sum of edges including wrap-around).

fn tour_cost(dist: &DistMatrix, tour: &[usize]) -> u64 {

    let n = tour.len();

    if n == 0 {

        return u64::MAX;

    }

    let mut total = 0u64;

    for i in 0..n {

        let edge = dist.get(tour[i], tour[(i + 1) % n]);

        if edge == u32::MAX {

            return u64::MAX;

        }

        total += edge as u64;

    }

    total

}


/// 2-Opt improvement. Modifies tour in-place; returns new cost.

fn two_opt(dist: &DistMatrix, tour: &mut Vec<usize>) -> u64 {

    let n = tour.len();

    let mut improved = true;

    let mut cost = tour_cost(dist, tour);


    while improved {

        improved = false;

        for i in 0..n - 1 {

            for j in i + 2..n {

                if j == n - 1 && i == 0 {

                    continue;

                } // skip wrap-around self-edge

                let a = tour[i];

                let b = tour[i + 1];

                let c = tour[j];

                let d = tour[(j + 1) % n];


                let old_cost = dist.get(a, b) as i64 + dist.get(c, d) as i64;

                let new_cost = dist.get(a, c) as i64 + dist.get(b, d) as i64;


                if new_cost < old_cost {

                    // Reverse segment [i+1..j]

                    tour[i + 1..=j].reverse();

                    cost = (cost as i64 - (old_cost - new_cost)) as u64;

                    improved = true;

                }

            }

        }

    }

    cost

}


/// Build best initial upper bound: run NN from multiple starts + 2-Opt.

fn build_upper_bound(dist: &DistMatrix) -> (u64, Vec<usize>) {

    let n = dist.n;

    let starts: Vec<usize> = if n <= 20 {

        (0..n).collect()

    } else {

        // Sample a few starts

        let step = (n / 10).max(1);

        (0..n).step_by(step).collect()

    };


    let mut best_cost = u64::MAX;

    let mut best_tour = vec![];


    for s in starts {

        let (_, mut tour) = nearest_neighbor(dist, s);

        if tour.len() != n {

            continue;

        }

        let c = two_opt(dist, &mut tour);

        if c < best_cost {

            best_cost = c;

            best_tour = tour;

        }

    }


    (best_cost, best_tour)

}


// ───────────────────────────── SOLVER STATE ─────────────────────────────


/// Shared upper bound across parallel workers.

struct SharedUB {

    cost: AtomicU64,

    tour: Mutex<Vec<usize>>,

}


impl SharedUB {

    fn new(cost: u64, tour: Vec<usize>) -> Self {

        Self {

            cost: AtomicU64::new(cost),

            tour: Mutex::new(tour),

        }

    }


    #[inline]

    fn get(&self) -> u64 {

        self.cost.load(Ordering::Relaxed)

    }


    fn try_update(&self, new_cost: u64, new_tour: &[usize]) -> bool {

        let old = self.cost.load(Ordering::Relaxed);

        if new_cost < old {

            // CAS loop

            if self

                .cost

                .compare_exchange(old, new_cost, Ordering::SeqCst, Ordering::Relaxed)

                .is_ok()

            {

                let mut guard = self.tour.lock().unwrap();

                *guard = new_tour.to_vec();

                return true;

            }

        }

        false

    }

}


// ───────────────────────────── BRANCH-AND-BOUND (u128 fast path) ─────────────────────────────


/// DFS B&B state for u128 bitmask path (N ≤ 128).

struct SolverState128<'a> {

    dist: &'a DistMatrix,

    n: usize,

    full_mask: u128,

    global_ub: &'a SharedUB,

    hk_ws: HeldKarpWorkspace,

    stat_threshold: u64, // μ + 2σ per-edge threshold (for branch ordering)

}


impl<'a> SolverState128<'a> {

    fn new(dist: &'a DistMatrix, global_ub: &'a SharedUB, mu: f64, sigma: f64) -> Self {

        let n = dist.n;

        let full_mask = if n == 128 {

            u128::MAX

        } else {

            (1u128 << n) - 1

        };

        let threshold = ((mu + 2.0 * sigma) as u64).saturating_add(1);

        Self {

            dist,

            n,

            full_mask,

            global_ub,

            hk_ws: HeldKarpWorkspace::new(n),

            stat_threshold: threshold,

        }

    }


    /// Held-Karp 1-tree lower bound on the remaining completion problem.

    ///

    /// At the root this is the standard cycle 1-tree.

    /// Once a path prefix exists, we switch to the path variant by adding

    /// an implicit zero-cost dummy root connected to `start` and `current`.

    #[inline]

    fn lower_bound(

        &mut self,

        current_cost: u64,

        current: usize,

        start: usize,

        unvisited: u128,

        unvisited_count: u32,

    ) -> u64 {

        if unvisited_count == 0 {

            // Only the return edge left

            let back = self.dist.get(current, start);

            return if back == u32::MAX {

                u64::MAX

            } else {

                current_cost + back as u64

            };

        }


        let remaining_ub = self.global_ub.get().saturating_sub(current_cost);

        let is_root_state = current == start && unvisited_count as usize + 1 == self.n;


        let remainder = if is_root_state {

            self.hk_ws.prepare_cycle_u128(start, unvisited, self.n);

            self.hk_ws

                .held_karp_bound(self.dist, HeldKarpMode::Cycle { root: start }, remaining_ub)

        } else {

            self.hk_ws

                .prepare_path_u128(start, current, unvisited, self.n);

            self.hk_ws.held_karp_bound(

                self.dist,

                HeldKarpMode::Path { start, current },

                remaining_ub,

            )

        };


        if remainder == u64::MAX {

            return u64::MAX;

        }


        current_cost.saturating_add(remainder)

    }


    /// Core DFS.

    fn dfs(

        &mut self,

        current: usize,

        start: usize,

        visited: u128,

        current_cost: u64,

        depth: usize,

        path: &mut Vec<usize>,

    ) {

        let ub = self.global_ub.get();


        // Already worse than best known?

        if current_cost >= ub {

            return;

        }


        let unvisited = self.full_mask & !visited;

        let unvisited_count = unvisited.count_ones();


        // Leaf: all cities visited, close the tour

        if unvisited_count == 0 {

            let back = self.dist.get(current, start);

            if back == u32::MAX {

                return;

            }

            let final_cost = current_cost + back as u64;

            if final_cost < ub {

                self.global_ub.try_update(final_cost, path);

            }

            return;

        }


        // Lower bound pruning

        let lb = self.lower_bound(current_cost, current, start, unvisited, unvisited_count);

        if lb >= ub {

            return;

        }


        // Collect candidates: all unvisited cities

        // Capacity 128 on stack via smallvec-style array

        let mut candidates: [(u32, usize); 128] = [(0, 0); 128]; // (cost, city)

        let mut n_cands = 0usize;

        {

            let mut m = unvisited;

            while m != 0 {

                let v = m.trailing_zeros() as usize;

                m &= m - 1;

                if v < self.n {

                    let edge = self.dist.get(current, v);

                    if edge != u32::MAX {

                        candidates[n_cands] = (edge, v);

                        n_cands += 1;

                    }

                }

            }

        }


        if n_cands == 0 {

            return;

        }


        // Sort: cheapest first; within same cost, prefer edges ≤ threshold

        candidates[..n_cands].sort_unstable_by(|a, b| {

            let a_hot = (a.0 as u64) <= self.stat_threshold;

            let b_hot = (b.0 as u64) <= self.stat_threshold;

            a.0.cmp(&b.0).then(b_hot.cmp(&a_hot)) // cheaper first, "hot" preferred

        });


        // Branch

        for i in 0..n_cands {

            let (edge_cost, next) = candidates[i];

            let new_cost = current_cost + edge_cost as u64;

            if new_cost >= self.global_ub.get() {

                continue;

            }


            let new_visited = visited | (1u128 << next);

            path.push(next);

            self.dfs(next, start, new_visited, new_cost, depth + 1, path);

            path.pop();

        }

    }

}


// ───────────────────────────── BRANCH-AND-BOUND (dynamic bitmask, N > 128) ─────────────────────────────


struct SolverStateDyn<'a> {

    dist: &'a DistMatrix,

    n: usize,

    words: usize,

    global_ub: &'a SharedUB,

    hk_ws: HeldKarpWorkspace,

    stat_threshold: u64,

}


impl<'a> SolverStateDyn<'a> {

    fn new(dist: &'a DistMatrix, global_ub: &'a SharedUB, mu: f64, sigma: f64) -> Self {

        let n = dist.n;

        let words = (n + 63) / 64;

        let threshold = ((mu + 2.0 * sigma) as u64).saturating_add(1);

        Self {

            dist,

            n,

            words,

            global_ub,

            hk_ws: HeldKarpWorkspace::new(n),

            stat_threshold: threshold,

        }

    }


    fn lower_bound(

        &mut self,

        current_cost: u64,

        current: usize,

        start: usize,

        unvisited: &[u64],

        unvisited_count: usize,

    ) -> u64 {

        if unvisited_count == 0 {

            let back = self.dist.get(current, start);

            return if back == u32::MAX {

                u64::MAX

            } else {

                current_cost + back as u64

            };

        }


        let remaining_ub = self.global_ub.get().saturating_sub(current_cost);

        let is_root_state = current == start && unvisited_count + 1 == self.n;


        let remainder = if is_root_state {

            self.hk_ws.prepare_cycle_dyn(start, unvisited, self.n);

            self.hk_ws

                .held_karp_bound(self.dist, HeldKarpMode::Cycle { root: start }, remaining_ub)

        } else {

            self.hk_ws

                .prepare_path_dyn(start, current, unvisited, self.n);

            self.hk_ws.held_karp_bound(

                self.dist,

                HeldKarpMode::Path { start, current },

                remaining_ub,

            )

        };


        if remainder == u64::MAX {

            return u64::MAX;

        }


        current_cost.saturating_add(remainder)

    }


    fn dfs(

        &mut self,

        current: usize,

        start: usize,

        visited: &mut Vec<u64>,

        current_cost: u64,

        path: &mut Vec<usize>,

    ) {

        let ub = self.global_ub.get();

        if current_cost >= ub {

            return;

        }


        // Build unvisited mask

        let n = self.n;

        let words = self.words;

        let mut unvisited = vec![0u64; words];

        let mut unvisited_count = 0usize;

        for wi in 0..words {

            let base = wi * 64;

            let bits_in_word = (n.saturating_sub(base)).min(64);

            if bits_in_word == 0 {

                break;

            }

            let domain = if bits_in_word == 64 {

                u64::MAX

            } else {

                (1u64 << bits_in_word) - 1

            };

            unvisited[wi] = (!visited[wi]) & domain;

            unvisited_count += unvisited[wi].count_ones() as usize;

        }


        if unvisited_count == 0 {

            let back = self.dist.get(current, start);

            if back == u32::MAX {

                return;

            }

            let final_cost = current_cost + back as u64;

            if final_cost < ub {

                self.global_ub.try_update(final_cost, path);

            }

            return;

        }


        let lb = self.lower_bound(current_cost, current, start, &unvisited, unvisited_count);

        if lb >= ub {

            return;

        }


        // Collect candidates

        let mut candidates: Vec<(u32, usize)> = Vec::with_capacity(unvisited_count);

        for wi in 0..words {

            let mut m = unvisited[wi];

            while m != 0 {

                let tz = m.trailing_zeros() as usize;

                let v = wi * 64 + tz;

                m &= m - 1;

                if v < n {

                    let edge = self.dist.get(current, v);

                    if edge != u32::MAX {

                        candidates.push((edge, v));

                    }

                }

            }

        }


        if candidates.is_empty() {

            return;

        }


        candidates.sort_unstable_by(|a, b| {

            let a_hot = (a.0 as u64) <= self.stat_threshold;

            let b_hot = (b.0 as u64) <= self.stat_threshold;

            a.0.cmp(&b.0).then(b_hot.cmp(&a_hot))

        });


        for (edge_cost, next) in candidates {

            let new_cost = current_cost + edge_cost as u64;

            if new_cost >= self.global_ub.get() {

                continue;

            }


            let wi = next / 64;

            let bi = next % 64;

            visited[wi] |= 1u64 << bi;

            path.push(next);

            self.dfs(next, start, visited, new_cost, path);

            path.pop();

            visited[wi] &= !(1u64 << bi);

        }

    }

}


// ───────────────────────────── PARALLEL SPLITTING ─────────────────────────────


/// A subtask for parallel B&B: fixed prefix decisions, rest is free DFS.

#[derive(Clone)]

struct SubTask {

    /// Partial path already committed (cities in order, starting from city 0).

    path_prefix: Vec<usize>,

    /// Cost of that prefix.

    prefix_cost: u64,

    /// Visited mask (u128 fast path, else empty → use visited_dyn).

    visited_u128: u128,

    /// Visited mask (dynamic, for N > 128).

    visited_dyn: Vec<u64>,

}


/// Enumerate subtasks by branching `depth` levels from the root.

/// Returns a list of independent subtasks.

fn enumerate_subtasks(dist: &DistMatrix, start: usize, depth: usize) -> Vec<SubTask> {

    if dist.n <= 1 {

        return vec![];

    }

    let use_fast = dist.n <= U128_LIMIT;


    let mut queue: Vec<SubTask> = vec![SubTask {

        path_prefix: vec![start],

        prefix_cost: 0,

        visited_u128: if use_fast { 1u128 << start } else { 0 },

        visited_dyn: {

            if use_fast {

                vec![]

            } else {

                let words = (dist.n + 63) / 64;

                let mut v = vec![0u64; words];

                v[start / 64] |= 1u64 << (start % 64);

                v

            }

        },

    }];


    for _ in 0..depth {

        let mut next_queue = vec![];

        for task in queue {

            let current = *task.path_prefix.last().unwrap();

            // Generate children: all unvisited cities from current

            if use_fast {

                let full_mask: u128 = if dist.n == 128 {

                    u128::MAX

                } else {

                    (1u128 << dist.n) - 1

                };

                let unvisited = full_mask & !task.visited_u128;

                if unvisited == 0 {

                    next_queue.push(task);

                    continue;

                }

                let mut m = unvisited;

                while m != 0 {

                    let v = m.trailing_zeros() as usize;

                    m &= m - 1;

                    if v < dist.n {

                        let edge_cost = dist.get(current, v);

                        if edge_cost == u32::MAX {

                            continue;

                        }

                        let mut new_path = task.path_prefix.clone();

                        new_path.push(v);

                        next_queue.push(SubTask {

                            path_prefix: new_path,

                            prefix_cost: task.prefix_cost + edge_cost as u64,

                            visited_u128: task.visited_u128 | (1u128 << v),

                            visited_dyn: vec![],

                        });

                    }

                }

            } else {

                let n = dist.n;

                let words = (n + 63) / 64;

                let mut any = false;

                for wi in 0..words {

                    let base = wi * 64;

                    let bits_in_word = (n.saturating_sub(base)).min(64);

                    if bits_in_word == 0 {

                        break;

                    }

                    let domain = if bits_in_word == 64 {

                        u64::MAX

                    } else {

                        (1u64 << bits_in_word) - 1

                    };

                    let mut m = (!task.visited_dyn[wi]) & domain;

                    while m != 0 {

                        let tz = m.trailing_zeros() as usize;

                        let v = wi * 64 + tz;

                        m &= m - 1;

                        if v < n {

                            let edge_cost = dist.get(current, v);

                            if edge_cost == u32::MAX {

                                continue;

                            }

                            any = true;

                            let mut new_path = task.path_prefix.clone();

                            new_path.push(v);

                            let mut new_vis = task.visited_dyn.clone();

                            new_vis[wi] |= 1u64 << tz;

                            next_queue.push(SubTask {

                                path_prefix: new_path,

                                prefix_cost: task.prefix_cost + edge_cost as u64,

                                visited_u128: 0,

                                visited_dyn: new_vis,

                            });

                        }

                    }

                }

                if !any {

                    next_queue.push(task);

                }

            }

        }

        queue = next_queue;

        if queue.is_empty() {

            break;

        }

    }

    queue

}


fn masked_dist_from_domain(dist: &DistMatrix, domain: &[Vec<usize>]) -> DistMatrix {

    let n = dist.n;

    let mut data = vec![u32::MAX; n * n];

    for u in 0..n {

        data[u * n + u] = 0;

        for &v in &domain[u] {

            data[u * n + v] = dist.get(u, v);

        }

    }

    DistMatrix::new(n, data)

}


// ───────────────────────────── TOP-LEVEL SOLVE ─────────────────────────────


struct TspSolver {

    dist: DistMatrix,

}


struct TspResult {

    cost: u64,

    tour: Vec<usize>,

    elapsed: f64,

    upper_bound_initial: u64,

}


impl TspSolver {

    fn new(dist: DistMatrix) -> Self {

        Self { dist }

    }


    fn solve(&self) -> TspResult {

        let t0 = Instant::now();

        let n = self.dist.n;


        if n == 1 {

            return TspResult {

                cost: 0,

                tour: vec![0],

                elapsed: 0.0,

                upper_bound_initial: 0,

            };

        }

        if n == 2 {

            let c = self.dist.get(0, 1) as u64 * 2;

            return TspResult {

                cost: c,

                tour: vec![0, 1],

                elapsed: 0.0,

                upper_bound_initial: c,

            };

        }


        // Step 1: build initial upper bound (NN + 2-Opt)

        let (ub_cost, ub_tour) = build_upper_bound(&self.dist);

        let ub_initial = ub_cost;

        let shared_ub = Arc::new(SharedUB::new(ub_cost, ub_tour));


        // Step 2: prune by a safe cycle lower bound, then run BCcarver's

        // structural preprocessing on the reduced domain.

        let base_graph = build_complete_graph(n);

        let cost_pruned_graph = match prune_edges_by_cost(&self.dist, &base_graph, ub_cost) {

            Some(graph) => graph,

            None => {

                return TspResult {

                    cost: ub_cost,

                    tour: shared_ub.tour.lock().unwrap().clone(),

                    elapsed: t0.elapsed().as_secs_f64(),

                    upper_bound_initial: ub_initial,

                };

            }

        };

        let final_domain = match bccarver::preprocess_graph(&cost_pruned_graph) {

            Some(graph) => graph,

            None => {

                return TspResult {

                    cost: ub_cost,

                    tour: shared_ub.tour.lock().unwrap().clone(),

                    elapsed: t0.elapsed().as_secs_f64(),

                    upper_bound_initial: ub_initial,

                };

            }

        };

        let tight_dist = masked_dist_from_domain(&self.dist, &final_domain);


        // Step 3: compute global statistics for branch ordering

        let (mu, sigma) = tight_dist.mean_stddev();


        // Step 4: enumerate subtasks for parallel splitting

        let subtasks = enumerate_subtasks(&tight_dist, 0, SPLIT_DEPTH);

        if subtasks.is_empty() {

            return TspResult {

                cost: shared_ub.get(),

                tour: shared_ub.tour.lock().unwrap().clone(),

                elapsed: t0.elapsed().as_secs_f64(),

                upper_bound_initial: ub_initial,

            };

        }


        // Step 5: parallel DFS over subtasks

        let dist_ref = &tight_dist;

        let ub_ref = &shared_ub;


        if n <= U128_LIMIT {

            subtasks.into_par_iter().for_each(|task| {

                let mut state = SolverState128::new(dist_ref, ub_ref, mu, sigma);

                let current = *task.path_prefix.last().unwrap();

                let mut path = task.path_prefix.clone();

                state.dfs(

                    current,

                    0,

                    task.visited_u128,

                    task.prefix_cost,

                    path.len(),

                    &mut path,

                );

            });

        } else {

            subtasks.into_par_iter().for_each(|task| {

                let mut state = SolverStateDyn::new(dist_ref, ub_ref, mu, sigma);

                let current = *task.path_prefix.last().unwrap();

                let mut path = task.path_prefix.clone();

                let mut visited = task.visited_dyn.clone();

                state.dfs(current, 0, &mut visited, task.prefix_cost, &mut path);

            });

        }


        let best_cost = shared_ub.get();

        let best_tour = shared_ub.tour.lock().unwrap().clone();


        TspResult {

            cost: best_cost,

            tour: best_tour,

            elapsed: t0.elapsed().as_secs_f64(),

            upper_bound_initial: ub_initial,

        }

    }


    /// Validate a tour: must visit every city exactly once and return to start.

    fn validate(&self, tour: &[usize]) -> bool {

        let n = self.dist.n;

        if tour.len() != n {

            return false;

        }

        let mut seen = vec![false; n];

        for &c in tour {

            if c >= n || seen[c] {

                return false;

            }

            seen[c] = true;

        }

        for i in 0..n {

            if !self.dist.is_edge_allowed(tour[i], tour[(i + 1) % n]) {

                return false;

            }

        }

        true

    }

}


// ───────────────────────────── TSPLIB PARSER ─────────────────────────────


fn parse_tsplib(path: &Path) -> Result<DistMatrix, String> {

    let content =

        fs::read_to_string(path).map_err(|e| format!("Cannot read {}: {}", path.display(), e))?;


    let mut n = 0usize;

    let mut edge_weight_type = String::new();

    let mut edge_weight_format = String::new();

    let mut node_coords: Vec<(f64, f64)> = vec![];

    let mut explicit_weights: Vec<u32> = vec![];

    let mut in_coord_section = false;

    let mut in_weight_section = false;


    for line in content.lines() {

        let line = line.trim();

        if line.is_empty() {

            continue;

        }


        if line.starts_with("DIMENSION") {

            n = line

                .splitn(2, ':')

                .nth(1)

                .unwrap_or("0")

                .trim()

                .parse::<usize>()

                .unwrap_or(0);

        } else if line.starts_with("EDGE_WEIGHT_TYPE") {

            edge_weight_type = line

                .splitn(2, ':')

                .nth(1)

                .unwrap_or("")

                .trim()

                .to_uppercase();

        } else if line.starts_with("EDGE_WEIGHT_FORMAT") {

            edge_weight_format = line

                .splitn(2, ':')

                .nth(1)

                .unwrap_or("")

                .trim()

                .to_uppercase();

        } else if line == "NODE_COORD_SECTION" {

            in_coord_section = true;

            in_weight_section = false;

        } else if line == "EDGE_WEIGHT_SECTION" {

            in_weight_section = true;

            in_coord_section = false;

        } else if line == "EOF" {

            break;

        } else if in_coord_section {

            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() >= 3 {

                if let (Ok(x), Ok(y)) = (parts[1].parse::<f64>(), parts[2].parse::<f64>()) {

                    node_coords.push((x, y));

                }

            }

        } else if in_weight_section {

            for s in line.split_whitespace() {

                if let Ok(w) = s.parse::<u32>() {

                    explicit_weights.push(w);

                }

            }

        }

    }


    if n == 0 {

        return Err("DIMENSION not found".to_string());

    }


    // Build distance matrix

    let mut data = vec![0u32; n * n];


    if !node_coords.is_empty() {

        // EUC_2D or GEO

        for i in 0..n {

            for j in 0..n {

                if i == j {

                    data[i * n + j] = 0;

                    continue;

                }

                if edge_weight_type.contains("GEO") {

                    // GEO: Haversine-style (TSPLIB convention)

                    let (lati, loni) = geo_coords(node_coords[i]);

                    let (latj, lonj) = geo_coords(node_coords[j]);

                    let q1 = (loni - lonj).cos();

                    let q2 = (lati - latj).cos();

                    let q3 = (lati + latj).cos();

                    let d = (6378.388 * (((1.0 + q1) * q2 - (1.0 - q1) * q3) / 2.0).acos() + 1.0)

                        as u32;

                    data[i * n + j] = d;

                } else {

                    // EUC_2D

                    let dx = node_coords[i].0 - node_coords[j].0;

                    let dy = node_coords[i].1 - node_coords[j].1;

                    let d = (dx * dx + dy * dy).sqrt().round() as u32;

                    data[i * n + j] = d;

                }

            }

        }

    } else if !explicit_weights.is_empty() {

        // FULL_MATRIX or LOWER/UPPER variants

        if edge_weight_format.contains("FULL_MATRIX") {

            for i in 0..n {

                for j in 0..n {

                    let idx = i * n + j;

                    if idx < explicit_weights.len() {

                        data[idx] = explicit_weights[idx];

                    }

                }

            }

        } else if edge_weight_format.contains("LOWER_DIAG_ROW")

            || edge_weight_format.contains("LOWER_ROW")

        {

            let mut k = 0;

            for i in 0..n {

                let cols = if edge_weight_format.contains("DIAG") {

                    i + 1

                } else {

                    i

                };

                for j in 0..cols {

                    if k < explicit_weights.len() {

                        data[i * n + j] = explicit_weights[k];

                        data[j * n + i] = explicit_weights[k];

                        k += 1;

                    }

                }

            }

        } else if edge_weight_format.contains("UPPER_DIAG_ROW")

            || edge_weight_format.contains("UPPER_ROW")

        {

            let mut k = 0;

            for i in 0..n {

                let start_j = if edge_weight_format.contains("DIAG") {

                    i

                } else {

                    i + 1

                };

                for j in start_j..n {

                    if k < explicit_weights.len() {

                        data[i * n + j] = explicit_weights[k];

                        data[j * n + i] = explicit_weights[k];

                        k += 1;

                    }

                }

            }

        }

    } else {

        return Err("No coordinate or weight data found".to_string());

    }


    Ok(DistMatrix::new(n, data))

}


fn geo_coords((x, y): (f64, f64)) -> (f64, f64) {

    use std::f64::consts::PI;

    let deg_x = x.trunc();

    let min_x = x - deg_x;

    let lat = PI * (deg_x + 5.0 * min_x / 3.0) / 180.0;

    let deg_y = y.trunc();

    let min_y = y - deg_y;

    let lon = PI * (deg_y + 5.0 * min_y / 3.0) / 180.0;

    (lat, lon)

}


// ───────────────────────────── RANDOM INSTANCE GENERATORS ─────────────────────────────


fn random_euclidean(n: usize, seed: u64) -> DistMatrix {

    // Simple LCG PRNG — no external crate needed

    let mut rng = seed;

    let next = |r: &mut u64| -> f64 {

        *r = r

            .wrapping_mul(6364136223846793005)

            .wrapping_add(1442695040888963407);

        ((*r >> 33) as f64) / (u32::MAX as f64)

    };


    let coords: Vec<(f64, f64)> = (0..n)

        .map(|_| (next(&mut rng) * 1000.0, next(&mut rng) * 1000.0))

        .collect();

    let mut data = vec![0u32; n * n];

    for i in 0..n {

        for j in 0..n {

            if i != j {

                let dx = coords[i].0 - coords[j].0;

                let dy = coords[i].1 - coords[j].1;

                data[i * n + j] = (dx * dx + dy * dy).sqrt().round() as u32;

            }

        }

    }

    DistMatrix::new(n, data)

}


fn random_symmetric(n: usize, max_w: u32, seed: u64) -> DistMatrix {

    let mut rng = seed;

    let next = |r: &mut u64| -> u32 {

        *r = r

            .wrapping_mul(6364136223846793005)

            .wrapping_add(1442695040888963407);

        ((*r >> 33) as u32) % max_w + 1

    };


    let mut data = vec![0u32; n * n];

    for i in 0..n {

        for j in i + 1..n {

            let w = next(&mut rng);

            data[i * n + j] = w;

            data[j * n + i] = w;

        }

    }

    DistMatrix::new(n, data)

}


// ───────────────────────────── TEST SUITE ─────────────────────────────


fn test_small_known() {

    println!("=== Small Known Instances ===\n");

    println!(

        "{:<25} | {:>6} | {:>12} | {:>12} | {:>9} | {:>9}",

        "Instance", "N", "Known OPT", "Found", "Gap%", "Time(s)"

    );

    println!("{}", "-".repeat(90));


    // 4-city cycle: 0→1→2→3→0, all edges = 1 except long diagonals

    {

        let data = vec![0u32, 1, 10, 1, 1, 0, 1, 10, 10, 1, 0, 1, 1, 10, 1, 0];

        let dist = DistMatrix::new(4, data);

        let solver = TspSolver::new(dist);

        let res = solver.solve();

        let opt = 4u64;

        let gap = (res.cost as f64 - opt as f64) / opt as f64 * 100.0;

        let status = if res.cost == opt { "✅" } else { "❌" };

        println!(

            "{:<25} | {:>6} | {:>12} | {:>12} | {:>8.2}% | {:>9.6} {}",

            "Square (opt=4)", 4, opt, res.cost, gap, res.elapsed, status

        );

    }


    // 5 cities on a regular pentagon (all equal edges = 1) → opt = 5

    {

        let n = 5;

        let mut data = vec![0u32; n * n];

        // Euclidean regular pentagon, all edges computed

        let coords: Vec<(f64, f64)> = (0..n)

            .map(|i| {

                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;

                (angle.cos(), angle.sin())

            })

            .collect();

        for i in 0..n {

            for j in 0..n {

                if i != j {

                    let dx = coords[i].0 - coords[j].0;

                    let dy = coords[i].1 - coords[j].1;

                    data[i * n + j] = ((dx * dx + dy * dy).sqrt() * 100.0).round() as u32;

                }

            }

        }

        let dist = DistMatrix::new(n, data.clone());

        let adj_cost = data[0 * n + 1];

        let opt = adj_cost as u64 * 5;

        let solver = TspSolver::new(dist);

        let res = solver.solve();

        let gap = (res.cost as f64 - opt as f64) / opt as f64 * 100.0;

        let status = if res.cost == opt { "✅" } else { "❌" };

        println!(

            "{:<25} | {:>6} | {:>12} | {:>12} | {:>8.2}% | {:>9.6} {}",

            "Pentagon (opt=5*adj)", n, opt, res.cost, gap, res.elapsed, status

        );

    }


    // Asymmetric-looking but symmetric matrix: known 6-city

    // Hand-crafted optimal = 14

    {

        let data = vec![

            0u32, 2, 9, 10, 10, 10, 2, 0, 6, 4, 3, 10, 9, 6, 0, 8, 5, 3, 10, 4, 8, 0, 7, 6, 10, 3,

            5, 7, 0, 2, 10, 10, 3, 6, 2, 0,

        ];

        let dist = DistMatrix::new(6, data);

        let solver = TspSolver::new(dist);

        let res = solver.solve();

        println!(

            "{:<25} | {:>6} | {:>12} | {:>12} | {:>9} | {:>9.6} {}",

            "6-city custom",

            6,

            "?",

            res.cost,

            "-",

            res.elapsed,

            if solver.validate(&res.tour) {

                "✅"

            } else {

                "❌"

            }

        );

    }


    println!();

}


fn bench_random(sizes: &[usize]) {

    println!("=== Random Euclidean Benchmark ===\n");

    println!(

        "{:>6} | {:>12} | {:>12} | {:>9} | {:>12}",

        "N", "NN+2-Opt UB", "B&B Optimal", "Time(s)", "Threads"

    );

    println!("{}", "-".repeat(65));


    let threads = rayon::current_num_threads();


    for &n in sizes {

        let dist = random_euclidean(n, 42 + n as u64);

        let solver = TspSolver::new(dist);

        let res = solver.solve();

        let gap = (res.upper_bound_initial as f64 - res.cost as f64) / res.cost as f64 * 100.0;

        let valid = if solver.validate(&res.tour) {

            "✅"

        } else {

            "❌"

        };

        println!(

            "{:>6} | {:>12} | {:>12} ({:+.1}%) | {:>9.4} | {:>12} {}",

            n, res.upper_bound_initial, res.cost, gap, res.elapsed, threads, valid

        );

    }

    println!();

}


// ───────────────────────────── PRE-PROCESSING ─────────────────────────────


#[derive(Clone, Copy)]

struct VertexMinima {

    first_cost: u64,

    first_neighbor: usize,

    second_cost: u64,

}


fn build_complete_graph(n: usize) -> Vec<Vec<usize>> {

    (0..n)

        .map(|u| (0..n).filter(|&v| v != u).collect())

        .collect()

}


fn compute_vertex_minima(

    dist: &DistMatrix,

    graph: &[Vec<usize>],

) -> Option<(Vec<VertexMinima>, u64)> {

    let n = dist.n;

    let mut minima = Vec::with_capacity(n);

    let mut total_two_min_sum = 0u64;


    for (u, neigh) in graph.iter().enumerate() {

        if neigh.len() < 2 {

            return None;

        }


        let mut first_cost = u64::MAX;

        let mut first_neighbor = usize::MAX;

        let mut second_cost = u64::MAX;


        for &v in neigh {

            let cost = dist.get(u, v) as u64;

            if cost < first_cost {

                second_cost = first_cost;

                first_cost = cost;

                first_neighbor = v;

            } else if cost < second_cost {

                second_cost = cost;

            }

        }


        if first_neighbor == usize::MAX || second_cost == u64::MAX {

            return None;

        }


        minima.push(VertexMinima {

            first_cost,

            first_neighbor,

            second_cost,

        });

        total_two_min_sum = total_two_min_sum.saturating_add(first_cost + second_cost);

    }


    Some((minima, total_two_min_sum))

}


fn prune_edges_by_cost(

    dist: &DistMatrix,

    graph: &[Vec<usize>],

    current_min: u64,

) -> Option<Vec<Vec<usize>>> {

    let n = dist.n;

    let mut pruned = graph.to_vec();


    loop {

        let (minima, total_two_min_sum) = compute_vertex_minima(dist, &pruned)?;

        let mut to_remove = Vec::new();


        for u in 0..n {

            let sum_u = minima[u].first_cost + minima[u].second_cost;

            for &v in &pruned[u] {

                if u >= v {

                    continue;

                }


                let sum_v = minima[v].first_cost + minima[v].second_cost;

                let edge_cost = dist.get(u, v) as u64;

                let extra_u = if minima[u].first_neighbor != v {

                    minima[u].first_cost

                } else {

                    minima[u].second_cost

                };

                let extra_v = if minima[v].first_neighbor != u {

                    minima[v].first_cost

                } else {

                    minima[v].second_cost

                };


                let forced_incident_sum = total_two_min_sum

                    .saturating_sub(sum_u)

                    .saturating_sub(sum_v)

                    .saturating_add(edge_cost.saturating_mul(2))

                    .saturating_add(extra_u)

                    .saturating_add(extra_v);

                let cycle_lb = (forced_incident_sum + 1) / 2;


                if cycle_lb >= current_min {

                    to_remove.push((u, v));

                }

            }

        }


        if to_remove.is_empty() {

            return Some(pruned);

        }


        for (u, v) in to_remove {

            pruned[u].retain(|&x| x != v);

            pruned[v].retain(|&x| x != u);

        }


        if pruned.iter().any(|adj| adj.len() < 2) {

            return None;

        }

    }

}


// ───────────────────────────── MAIN ─────────────────────────────


fn main() {

    let args: Vec<String> = env::args().collect();


    // --threads N override

    let mut arg_idx = 1;

    if args.get(1).map(|s| s.as_str()) == Some("--threads") {

        if let Some(n) = args.get(2).and_then(|s| s.parse::<usize>().ok()) {

            rayon::ThreadPoolBuilder::new()

                .num_threads(n)

                .build_global()

                .ok();

            arg_idx = 3;

        }

    }


    let threads = rayon::current_num_threads();

    println!("BCtravel v2 - Exact TSP with Held-Karp + BCcarver preprocessing");

    println!(

        "Threads: {}  TSP split depth: {}  BCcarver split depth: {}\n",

        threads,

        SPLIT_DEPTH,

        bccarver::split_depth()

    );


    match args.get(arg_idx).map(|s| s.as_str()) {

        None => {

            test_small_known();

            bench_random(&[

                6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 41, 50, 75, 100, 125, 150, 200, 250, 300,

                400, 500, 750, 1000,

            ]);

        }


        Some("--tsp") => {

            let path = match args.get(arg_idx + 1) {

                Some(p) => Path::new(p),

                None => {

                    eprintln!("Usage: --tsp <file.tsp>");

                    return;

                }

            };

            match parse_tsplib(path) {

                Err(e) => eprintln!("Parse error: {}", e),

                Ok(dist) => {

                    println!("Instance: {}  N={}", path.display(), dist.n);

                    let solver = TspSolver::new(dist);

                    let res = solver.solve();

                    println!("Optimal tour cost : {}", res.cost);

                    println!("Initial UB (NN+2-Opt): {}", res.upper_bound_initial);

                    println!("Valid: {}", solver.validate(&res.tour));

                    println!("Time: {:.4}s", res.elapsed);

                    println!("Tour: {:?}", &res.tour[..res.tour.len().min(30)]);

                }

            }

        }


        Some("--random") => {

            let n: usize = args

                .get(arg_idx + 1)

                .and_then(|s| s.parse().ok())

                .unwrap_or(12);

            let seed: u64 = args

                .get(arg_idx + 2)

                .and_then(|s| s.parse().ok())

                .unwrap_or(42);

            println!("B&B random Euclidean N={} seed={}", n, seed);

            let dist = random_euclidean(n, seed);

            let solver = TspSolver::new(dist);

            let res = solver.solve();

            println!("Optimal cost: {}", res.cost);

            println!("NN+2-Opt UB:  {}", res.upper_bound_initial);

            println!("Valid: {}", solver.validate(&res.tour));

            println!("Time: {:.4}s", res.elapsed);

            if res.tour.len() <= 30 {

                println!("Tour: {:?}", res.tour);

            }

        }


        Some("--bench") => {

            let n0: usize = args

                .get(arg_idx + 1)

                .and_then(|s| s.parse().ok())

                .unwrap_or(6);

            let n1: usize = args

                .get(arg_idx + 2)

                .and_then(|s| s.parse().ok())

                .unwrap_or(20);

            bench_random(&(n0..=n1).collect::<Vec<_>>());

        }


        Some("--symmetric") => {

            let n: usize = args

                .get(arg_idx + 1)

                .and_then(|s| s.parse().ok())

                .unwrap_or(12);

            let max_w: u32 = args

                .get(arg_idx + 2)

                .and_then(|s| s.parse().ok())

                .unwrap_or(100);

            let seed: u64 = args

                .get(arg_idx + 3)

                .and_then(|s| s.parse().ok())

                .unwrap_or(42);

            println!("B&B random symmetric N={} max_w={} seed={}", n, max_w, seed);

            let dist = random_symmetric(n, max_w, seed);

            let solver = TspSolver::new(dist);

            let res = solver.solve();

            println!("Optimal cost: {}", res.cost);

            println!("NN+2-Opt UB:  {}", res.upper_bound_initial);

            println!("Valid: {}", solver.validate(&res.tour));

            println!("Time: {:.4}s", res.elapsed);

        }


        _ => {

            eprintln!("Usage: bctravel [--threads N] <command>");

            eprintln!("  bctravel                              default suite");

            eprintln!("  --tsp <file.tsp>                      B&B solve TSPLIB");

            eprintln!("  --random N [seed]                     B&B random instance");

            eprintln!("  --bench N0 N1                         B&B benchmark range");

            eprintln!("  --symmetric N [max_w] [seed]          B&B non-metric instance");

        }

    }

} 