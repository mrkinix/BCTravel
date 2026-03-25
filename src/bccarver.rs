// bc_craver_v7.rs — Hamiltonian Cycle Solver (Ben-Chiboub Carver) — Parallel Edition
//
// USAGE:
//   cargo run --release                          → built-in suite + parallel random audit
//   cargo run --release -- file.hcp              → solve one .hcp file (parallel branching)
//   cargo run --release -- --fhcp dir/ [timeout] → FHCP benchmark (parallel instances)
//   cargo run --release -- --random N0 N1 [p]    → random audit (parallel instances)
//   cargo run --release -- --threads N           → override thread count (default: num_cpus)

//
// PARALLELISM STRATEGY:
//   Single instance  → parallel tree splitting: run initial propagation once,
//                      enumerate branch decisions to depth SPLIT_DEPTH (default 4),
//                      producing up to 2^SPLIT_DEPTH independent subtrees,
//                      solved concurrently on the rayon thread pool.
//                      First SAT wins; all must report UNSAT for global UNSAT.
//
//   Benchmark runner → parallel instances: each graph solved on its own thread.
//                      8 cores → ~8x throughput. No synchronisation needed.

use rayon::prelude::*;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// How many branch levels to split before parallelising.
// 4 → up to 16 subtrees. 8 cores → tune to 3 or 4.
// Higher values give more subtrees but more wasted work if SAT is found early.
const SPLIT_DEPTH: usize = 4;

// ===================== DATA TYPES =====================

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub struct Edge(pub usize, pub usize);

#[derive(Copy, Clone)]
enum Change {
    Lock(usize),
    Delete(usize),
}

enum PropResult {
    Continue,
    Contradiction,
    Solved,
}

// ===================== SOLVER STRUCT =====================

#[derive(Clone)]
pub struct BcCraver {
    n: usize,
    g_orig: Vec<Vec<usize>>,
    memo_cache: HashMap<(Vec<u64>, Vec<u64>), ()>,
    best_path: Option<Vec<Edge>>,
    all_edges: Vec<Edge>,
    edge_id: HashMap<Edge, usize>,
    node_edges: Vec<Vec<usize>>, // node → list of edge ids incident to it
    locked_bits: Vec<u64>,
    deleted_bits: Vec<u64>,
    undo_stack: Vec<Change>,
    locked_degree: Vec<usize>,
    total_deletions: usize,
    words: usize,                // number of u64 words per adjacency row
    g_avail_bits: Vec<Vec<u64>>, // current available adjacency (bitset)
    avail_deg: Vec<usize>,
    orig_bits: Vec<Vec<u64>>,
    orig_deg: Vec<usize>,
    // Path endpoint tracking: the partial locked path has exactly 2 "open ends"
    // (nodes with locked_degree == 1). Tracking them lets us apply the
    // path-closure prune without a full graph scan.
    path_endpoints: Vec<usize>, // 0 or 2 nodes with locked_degree == 1
}

impl BcCraver {
    pub fn new(g: &[Vec<usize>]) -> Self {
        let n = g.len();

        let mut edge_set: HashSet<Edge> = HashSet::new();
        for u in 0..n {
            for &v in &g[u] {
                if u < v {
                    edge_set.insert(Edge(u, v));
                }
            }
        }
        let mut all_edges: Vec<Edge> = edge_set.into_iter().collect();
        // Sort for deterministic behaviour across runs
        all_edges.sort_unstable_by_key(|e| (e.0, e.1));

        let mut edge_id: HashMap<Edge, usize> = HashMap::new();
        for (i, &e) in all_edges.iter().enumerate() {
            edge_id.insert(e, i);
        }

        let m = all_edges.len();
        let num_words = (m * 2 + 63) / 64;

        let mut node_edges: Vec<Vec<usize>> = vec![vec![]; n];
        for (id, &Edge(u, v)) in all_edges.iter().enumerate() {
            node_edges[u].push(id);
            node_edges[v].push(id);
        }

        let words = (n + 63) / 64;
        let mut g_avail_bits = vec![vec![0u64; words]; n];
        let mut avail_deg = vec![0usize; n];
        for u in 0..n {
            avail_deg[u] = g[u].len();
            for &v in &g[u] {
                let w = v / 64;
                let b = v % 64;
                if w < words {
                    g_avail_bits[u][w] |= 1u64 << b;
                }
            }
        }
        let orig_bits = g_avail_bits.clone();
        let orig_deg = avail_deg.clone();

        BcCraver {
            n,
            g_orig: g.to_vec(),
            memo_cache: HashMap::new(),
            best_path: None,
            all_edges,
            edge_id,
            node_edges,
            locked_bits: vec![0u64; num_words],
            deleted_bits: vec![0u64; num_words],
            undo_stack: vec![],
            locked_degree: vec![0; n],
            total_deletions: 0,
            words,
            g_avail_bits,
            avail_deg,
            orig_bits,
            orig_deg,
            path_endpoints: vec![],
        }
    }

    // ===================== BIT OPERATIONS =====================

    #[inline]
    fn is_locked(&self, id: usize) -> bool {
        (self.locked_bits[(id * 2) / 64] & (1u64 << ((id * 2) % 64))) != 0
    }
    #[inline]
    fn is_deleted(&self, id: usize) -> bool {
        (self.deleted_bits[(id * 2 + 1) / 64] & (1u64 << ((id * 2 + 1) % 64))) != 0
    }
    #[inline]
    fn is_active(&self, id: usize) -> bool {
        !self.is_locked(id) && !self.is_deleted(id)
    }
    #[inline]
    fn clear_avail(&mut self, u: usize, v: usize) {
        let w = v / 64;
        let b = v % 64;
        if w < self.words {
            self.g_avail_bits[u][w] &= !(1u64 << b);
        }
    }
    #[inline]
    fn set_avail(&mut self, u: usize, v: usize) {
        let w = v / 64;
        let b = v % 64;
        if w < self.words {
            self.g_avail_bits[u][w] |= 1u64 << b;
        }
    }
    #[inline]
    fn has_edge(&self, u: usize, v: usize) -> bool {
        if v >= self.n {
            return false;
        }
        (self.g_avail_bits[u][v / 64] & (1u64 << (v % 64))) != 0
    }

    // ===================== APPLY / UNDO =====================

    fn apply_lock(&mut self, id: usize) {
        self.locked_bits[(id * 2) / 64] |= 1u64 << ((id * 2) % 64);
        let Edge(u, v) = self.all_edges[id];
        self.locked_degree[u] += 1;
        self.locked_degree[v] += 1;
        // Update path endpoints
        for &node in &[u, v] {
            let ld = self.locked_degree[node];
            if ld == 1 {
                self.path_endpoints.push(node);
            } else if ld == 2 {
                self.path_endpoints.retain(|&x| x != node);
            }
        }
        self.undo_stack.push(Change::Lock(id));
    }

    fn apply_delete(&mut self, id: usize) {
        self.deleted_bits[(id * 2 + 1) / 64] |= 1u64 << ((id * 2 + 1) % 64);
        let Edge(u, v) = self.all_edges[id];
        self.clear_avail(u, v);
        self.clear_avail(v, u);
        self.avail_deg[u] = self.avail_deg[u].saturating_sub(1);
        self.avail_deg[v] = self.avail_deg[v].saturating_sub(1);
        self.total_deletions += 1;
        self.undo_stack.push(Change::Delete(id));
    }

    fn undo(&mut self) {
        match self.undo_stack.pop() {
            Some(Change::Lock(id)) => {
                self.locked_bits[(id * 2) / 64] &= !(1u64 << ((id * 2) % 64));
                let Edge(u, v) = self.all_edges[id];
                self.locked_degree[u] -= 1;
                self.locked_degree[v] -= 1;
                // Restore path endpoints
                for &node in &[u, v] {
                    let ld = self.locked_degree[node];
                    if ld == 1 {
                        self.path_endpoints.push(node);
                    } else if ld == 0 {
                        self.path_endpoints.retain(|&x| x != node);
                    }
                }
            }
            Some(Change::Delete(id)) => {
                self.deleted_bits[(id * 2 + 1) / 64] &= !(1u64 << ((id * 2 + 1) % 64));
                let Edge(u, v) = self.all_edges[id];
                self.set_avail(u, v);
                self.set_avail(v, u);
                self.avail_deg[u] += 1;
                self.avail_deg[v] += 1;
                self.total_deletions -= 1;
            }
            None => {}
        }
    }

    fn undo_to(&mut self, target: usize) {
        while self.undo_stack.len() > target {
            self.undo();
        }
    }

    // ===================== MEMOIZATION =====================

    fn is_seen(&self) -> bool {
        let key = (self.locked_bits.clone(), self.deleted_bits.clone());
        self.memo_cache.contains_key(&key)
    }

    fn memoize(&mut self) {
        let key = (self.locked_bits.clone(), self.deleted_bits.clone());
        self.memo_cache.insert(key, ());
    }

    fn get_memo_size(&self) -> usize {
        self.memo_cache.len()
    }

    // ===================== GRAPH QUERIES =====================

    fn get_avail_neighbors(&self, u: usize) -> Vec<usize> {
        let mut res = Vec::with_capacity(self.avail_deg[u]);
        for wi in 0..self.words {
            let mut word = self.g_avail_bits[u][wi];
            while word != 0 {
                let tz = word.trailing_zeros() as usize;
                let v = wi * 64 + tz;
                if v < self.n {
                    res.push(v);
                }
                word &= word - 1;
            }
        }
        res
    }

    fn build_avail_adj(&self) -> Vec<Vec<usize>> {
        (0..self.n).map(|u| self.get_avail_neighbors(u)).collect()
    }

    fn build_locked_graph(&self) -> Vec<Vec<usize>> {
        let mut gl = vec![vec![]; self.n];
        for (id, &Edge(u, v)) in self.all_edges.iter().enumerate() {
            if self.is_locked(id) {
                gl[u].push(v);
                gl[v].push(u);
            }
        }
        gl
    }

    fn collect_locked(&self) -> Vec<Edge> {
        self.all_edges
            .iter()
            .enumerate()
            .filter(|&(id, _)| self.is_locked(id))
            .map(|(_, &e)| e)
            .collect()
    }

    fn is_connected_iter(&self, g: &[Vec<usize>]) -> bool {
        if self.n == 0 {
            return true;
        }
        let mut visited = vec![false; self.n];
        let mut stack = vec![0usize];
        visited[0] = true;
        while let Some(u) = stack.pop() {
            for &v in &g[u] {
                if !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        visited.iter().all(|&x| x)
    }

    fn connected_components(&self, g: &[Vec<usize>]) -> usize {
        let mut visited = vec![false; self.n];
        let mut count = 0;
        for start in 0..self.n {
            if !visited[start] {
                count += 1;
                let mut stack = vec![start];
                visited[start] = true;
                while let Some(u) = stack.pop() {
                    for &v in &g[u] {
                        if !visited[v] {
                            visited[v] = true;
                            stack.push(v);
                        }
                    }
                }
            }
        }
        count
    }

    fn has_subcycle(&self, gl: &[Vec<usize>]) -> bool {
        let mut visited = vec![false; self.n];
        for start in 0..self.n {
            if !visited[start] {
                let mut comp = vec![];
                let mut stack = vec![start];
                visited[start] = true;
                comp.push(start);
                while let Some(u) = stack.pop() {
                    for &v in &gl[u] {
                        if !visited[v] {
                            visited[v] = true;
                            comp.push(v);
                            stack.push(v);
                        }
                    }
                }
                if comp.len() < self.n && comp.iter().all(|&u| gl[u].len() == 2) {
                    return true;
                }
            }
        }
        false
    }

    fn is_full_cycle(&self, gl: &[Vec<usize>]) -> bool {
        gl.iter().all(|adj| adj.len() == 2) && self.is_connected_iter(gl)
    }

    fn has_articulation_point(&self, g: &[Vec<usize>]) -> bool {
        if self.n <= 2 {
            return false;
        }
        let mut disc = vec![-1i32; self.n];
        let mut low = vec![0i32; self.n];
        let mut parent = vec![-1i32; self.n];
        let mut timer = 0i32;
        let mut stack: Vec<(usize, usize)> = vec![];

        for root in 0..self.n {
            if disc[root] != -1 {
                continue;
            }
            disc[root] = timer;
            low[root] = timer;
            timer += 1;
            stack.push((root, 0));
            let mut root_children = 0usize;

            while let Some((u, ni)) = stack.last_mut() {
                let u = *u;
                if *ni < g[u].len() {
                    let v = g[u][*ni];
                    *ni += 1;
                    if disc[v] == -1 {
                        if parent[u] == -1 {
                            root_children += 1;
                        }
                        parent[v] = u as i32;
                        disc[v] = timer;
                        low[v] = timer;
                        timer += 1;
                        stack.push((v, 0));
                    } else if v as i32 != parent[u] {
                        low[u] = min(low[u], disc[v]);
                    }
                } else {
                    stack.pop();
                    if let Some(&(p, _)) = stack.last() {
                        low[p] = min(low[p], low[u]);
                        if parent[p] != -1 && low[u] >= disc[p] {
                            return true;
                        }
                    }
                }
            }
            if root_children > 1 {
                return true;
            }
        }
        false
    }

    fn path_endpoints_connected(&self) -> bool {
        if self.path_endpoints.len() != 2 {
            return true;
        }
        let a = self.path_endpoints[0];
        let b = self.path_endpoints[1];
        let mut visited = vec![false; self.n];
        let mut queue = VecDeque::new();
        visited[a] = true;
        queue.push_back(a);

        while let Some(u) = queue.pop_front() {
            if u == b {
                return true;
            }
            for wi in 0..self.words {
                let mut word = self.g_avail_bits[u][wi];
                while word != 0 {
                    let tz = word.trailing_zeros() as usize;
                    let v = wi * 64 + tz;
                    word &= word - 1;
                    if v < self.n && !visited[v] {
                        visited[v] = true;
                        queue.push_back(v);
                    }
                }
            }
        }
        false
    }

    // ===================== BEN-CHIBOUB RULES =====================

    fn rule_bc1_diamond_chain(&mut self) -> bool {
        let mut changed = false;
        for c in 0..self.n {
            if self.avail_deg[c] != 2 || self.locked_degree[c] >= 2 {
                continue;
            }
            let c_neigh = self.get_avail_neighbors(c);
            if c_neigh.len() != 2 {
                continue;
            }
            let (b, d) = (c_neigh[0], c_neigh[1]);

            if !self.has_edge(b, d) {
                continue;
            }

            let b_neigh: Vec<usize> = self
                .get_avail_neighbors(b)
                .into_iter()
                .filter(|&x| x != c && x != d)
                .collect();
            if b_neigh.len() != 1 {
                continue;
            }

            let d_neigh: Vec<usize> = self
                .get_avail_neighbors(d)
                .into_iter()
                .filter(|&x| x != b && x != c)
                .collect();
            if d_neigh.len() != 1 {
                continue;
            }

            let e_bd = Edge(min(b, d), max(b, d));
            if let Some(&id) = self.edge_id.get(&e_bd) {
                if self.is_active(id) {
                    self.apply_delete(id);
                    changed = true;
                }
            }
            let a = b_neigh[0];
            let e_node = d_neigh[0];

            let e_ab = Edge(min(a, b), max(a, b));
            if let Some(&id) = self.edge_id.get(&e_ab) {
                if self.is_active(id) && self.locked_degree[a] < 2 && self.locked_degree[b] < 2 {
                    self.apply_lock(id);
                    changed = true;
                }
            }
            let e_de = Edge(min(d, e_node), max(d, e_node));
            if let Some(&id) = self.edge_id.get(&e_de) {
                if self.is_active(id) && self.locked_degree[d] < 2 && self.locked_degree[e_node] < 2
                {
                    self.apply_lock(id);
                    changed = true;
                }
            }
        }
        changed
    }

    fn rule_bc2_ladder_rung(&mut self) -> bool {
        let mut changed = false;
        for u in 0..self.n {
            if self.avail_deg[u] != 3 || self.locked_degree[u] != 0 {
                continue;
            }
            let u_neigh = self.get_avail_neighbors(u);
            if u_neigh.len() != 3 {
                continue;
            }

            for &v in &u_neigh {
                if v <= u {
                    continue;
                }
                if self.avail_deg[v] != 3 || self.locked_degree[v] != 0 {
                    continue;
                }

                let u_others: Vec<usize> = u_neigh.iter().copied().filter(|&x| x != v).collect();
                if u_others.len() != 2 {
                    continue;
                }
                let (u1, u2) = (u_others[0], u_others[1]);

                let v_neigh = self.get_avail_neighbors(v);
                let v_others: Vec<usize> = v_neigh.iter().copied().filter(|&x| x != u).collect();
                if v_others.len() != 2 {
                    continue;
                }
                let (v1, v2) = (v_others[0], v_others[1]);

                if !self.has_edge(u1, u2) || !self.has_edge(v1, v2) {
                    continue;
                }

                if self.has_edge(u1, v1)
                    || self.has_edge(u1, v2)
                    || self.has_edge(u2, v1)
                    || self.has_edge(u2, v2)
                {
                    continue;
                }

                let e_uv = Edge(min(u, v), max(u, v));
                if let Some(&id) = self.edge_id.get(&e_uv) {
                    if self.is_active(id) {
                        self.apply_lock(id);
                        changed = true;
                    }
                }
                let e_u12 = Edge(min(u1, u2), max(u1, u2));
                if let Some(&id) = self.edge_id.get(&e_u12) {
                    if self.is_active(id) {
                        self.apply_delete(id);
                        changed = true;
                    }
                }
                let e_v12 = Edge(min(v1, v2), max(v1, v2));
                if let Some(&id) = self.edge_id.get(&e_v12) {
                    if self.is_active(id) {
                        self.apply_delete(id);
                        changed = true;
                    }
                }
                break;
            }
        }
        changed
    }

    fn rule_bc3_square_close(&mut self) -> bool {
        let mut changed = false;
        let mut to_delete: Vec<usize> = vec![];

        for b in 0..self.n {
            if self.locked_degree[b] < 1 {
                continue;
            }
            let locked_b: Vec<usize> = self.node_edges[b]
                .iter()
                .copied()
                .filter(|&id| self.is_locked(id))
                .map(|id| {
                    let Edge(u, v) = self.all_edges[id];
                    if u == b {
                        v
                    } else {
                        u
                    }
                })
                .collect();

            for i in 0..locked_b.len() {
                let a = locked_b[i];
                for j in (i + 1)..locked_b.len() {
                    let c = locked_b[j];
                    if self.has_edge(a, c) {
                        continue;
                    }

                    let a_neigh = self.get_avail_neighbors(a);
                    for &d in &a_neigh {
                        if d == b || d == c {
                            continue;
                        }
                        if !self.has_edge(c, d) {
                            continue;
                        }

                        let e_ad = Edge(min(a, d), max(a, d));
                        let e_cd = Edge(min(c, d), max(c, d));
                        let ad_locked = self
                            .edge_id
                            .get(&e_ad)
                            .map(|&id| self.is_locked(id))
                            .unwrap_or(false);
                        let cd_locked = self
                            .edge_id
                            .get(&e_cd)
                            .map(|&id| self.is_locked(id))
                            .unwrap_or(false);

                        let locked_count = 2 + ad_locked as usize + cd_locked as usize;

                        if locked_count == 3 {
                            if !ad_locked {
                                if let Some(&id) = self.edge_id.get(&e_ad) {
                                    if self.is_active(id) {
                                        to_delete.push(id);
                                    }
                                }
                            }
                            if !cd_locked {
                                if let Some(&id) = self.edge_id.get(&e_cd) {
                                    if self.is_active(id) {
                                        to_delete.push(id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        to_delete.sort_unstable();
        to_delete.dedup();
        for id in to_delete {
            if self.is_active(id) {
                self.apply_delete(id);
                changed = true;
            }
        }
        changed
    }

    // Rule BC-4: "Bipartite cluster capacity" (Meredith Pruning)
    // Identifies subsets of nodes forming an independent set I that share an exact
    // identical available neighborhood P. Imposes strict capacity limits on edges leaving P.
    fn rule_bc4_bipartite_capacity(&mut self) -> bool {
        let mut changed = false;
        let mut nodes: Vec<usize> = (0..self.n).collect();

        // Group by identical available neighborhoods. Sort by avail_deg first to speed up.
        nodes.sort_unstable_by(|&a, &b| {
            self.avail_deg[a]
                .cmp(&self.avail_deg[b])
                .then_with(|| self.g_avail_bits[a].cmp(&self.g_avail_bits[b]))
        });

        let mut is_i = vec![false; self.n];
        let mut is_p = vec![false; self.n];

        let mut i = 0;
        while i < self.n {
            let u = nodes[i];
            if self.avail_deg[u] == 0 {
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < self.n
                && self.avail_deg[u] == self.avail_deg[nodes[j]]
                && self.g_avail_bits[u] == self.g_avail_bits[nodes[j]]
            {
                j += 1;
            }

            let k = j - i;
            if k >= 2 {
                let p_set = self.get_avail_neighbors(u);
                let m = p_set.len();

                // Standard bipartite toughness check
                if m < k {
                    self.avail_deg[u] = 0; // Trigger contradiction in main loop
                    return true;
                }

                // If isolated from graph
                if 2 * m == 2 * k && self.n > k + m {
                    self.avail_deg[u] = 0;
                    return true;
                }

                let max_cap = 2 * m - 2 * k;
                let mut used_cap = 0;
                let mut avail_targets = vec![];

                for x in i..j {
                    is_i[nodes[x]] = true;
                }
                for &p in &p_set {
                    is_p[p] = true;
                }

                for &p in &p_set {
                    for &id in &self.node_edges[p] {
                        let Edge(eu, ev) = self.all_edges[id];
                        let other = if eu == p { ev } else { eu };

                        // Edges directly into the independent set I consume the dedicated 2k capacity
                        if is_i[other] {
                            continue;
                        }

                        if self.is_locked(id) {
                            used_cap += 1;
                        } else if self.is_active(id) {
                            // Only collect each potential edge once
                            if p < other || !is_p[other] {
                                avail_targets.push(id);
                            }
                        }
                    }
                }

                for x in i..j {
                    is_i[nodes[x]] = false;
                }
                for &p in &p_set {
                    is_p[p] = false;
                }

                if used_cap > max_cap {
                    self.avail_deg[u] = 0; // Trigger contradiction
                    return true;
                } else if used_cap == max_cap && !avail_targets.is_empty() {
                    for id in avail_targets {
                        if self.is_active(id) {
                            self.apply_delete(id);
                            changed = true;
                        }
                    }
                }
            }
            i = j;
        }
        changed
    }

    // ===================== FORCED PROPAGATION =====================

    fn do_forced_propagation(&mut self, entry_len: usize) -> PropResult {
        let mut last_deletion_count = usize::MAX;
        let mut changed = true;

        while changed {
            changed = false;

            if self.avail_deg.iter().any(|&d| d < 2) {
                self.undo_to(entry_len);
                return PropResult::Contradiction;
            }

            if self.locked_degree.iter().any(|&d| d > 2) {
                self.undo_to(entry_len);
                return PropResult::Contradiction;
            }

            if self.total_deletions != last_deletion_count {
                let adj = self.build_avail_adj();
                if self.connected_components(&adj) > 1 {
                    self.undo_to(entry_len);
                    return PropResult::Contradiction;
                }
                if self.has_articulation_point(&adj) {
                    self.undo_to(entry_len);
                    return PropResult::Contradiction;
                }
                last_deletion_count = self.total_deletions;
            }

            {
                let mut forced_onto = vec![0usize; self.n];
                for i in 0..self.n {
                    if self.avail_deg[i] == 2 {
                        for wi in 0..self.words {
                            let mut word = self.g_avail_bits[i][wi];
                            while word != 0 {
                                let tz = word.trailing_zeros() as usize;
                                let v = wi * 64 + tz;
                                word &= word - 1;
                                if v < self.n {
                                    forced_onto[v] += 1;
                                }
                            }
                        }
                    }
                }
                if forced_onto.iter().any(|&c| c > 2) {
                    self.undo_to(entry_len);
                    return PropResult::Contradiction;
                }
            }

            for node in 0..self.n {
                if self.avail_deg[node] == 2 && self.locked_degree[node] < 2 {
                    let neigh = self.get_avail_neighbors(node);
                    for &v in &neigh {
                        let e = Edge(min(node, v), max(node, v));
                        if let Some(&id) = self.edge_id.get(&e) {
                            if !self.is_locked(id) {
                                self.apply_lock(id);
                                changed = true;
                            }
                        }
                    }
                    if neigh.len() == 2 {
                        let (m1, m2) = (neigh[0], neigh[1]);
                        if m1 != m2 && self.has_edge(m1, m2) {
                            let ec = Edge(min(m1, m2), max(m1, m2));
                            if let Some(&id) = self.edge_id.get(&ec) {
                                if !self.is_locked(id) && !self.is_deleted(id) {
                                    self.apply_delete(id);
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                if self.locked_degree[node] == 2 && self.avail_deg[node] > 2 {
                    let to_del: Vec<usize> = self.node_edges[node]
                        .iter()
                        .copied()
                        .filter(|&id| self.is_active(id))
                        .collect();
                    for id in to_del {
                        self.apply_delete(id);
                        changed = true;
                    }
                }

                if self.avail_deg[node] == 3 && self.locked_degree[node] == 1 {
                    let neigh = self.get_avail_neighbors(node);
                    for &v in &neigh {
                        let e = Edge(min(node, v), max(node, v));
                        if let Some(&eid) = self.edge_id.get(&e) {
                            if !self.is_active(eid) {
                                continue;
                            }
                            let v_locked_after = self.locked_degree[v] + 1;
                            if v_locked_after == 2 {
                                let v_neigh = self.get_avail_neighbors(v);
                                let mut starvation = false;
                                for &w in &v_neigh {
                                    if w == node {
                                        continue;
                                    }
                                    if self.avail_deg[w].saturating_sub(1) < 2
                                        && self.locked_degree[w] < 2
                                    {
                                        starvation = true;
                                        break;
                                    }
                                }
                                if starvation {
                                    if !self.is_deleted(eid) {
                                        self.apply_delete(eid);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if self.rule_bc1_diamond_chain() {
                changed = true;
            }
            if self.rule_bc2_ladder_rung() {
                changed = true;
            }
            if self.rule_bc3_square_close() {
                changed = true;
            }
            if self.rule_bc4_bipartite_capacity() {
                changed = true;
            }

            if self.avail_deg.iter().any(|&d| d < 2) || self.locked_degree.iter().any(|&d| d > 2) {
                self.undo_to(entry_len);
                return PropResult::Contradiction;
            }

            if self.undo_stack.len() > entry_len {
                let gl = self.build_locked_graph();
                if self.has_subcycle(&gl) {
                    self.undo_to(entry_len);
                    return PropResult::Contradiction;
                }
                if self.is_full_cycle(&gl) {
                    self.best_path = Some(self.collect_locked());
                    return PropResult::Solved;
                }
            }

            if self.path_endpoints.len() == 2 {
                if !self.path_endpoints_connected() {
                    self.undo_to(entry_len);
                    return PropResult::Contradiction;
                }
            }
        }

        PropResult::Continue
    }

    // ===================== BRANCH VARIABLE SELECTION (MRV) =====================

    fn select_branch_edge(&self) -> Option<usize> {
        let branch_node = (0..self.n)
            .filter(|&v| self.locked_degree[v] < 2 && self.avail_deg[v] >= 2)
            .min_by_key(|&v| (self.avail_deg[v] * 4).saturating_sub(self.locked_degree[v]))?;

        let best_id = self.node_edges[branch_node]
            .iter()
            .copied()
            .filter(|&id| self.is_active(id))
            .max_by_key(|&id| {
                let Edge(u, v) = self.all_edges[id];
                let other = if u == branch_node { v } else { u };
                self.locked_degree[other] * 100 + 50usize.saturating_sub(self.avail_deg[other])
            })?;

        Some(best_id)
    }

    // ===================== MAIN SEARCH =====================

    fn _search(&mut self) -> bool {
        if self.is_seen() {
            return false;
        }

        let entry_len = self.undo_stack.len();

        match self.do_forced_propagation(entry_len) {
            PropResult::Contradiction => {
                self.memoize();
                return false;
            }
            PropResult::Solved => {
                return true;
            }
            PropResult::Continue => {}
        }

        if self.locked_degree.iter().all(|&d| d == 2) {
            let gl = self.build_locked_graph();
            if self.is_connected_iter(&gl) && gl.iter().all(|a| a.len() == 2) {
                self.best_path = Some(self.collect_locked());
                return true;
            }
            self.undo_to(entry_len);
            self.memoize();
            return false;
        }

        let branch_id = match self.select_branch_edge() {
            Some(id) => id,
            None => {
                self.undo_to(entry_len);
                self.memoize();
                return false;
            }
        };

        self.apply_lock(branch_id);
        if self._search() {
            return true;
        }
        self.undo();

        self.apply_delete(branch_id);
        if self._search() {
            return true;
        }
        self.undo();

        self.undo_to(entry_len);
        self.memoize();
        false
    }

    // ===================== PUBLIC SOLVE (with timeout) =====================

    fn solve(&mut self) -> Option<Vec<Edge>> {
        if self.is_bipartite() {
            let color = self.get_color();
            let even = color.iter().filter(|&&c| c == 0).count();
            if even != self.n - even {
                return None;
            }
        }
        if self.has_bridges_check() {
            return None;
        }
        self.reset_state();
        if self._search() {
            self.best_path.clone()
        } else {
            None
        }
    }

    fn solve_with_timeout(&mut self, timeout_secs: f64) -> SolveResult {
        let start = Instant::now();
        if self.is_bipartite() {
            let color = self.get_color();
            let even = color.iter().filter(|&&c| c == 0).count();
            if even != self.n - even {
                return SolveResult::Unsat;
            }
        }
        if self.has_bridges_check() {
            return SolveResult::Unsat;
        }
        self.reset_state();
        if self._search_timeout(&start, timeout_secs) {
            if let Some(path) = &self.best_path {
                SolveResult::Sat(path.clone())
            } else {
                SolveResult::Unsat
            }
        } else if start.elapsed().as_secs_f64() >= timeout_secs {
            SolveResult::Timeout
        } else {
            SolveResult::Unsat
        }
    }

    fn _search_timeout(&mut self, start: &Instant, timeout_secs: f64) -> bool {
        if start.elapsed().as_secs_f64() >= timeout_secs {
            return false;
        }
        if self.is_seen() {
            return false;
        }

        let entry_len = self.undo_stack.len();

        match self.do_forced_propagation(entry_len) {
            PropResult::Contradiction => {
                self.memoize();
                return false;
            }
            PropResult::Solved => {
                return true;
            }
            PropResult::Continue => {}
        }

        if self.locked_degree.iter().all(|&d| d == 2) {
            let gl = self.build_locked_graph();
            if self.is_connected_iter(&gl) && gl.iter().all(|a| a.len() == 2) {
                self.best_path = Some(self.collect_locked());
                return true;
            }
            self.undo_to(entry_len);
            self.memoize();
            return false;
        }

        let branch_id = match self.select_branch_edge() {
            Some(id) => id,
            None => {
                self.undo_to(entry_len);
                self.memoize();
                return false;
            }
        };

        self.apply_lock(branch_id);
        if self._search_timeout(start, timeout_secs) {
            return true;
        }
        self.undo();

        if start.elapsed().as_secs_f64() >= timeout_secs {
            return false;
        }

        self.apply_delete(branch_id);
        if self._search_timeout(start, timeout_secs) {
            return true;
        }
        self.undo();

        self.undo_to(entry_len);
        self.memoize();
        false
    }

    // ===================== PARALLEL SOLVE =====================

    pub fn solve_parallel(&mut self, timeout_secs: f64) -> SolveResult {
        let start = Instant::now();

        if self.is_bipartite() {
            let color = self.get_color();
            let even = color.iter().filter(|&&c| c == 0).count();
            if even != self.n - even {
                return SolveResult::Unsat;
            }
        }
        if self.has_bridges_check() {
            return SolveResult::Unsat;
        }

        self.reset_state();

        let entry_len = 0;
        match self.do_forced_propagation(entry_len) {
            PropResult::Contradiction => {
                return SolveResult::Unsat;
            }
            PropResult::Solved => {
                return SolveResult::Sat(self.best_path.clone().unwrap_or_default());
            }
            PropResult::Continue => {}
        }

        let subtrees = self.enumerate_subtrees(SPLIT_DEPTH);

        if subtrees.is_empty() {
            return SolveResult::Unsat;
        }

        let found = Arc::new(AtomicBool::new(false));
        let timeout_secs_copy = timeout_secs;
        let start_copy = start;

        let results: Vec<Option<Vec<Edge>>> = subtrees
            .into_par_iter()
            .map(|mut subtree| {
                if found.load(Ordering::Relaxed) {
                    return None;
                }
                if start_copy.elapsed().as_secs_f64() >= timeout_secs_copy {
                    return None;
                }

                let found_clone = Arc::clone(&found);
                subtree.memo_cache.clear();
                if subtree._search_parallel(&found_clone, &start_copy, timeout_secs_copy) {
                    found.store(true, Ordering::Relaxed);
                    subtree.best_path.clone()
                } else {
                    None
                }
            })
            .collect();

        if start.elapsed().as_secs_f64() >= timeout_secs {
            if found.load(Ordering::Relaxed) {
                // Found before timeout — handled below
            } else {
                return SolveResult::Timeout;
            }
        }

        for r in results {
            if let Some(path) = r {
                return SolveResult::Sat(path);
            }
        }

        SolveResult::Unsat
    }

    fn enumerate_subtrees(&self, max_depth: usize) -> Vec<BcCraver> {
        let mut queue: Vec<(BcCraver, usize)> = vec![(self.clone(), 0)];
        let mut leaves: Vec<BcCraver> = vec![];

        while let Some((mut solver, depth)) = queue.pop() {
            if solver.is_seen() {
                continue;
            }

            let entry_len = solver.undo_stack.len();

            match solver.do_forced_propagation(entry_len) {
                PropResult::Contradiction => {
                    continue;
                }
                PropResult::Solved => {
                    leaves.push(solver);
                    continue;
                }
                PropResult::Continue => {}
            }

            if depth >= max_depth {
                leaves.push(solver);
                continue;
            }

            match solver.select_branch_edge() {
                None => {
                    continue;
                }
                Some(branch_id) => {
                    let mut lock_child = solver.clone();
                    lock_child.apply_lock(branch_id);
                    queue.push((lock_child, depth + 1));

                    let mut del_child = solver;
                    del_child.apply_delete(branch_id);
                    queue.push((del_child, depth + 1));
                }
            }
        }

        leaves
    }

    fn _search_parallel(&mut self, found: &AtomicBool, start: &Instant, timeout_secs: f64) -> bool {
        if found.load(Ordering::Relaxed) {
            return false;
        }
        if start.elapsed().as_secs_f64() >= timeout_secs {
            return false;
        }
        if self.is_seen() {
            return false;
        }

        let entry_len = self.undo_stack.len();

        match self.do_forced_propagation(entry_len) {
            PropResult::Contradiction => {
                self.memoize();
                return false;
            }
            PropResult::Solved => {
                return true;
            }
            PropResult::Continue => {}
        }

        if self.locked_degree.iter().all(|&d| d == 2) {
            let gl = self.build_locked_graph();
            if self.is_connected_iter(&gl) && gl.iter().all(|a| a.len() == 2) {
                self.best_path = Some(self.collect_locked());
                return true;
            }
            self.undo_to(entry_len);
            self.memoize();
            return false;
        }

        let branch_id = match self.select_branch_edge() {
            Some(id) => id,
            None => {
                self.undo_to(entry_len);
                self.memoize();
                return false;
            }
        };

        self.apply_lock(branch_id);
        if self._search_parallel(found, start, timeout_secs) {
            return true;
        }
        self.undo();

        if found.load(Ordering::Relaxed) {
            return false;
        }
        if start.elapsed().as_secs_f64() >= timeout_secs {
            return false;
        }

        self.apply_delete(branch_id);
        if self._search_parallel(found, start, timeout_secs) {
            return true;
        }
        self.undo();

        self.undo_to(entry_len);
        self.memoize();
        false
    }

    fn reset_state(&mut self) {
        self.locked_bits.fill(0);
        self.deleted_bits.fill(0);
        self.undo_stack.clear();
        self.locked_degree.fill(0);
        self.g_avail_bits.clone_from(&self.orig_bits);
        self.avail_deg.clone_from(&self.orig_deg);
        self.best_path = None;
        self.total_deletions = 0;
        self.memo_cache.clear();
        self.path_endpoints.clear();
    }

    // ===================== BIPARTITE =====================

    fn is_bipartite(&self) -> bool {
        let mut color = vec![-1i32; self.n];
        for i in 0..self.n {
            if color[i] == -1 && !self.bfs_color(i, &mut color) {
                return false;
            }
        }
        true
    }

    fn get_color(&self) -> Vec<i32> {
        let mut color = vec![-1i32; self.n];
        for i in 0..self.n {
            if color[i] == -1 {
                self.bfs_color(i, &mut color);
            }
        }
        color
    }

    fn bfs_color(&self, start: usize, color: &mut Vec<i32>) -> bool {
        let mut q = VecDeque::new();
        q.push_back(start);
        color[start] = 0;
        while let Some(u) = q.pop_front() {
            for &v in &self.g_orig[u] {
                if color[v] == -1 {
                    color[v] = 1 - color[u];
                    q.push_back(v);
                } else if color[v] == color[u] {
                    return false;
                }
            }
        }
        true
    }

    // ===================== BRIDGE CHECK (iterative Tarjan) =====================

    fn has_bridges_check(&self) -> bool {
        if self.n == 0 {
            return false;
        }
        let g = &self.g_orig;
        let mut disc = vec![-1i32; self.n];
        let mut low = vec![0i32; self.n];
        let mut parent = vec![-1i32; self.n];
        let mut timer = 0i32;

        for root in 0..self.n {
            if disc[root] != -1 {
                continue;
            }
            disc[root] = timer;
            low[root] = timer;
            timer += 1;
            let mut stack: Vec<(usize, usize)> = vec![(root, 0)];

            while let Some((u, ni)) = stack.last_mut() {
                let u = *u;
                if *ni < g[u].len() {
                    let v = g[u][*ni];
                    *ni += 1;
                    if disc[v] == -1 {
                        parent[v] = u as i32;
                        disc[v] = timer;
                        low[v] = timer;
                        timer += 1;
                        stack.push((v, 0));
                    } else if v as i32 != parent[u] {
                        low[u] = min(low[u], disc[v]);
                    }
                } else {
                    stack.pop();
                    if let Some(&(p, _)) = stack.last() {
                        if low[u] > disc[p] {
                            return true;
                        } // bridge found
                        low[p] = min(low[p], low[u]);
                    }
                }
            }
        }
        false
    }
}

// ===================== SOLVE RESULT =====================

#[derive(Debug)]
pub enum SolveResult {
    Sat(Vec<Edge>),
    Unsat,
    Timeout,
}

// ===================== VALIDATION =====================

fn validate_cycle(g: &[Vec<usize>], edges: &[Edge]) -> bool {
    let n = g.len();
    if edges.len() != n {
        return false;
    }
    let mut adj = vec![vec![]; n];
    for &Edge(u, v) in edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    if !adj.iter().all(|a| a.len() == 2) {
        return false;
    }
    // connectivity
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    while let Some(u) = stack.pop() {
        for &v in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
    visited.iter().all(|&x| x)
}

fn is_connected_free(g: &[Vec<usize>], n: usize) -> bool {
    if n == 0 {
        return true;
    }
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    while let Some(u) = stack.pop() {
        for &v in &g[u] {
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
    visited.iter().all(|&x| x)
}

// ───────────────────────────── PUBLIC API ─────────────────────────────

/// Solve HCP on `g` using parallel branch-and-bound. Returns the first
/// Hamiltonian cycle found (SAT), UNSAT, or TIMEOUT.
pub fn solve_graph_parallel(g: &[Vec<usize>], timeout_secs: f64) -> SolveResult {
    let mut solver = BcCraver::new(g);
    solver.solve_parallel(timeout_secs)
}

/// Run static BC preprocessing on `g` (no branching). Returns the reduced
/// adjacency list if an HC may still exist, or `None` if provably UNSAT.
pub fn preprocess_graph(g: &[Vec<usize>]) -> Option<Vec<Vec<usize>>> {
    let mut solver = BcCraver::new(g);
    solver.reset_state();
    match solver.do_forced_propagation(0) {
        PropResult::Contradiction => None,
        PropResult::Solved | PropResult::Continue => Some(solver.build_avail_adj()),
    }
}

/// Convert a set of Hamiltonian-cycle edges into a node-visit order
/// starting from node 0. Returns `None` if the edges don't form a valid HC.
pub fn cycle_order_from_edges(n: usize, edges: &[Edge]) -> Option<Vec<usize>> {
    if n == 0 || edges.len() != n {
        return None;
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &Edge(u, v) in edges {
        if u >= n || v >= n {
            return None;
        }
        adj[u].push(v);
        adj[v].push(u);
    }
    if !adj.iter().all(|a| a.len() == 2) {
        return None;
    }
    let mut order = Vec::with_capacity(n);
    let mut prev = usize::MAX;
    let mut cur = 0usize;
    loop {
        order.push(cur);
        let next = if adj[cur][0] != prev {
            adj[cur][0]
        } else {
            adj[cur][1]
        };
        if next == 0 {
            break;
        }
        if order.len() == n {
            return None;
        }
        prev = cur;
        cur = next;
    }
    if order.len() == n {
        Some(order)
    } else {
        None
    }
}

pub fn split_depth() -> usize {
    SPLIT_DEPTH
}
