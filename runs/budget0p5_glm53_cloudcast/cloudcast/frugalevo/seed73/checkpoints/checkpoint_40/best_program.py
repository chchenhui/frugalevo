# EVOLVE-BLOCK-START
import networkx as nx
import json
import math
import os
import random
import pandas as pd
from itertools import islice
from typing import Dict, List


def exact_steiner_tree(Uw, terms):
    """Dreyfus-Wagner exact Steiner tree on undirected graph Uw (edge
    attr 'cost') over terminal list terms (terms[0] must be src).

    Subset DP: C[v][mask] = min cost of a tree connecting terminals in
    `mask` plus node v. Two-phase recurrence per mask: (1) merge submasks
    (3^k total work), (2) one min-plus edge relaxation pass using the
    *final* C[u][mask] of neighbors (single pass suffices since the edge
    term is min_u C[u][mask] + w(u,v)). Argmin decomposition choices are
    recorded in merge_split / epar and backtracked to recover the optimal
    tree edges. Returns an nx.Graph of the optimal tree, or None if some
    terminal is unreachable. O(3^k * n + 2^k * m).
    """
    nodes = list(Uw.nodes)
    k = len(terms)
    full = (1 << k) - 1
    INF = float("inf")
    C = {v: [INF] * (full + 1) for v in nodes}
    merge_split = {}
    epar = {}
    for i, t in enumerate(terms):
        if t in C:
            C[t][1 << i] = 0.0
    adj = {v: [(u, d["cost"]) for u, d in Uw[v].items()] for v in nodes}
    for mask in range(1, full + 1):
        for v in nodes:
            cv = C[v]
            sub = (mask - 1) & mask
            while sub:
                if cv[sub] + cv[mask ^ sub] < cv[mask]:
                    cv[mask] = cv[sub] + cv[mask ^ sub]
                    merge_split[(v, mask)] = sub
                    epar.pop((v, mask), None)
                sub = (sub - 1) & mask
            for u, w in adj[v]:
                if C[u][mask] + w < cv[mask]:
                    cv[mask] = C[u][mask] + w
                    epar[(v, mask)] = u
                    merge_split.pop((v, mask), None)
    best_v, best_c = None, INF
    for v in nodes:
        if C[v][full] < best_c:
            best_c, best_v = C[v][full], v
    if best_v is None or best_c == INF:
        return None
    tree_edges = set()

    def bt(v, mask):
        if (v, mask) in merge_split:
            s = merge_split[(v, mask)]
            bt(v, s)
            bt(v, mask ^ s)
        elif (v, mask) in epar:
            u = epar[(v, mask)]
            tree_edges.add((u, v))
            bt(u, mask)

    bt(best_v, full)
    T = nx.Graph()
    for u, v in tree_edges:
        T.add_edge(u, v, cost=Uw[u][v]["cost"])
    return T


def search_algorithm(src, dsts, G, num_partitions):
    """Shared-multicast Steiner-tree broadcast.

    Builds an undirected symmetrized graph (edge weight = min of the two
    directed costs), computes a Steiner tree over terminals {src} + dsts,
    orients it from src via BFS, and assigns each dst its unique tree path
    so shared prefixes carry data once instead of per-dst duplicates.
    Falls back to plain Dijkstra for dsts not covered by the tree.
    """
    from networkx.algorithms.approximation.steinertree import steiner_tree

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Symmetrize into an undirected graph with cheap edge weights.
    U = nx.Graph()
    for s, t, data in G.edges(data=True):
        if s == t or data.get("cost") is None:
            continue
        w = data["cost"]
        if U.has_edge(s, t):
            U[s][t]["cost"] = min(U[s][t]["cost"], w)
        else:
            U.add_edge(s, t, cost=w)

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def edge_data(s, t):
        if G.has_edge(s, t):
            return G[s][t]
        return G[t][s]

    def record(dst, path_nodes):
        for i in range(len(path_nodes) - 1):
            s, t = path_nodes[i], path_nodes[i + 1]
            for j in range(num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, edge_data(s, t)])

    terminals = {src} | {d for d in dsts if d != src and d in U}

    def dcost(s, t):
        d = edge_data(s, t)
        c = d.get("cost")
        return 0.0 if c is None else c

    def sol_cost(pmap):
        """Exact scored objective: sum of directed costs over the union of
        unique directed edges used by all per-dst paths (shared edges
        charged once, matching the multicast evaluator)."""
        used = set()
        for p in pmap.values():
            for i in range(len(p) - 1):
                used.add((p[i], p[i + 1]))
        return sum(dcost(s, t) for s, t in used)

    def build_solution(weight_mode):
        """Build a Steiner-tree broadcast plan under one symmetrization
        weight mode ('min' or 'sum' of the two directed edge costs), orient
        the tree from src, and polish it with bounded local search
        (re-attach, relay-prune, junction re-root) evaluated on the exact
        directed union-cost objective. Returns (paths_map, cost)."""
        Uw = nx.Graph()
        for s, t, data in G.edges(data=True):
            if s == t or data.get("cost") is None:
                continue
            w = data["cost"]
            if Uw.has_edge(s, t):
                Uw[s][t]["cost"] = (
                    min(Uw[s][t]["cost"], w) if weight_mode == "min"
                    else Uw[s][t]["cost"] + w
                )
            else:
                Uw.add_edge(s, t, cost=w)
        # Multi-seed construction + per-seed annealing on the exact directed
        # union-cost objective. Three structurally different seeds:
        #   A: exact/2-approx Steiner tree on the symmetrized undirected
        #      graph, oriented from src (maximal shared-prefix savings).
        #   B: directed Prim-Steiner growth on the original digraph — every
        #      tree edge is traversed in its true forward direction, so
        #      asymmetric per-direction prices are honored exactly (one
        #      cached reversed-graph Dijkstra per uncovered terminal; each
        #      round attaches the cheapest tree-node -> terminal path).
        #   C: union of independent per-dst Dijkstra paths (baseline).
        # Each seed is annealed SEPARATELY (distinct basins of attraction:
        # the reroute/splice moves cannot flip segment direction, so only a
        # directed seed can reach direction-aware optima) and the best
        # annealed result is returned, so output is never worse than the
        # incumbent single-seed pipeline. Budget: 3 seeds x 5000 iterations
        # of cheap sol_cost evaluations per weight mode, deterministic RNG.
        seeds = []
        if len(terminals) > 1:
            # --- Seed A: undirected Steiner tree oriented from src ---
            try:
                if nx.is_connected(Uw.subgraph(Uw.nodes)):
                    tl = [src] + [t for t in terminals if t != src]
                    if len(tl) <= 12:
                        T = exact_steiner_tree(Uw, tl)
                    else:
                        T = steiner_tree(Uw, tl, weight="cost")
                    if T is not None and src in T:
                        parent = {src: None}
                        order = [src]
                        for node in order:
                            for nbr in T[node]:
                                if nbr not in parent:
                                    parent[nbr] = node
                                    order.append(nbr)
                        amap = {}
                        for dst in dsts:
                            if dst != src and dst in parent:
                                pn = []
                                node = dst
                                while node is not None:
                                    pn.append(node)
                                    node = parent[node]
                                pn.reverse()
                                amap[dst] = pn
                        if amap:
                            seeds.append(amap)
            except Exception:
                pass

            # --- Seed B: directed Prim-Steiner growth on G ---
            try:
                Gc = nx.DiGraph(
                    (s, t, {"cost": d["cost"]}) for s, t, d in G.edges(data=True)
                    if s != t and d.get("cost") is not None
                )
                unc = [t for t in terminals if t != src and t in Gc]
                if unc:
                    Grev = Gc.reverse(copy=False)
                    tree_nodes = {src}
                    parent = {}
                    term_dist = {}
                    term_paths = {}
                    while unc:
                        best = None
                        for t in unc:
                            if t not in term_dist:
                                try:
                                    term_dist[t], term_paths[t] = nx.single_source_dijkstra(
                                        Grev, t, weight="cost")
                                except nx.NetworkXNoPath:
                                    term_dist[t] = {}
                            for v in tree_nodes:
                                dv = term_dist[t].get(v)
                                if dv is not None and (best is None or dv < best[0]):
                                    best = (dv, v, t)
                        if best is None:
                            break
                        _, v, t = best
                        seg = list(reversed(term_paths[t][v]))  # v -> t in G
                        for i in range(len(seg) - 1):
                            parent[seg[i + 1]] = seg[i]
                        tree_nodes.update(seg)
                        unc.remove(t)
                    bmap = {}
                    for dst in dsts:
                        if dst != src and dst in parent:
                            pn = []
                            node = dst
                            while node is not None:
                                pn.append(node)
                                node = parent.get(node)
                            if pn[-1] == src:
                                pn.reverse()
                                bmap[dst] = pn
                    if bmap:
                        seeds.append(bmap)
            except Exception:
                pass

            # --- Seed C: independent per-dst Dijkstra union ---
            try:
                cmap = {}
                for dst in dsts:
                    if dst == src or dst not in U:
                        continue
                    try:
                        cmap[dst] = nx.dijkstra_path(h, src, dst, weight="cost")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                if cmap:
                    seeds.append(cmap)
            except Exception:
                pass

        # Per-seed simulated-annealing polish on the exact directed
        # union-cost objective (uphill moves allowed under a geometric
        # temperature schedule). Moves: (a) reroute from the <=5 cheapest
        # simple paths; (b) splice at a shared node with another dst's
        # current path. Best-ever map per seed is kept; the overall best
        # across seeds is returned.
        pmap, base = {}, float("inf")
        for seed in seeds:
            rng = random.Random(12345)
            cand_paths = {}
            for d in seed:
                try:
                    cand_paths[d] = list(
                        islice(nx.shortest_simple_paths(h, src, d, weight="cost"), 5)
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    cand_paths[d] = [seed[d]]
            cur, cur_cost = dict(seed), sol_cost(seed)
            best_map, best_cost = dict(seed), cur_cost
            dst_list = list(cur)
            T0 = 0.5 * cur_cost / max(1, len(dst_list))
            T_end = 0.01
            iters = 5000
            for it in range(iters):
                T = T0 * (T_end / T0) ** (it / max(1, iters - 1)) if T0 > 0 else T_end
                dst = rng.choice(dst_list)
                cp = cur[dst]
                cands = [rng.choice(cand_paths[dst])]
                other = rng.choice(dst_list)
                op = cur[other]
                cp_set = set(cp)
                shared = [v for v in op if v in cp_set]
                if shared:
                    v = rng.choice(shared)
                    spliced = op[: op.index(v) + 1] + cp[cp.index(v) + 1:]
                    cands.append(spliced)
                p = rng.choice(cands)
                if not p or len(p) < 2 or p[0] != src or p[-1] != dst:
                    continue
                trial = dict(cur)
                trial[dst] = p
                c = sol_cost(trial)
                delta = c - cur_cost
                if delta <= 0 or (cur_cost > 0 and rng.random() < math.exp(-delta / max(T, 1e-12))):
                    cur, cur_cost = trial, c
                    if c < best_cost:
                        best_map, best_cost = dict(trial), c
            if best_cost < base:
                pmap, base = best_map, best_cost
        return pmap, base

    # Run the build+polish pipeline under both symmetrization metrics and
    # keep the cheaper plan (evaluated on the exact scored objective).
    best_map, best_cost = {}, float("inf")
    for mode in ("min", "sum"):
        try:
            pm, c = build_solution(mode)
            if c < best_cost:
                best_map, best_cost = pm, c
        except Exception:
            pass
    paths_map = best_map

    for dst, p in paths_map.items():
        record(dst, p)

    # Fallback: Dijkstra for any dst not covered by the polished tree.
    for dst in dsts:
        if dst == src or dst in paths_map:
            continue
        try:
            record(dst, nx.dijkstra_path(h, src, dst, weight="cost"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def get_paths(self):
        print(f"now the set path is: {self.paths}")
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst
        """
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Default graph with capacity constraints and cost info
    nodes: regions, edges: links
    per edge:
        throughput: max tput achievable (gbps)
        cost: $/GB
        flow: actual flow (gbps), must be < throughput, default = 0
    """
    # Use relative path from this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if cost_path is None:
        cost = pd.read_csv(os.path.join(current_dir, "profiles/cost.csv"))
    else:
        cost = pd.read_csv(cost_path)

    if throughput_path is None:
        throughput = pd.read_csv(os.path.join(current_dir, "profiles/throughput.csv"))
    else:
        throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(row["src_region"], row["dst_region"], cost=None, throughput=num_vms * row["throughput_sent"] / 1e9)

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    # some pairs not in the cost grid
    no_cost_pairs = []
    for edge in G.edges.data():
        src, dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src, dst))
    print("Unable to get costs for: ", no_cost_pairs)

    return G


# EVOLVE-BLOCK-END

# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance"""
    return BroadCastTopology(src, dsts, num_partitions)

def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology"""
    return search_algorithm(src, dsts, G, num_partitions)
