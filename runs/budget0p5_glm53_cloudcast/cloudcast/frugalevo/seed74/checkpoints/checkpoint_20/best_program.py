# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Approach: global Steiner-tree multicast. Build a collapsed undirected view S
    of G where each pair's weight is the min cost over parallel directed edges
    (None-cost edges treated as free, matching make_nx_graph's unpriced pairs).
    Run steiner_tree(S, [src]+dsts) so expensive trunk edges near the source are
    shared across destinations instead of paid once per dst (per partition).
    Expand each destination's unique tree path from src into directed hops using
    the cheapest real directed edge of G for each consecutive pair, appending
    [s, t, edge_data] per partition exactly like the incumbent. Per-dst Dijkstra
    fallback preserves validity if the tree route is incomplete.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    def dijkstra_route(dst):
        """Incumbent per-dst Dijkstra fallback; appends and returns True on success."""
        try:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NetworkXException):
            return False
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
        return True

    # Directed view with None costs treated as free (used for all routing
    # decisions and cost simulation).
    W = nx.DiGraph()
    for u, v, data in h.edges(data=True):
        c = data.get("cost")
        W.add_edge(u, v, cost=0.0 if c is None else c)

    inf = float("inf")

    def sp_cost(paths):
        """Simulated cost: each distinct directed edge in the union of paths
        is charged exactly once (preserves the sharing benefit)."""
        seen = set()
        total = 0.0
        for p in paths:
            for i in range(len(p) - 1):
                e = (p[i], p[i + 1])
                if e not in seen:
                    seen.add(e)
                    total += W[p[i]][p[i + 1]]["cost"]
        return total

    def splice(a, b):
        """Concatenate two node paths sharing their junction node."""
        return a + b[1:]

    reachable = [d for d in dsts if d in W and nx.has_path(W, src, d)]
    if not reachable:
        for dst in dsts:
            dijkstra_route(dst)
        return bc_topology

    # ---- Candidate A: incumbent collapsed-undirected Steiner tree ----
    S = nx.Graph()
    for u, v, data in h.edges(data=True):
        c = data.get("cost")
        c = 0.0 if c is None else c
        if not S.has_edge(u, v) or c < S[u][v]["cost"]:
            S.add_edge(u, v, cost=c)

    # ---- Candidate A: perturbed-Steiner multistart ----
    # steiner_tree is a deterministic 2-approximation whose output depends on
    # tie-breaking among near-equal weights; a single call sees one point of
    # that landscape. Run R=8 additional solves on deterministically perturbed
    # weight views S_r (each edge cost scaled by 1 + eps_r * phi(u,v), phi a
    # fixed hash of the node pair), expand each tree to directed per-dst
    # routes, verify hop validity against h, and score with true costs via
    # sp_cost. Keep the best-valid candidate (eps=0 run included), so the
    # result is <= the single-solve incumbent by construction.
    def expand_tree(tree):
        """Expand a tree into per-dst src paths; verify directed hops; return
        (routes, union cost) or (None, inf) if any route is invalid."""
        tree_paths = {src: [src]}
        queue = [src]
        while queue:
            u = queue.pop(0)
            for w in tree.neighbors(u):
                if w not in tree_paths:
                    tree_paths[w] = tree_paths[u] + [w]
                    queue.append(w)
        routes = {}
        for dst in reachable:
            if dst not in tree_paths:
                return None, inf
            tpath = tree_paths[dst]
            if not all(
                h.has_edge(tpath[i], tpath[i + 1])
                for i in range(len(tpath) - 1)
            ):
                return None, inf
            routes[dst] = tpath
        return routes, sp_cost(list(routes.values()))

    def pair_pert(u, v, salt):
        """Deterministic per-(pair, salt) perturbation factor phi in [-1, 1].

        A polynomial rolling hash over the sorted node names plus a salt gives
        a fine-grained, reproducible factor (7 coarse buckets in the parent
        caused many restarts to reproduce the identical tree)."""
        a, b = sorted((str(u), str(v)))
        hv = 2166136261
        for ch in (a + "|" + b + "|" + str(salt)):
            hv = ((hv ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        return (hv % 2001 - 1000) / 1000.0

    tree_routes = None
    tree_cost = inf
    if len(reachable) > 1:
        for r in range(13):  # r=0 is the unperturbed incumbent solve
            eps = 0.01 * r
            Sr = nx.Graph()
            for u, v, data in S.edges(data=True):
                Sr.add_edge(u, v, cost=data["cost"] * (1.0 + eps * pair_pert(u, v, r)))
            try:
                tree = nx.algorithms.approximation.steinertree.steiner_tree(
                    Sr, [src] + reachable, weight="cost"
                )
            except (nx.NetworkXException, ValueError):
                continue
            routes, c = expand_tree(tree)
            if routes is not None and c < tree_cost:
                tree_cost = c
                tree_routes = routes

    # ---- Candidate B: hub-replication two-stage trunk ----
    # Rank dsts by total directed distance to all other dsts; take the most
    # central 3; collect nodes on their shortest src->dst paths as hubs.
    def centrality(d):
        tot = 0.0
        for o in reachable:
            if o == d:
                continue
            try:
                tot += nx.dijkstra_path_length(W, d, o)
            except nx.NetworkXException:
                return inf
        return tot

    # Precompute shortest paths between all relevant pairs ONCE. Hub evaluation
    # then costs only path concatenation, which makes exhaustive single-hub
    # (every node) and two-hub chain (src->h1->h2->dsts) search affordable —
    # strictly dominating the old top-3-centrality heuristic.
    nodes = [n for n in W.nodes if n != src]
    pair_path = {}
    for a in [src] + nodes:
        for b in nodes:
            if a == b:
                continue
            try:
                if nx.has_path(W, a, b):
                    pair_path[(a, b)] = nx.dijkstra_path(W, a, b)
            except nx.NetworkXException:
                pass

    hub_routes = None
    hub_cost = inf

    def eval_hubs(chain):
        """Evaluate trunk src->chain[0]->chain[1]->...->dsts using precomputed
        pair paths. Returns (full per-dst paths, union cost) or (None, inf)."""
        in_path = [src]
        for h in chain:
            if (in_path[-1], h) not in pair_path:
                return None, inf
            in_path = splice(in_path, pair_path[(in_path[-1], h)])
        out_paths = {}
        for d in reachable:
            if (in_path[-1], d) not in pair_path:
                return None, inf
            out_paths[d] = splice(in_path, pair_path[(in_path[-1], d)])
        return out_paths, sp_cost([in_path] + list(out_paths.values()))

    # Single-hub: every node. Two-hub chains: bounded to keep compute small
    # (first-stage hubs limited to the 25 most central nodes if graph is big).
    two_stage = nodes
    if len(nodes) > 25:
        two_stage = sorted(nodes, key=centrality)[:25]
    for u in nodes:
        out_paths, c = eval_hubs([u])
        if out_paths is not None and c < hub_cost:
            hub_cost = c
            hub_routes = out_paths
    for u in two_stage:
        for v in nodes:
            if u == v:
                continue
            out_paths, c = eval_hubs([u, v])
            if out_paths is not None and c < hub_cost:
                hub_cost = c
                hub_routes = out_paths

    # Commit the cheaper structure (argmin guarantees no regression vs the
    # simulated incumbent-tree cost).
    if hub_routes is not None and hub_cost <= tree_cost:
        routes = hub_routes
    elif tree_routes is not None:
        routes = tree_routes
    else:
        routes = None

    if routes is None:
        for dst in dsts:
            dijkstra_route(dst)
        return bc_topology

    # ---- Local search: marginal-cost path substitution on shared ledger ----
    # Each sweep rebuilds the committed edge set E from the surviving paths,
    # copies W with all E-edges priced 0 (marginal view), and re-runs per-dst
    # Dijkstra on it. A candidate is adopted only if the TOTAL union cost of
    # all paths strictly decreases, so improvement is monotone and can never
    # regress below the committed structure. Rebuilding E each sweep means any
    # edge orphaned by a substitution automatically loses its free status
    # (leaf pruning of non-terminal detours). Bounded to <= 3 sweeps.
    paths = {d: list(routes[d]) for d in routes}
    cur_cost = sp_cost(list(paths.values()))
    for _sweep in range(3):
        E = set()
        for p in paths.values():
            for i in range(len(p) - 1):
                E.add((p[i], p[i + 1]))
        M = nx.DiGraph()
        for u, v, data in W.edges(data=True):
            M.add_edge(u, v, cost=0.0 if (u, v) in E else data["cost"])
        improved = False
        for dst in list(paths):
            try:
                cand = nx.dijkstra_path(M, src, dst, weight="cost")
            except nx.NetworkXException:
                continue
            if cand == paths[dst]:
                continue
            trial = dict(paths)
            trial[dst] = cand
            tc = sp_cost(list(trial.values()))
            if tc < cur_cost - 1e-12:
                paths = trial
                cur_cost = tc
                improved = True
        if not improved:
            break
    routes = paths

    for dst in dsts:
        if dst in routes:
            tpath = routes[dst]
            for i in range(len(tpath) - 1):
                s, t = tpath[i], tpath[i + 1]
                for j in range(bc_topology.num_partitions):
                    bc_topology.append_dst_partition_path(dst, j, [s, t, h[s][t]])
        else:
            dijkstra_route(dst)

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
