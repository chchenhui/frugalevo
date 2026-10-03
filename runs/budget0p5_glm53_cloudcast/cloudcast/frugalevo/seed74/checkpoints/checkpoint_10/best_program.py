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

    tree_routes = None
    tree_cost = inf
    if len(reachable) > 1:
        try:
            tree = nx.algorithms.approximation.steinertree.steiner_tree(
                S, [src] + reachable, weight="cost"
            )
        except (nx.NetworkXException, ValueError):
            tree = None
        if tree is not None:
            tree_paths = {src: [src]}
            queue = [src]
            while queue:
                u = queue.pop(0)
                for w in tree.neighbors(u):
                    if w not in tree_paths:
                        tree_paths[w] = tree_paths[u] + [w]
                        queue.append(w)
            routes = {}
            ok = True
            for dst in reachable:
                if dst not in tree_paths:
                    ok = False
                    break
                tpath = tree_paths[dst]
                if not all(
                    h.has_edge(tpath[i], tpath[i + 1])
                    for i in range(len(tpath) - 1)
                ):
                    ok = False
                    break
                routes[dst] = tpath
            if ok:
                tree_routes = routes
                tree_cost = sp_cost(list(routes.values()))

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

    hub_cands = set()
    for d in sorted(reachable, key=centrality)[:3]:
        try:
            hub_cands.update(nx.dijkstra_path(W, src, d))
        except nx.NetworkXException:
            continue
    hub_cands.discard(src)
    hub_cands = list(hub_cands)[:10]

    hub_routes = None
    hub_cost = inf
    for u in hub_cands:
        try:
            if not nx.has_path(W, src, u):
                continue
            in_path = nx.dijkstra_path(W, src, u)
            out_paths = {}
            bad = False
            for d in reachable:
                if not nx.has_path(W, u, d):
                    bad = True
                    break
                out_paths[d] = nx.dijkstra_path(W, u, d)
            if bad:
                continue
        except nx.NetworkXException:
            continue
        c = sp_cost([in_path] + list(out_paths.values()))
        if c < hub_cost:
            hub_cost = c
            hub_routes = {d: splice(in_path, out_paths[d]) for d in reachable}

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
