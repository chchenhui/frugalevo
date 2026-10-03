# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def _w(u, v, d):
    """Safe edge weight: missing/None cost gets a large penalty."""
    c = d.get("cost")
    return c if c is not None else 1e9


def _emit(bc, dst, path, G, nparts):
    """Record one path for every partition of dst."""
    for i in range(len(path) - 1):
        s, t = path[i], path[i + 1]
        for j in range(nparts):
            bc.append_dst_partition_path(dst, j, [s, t, G[s][t]])


def _tree_cost(paths, h):
    """Total cost of the union of edges across all paths (edge reuse = free)."""
    used = set()
    total = 0.0
    for p in paths:
        for i in range(len(p) - 1):
            e = (p[i], p[i + 1])
            if e not in used:
                used.add(e)
                total += _w(e[0], e[1], h[e[0]][e[1]])
    return total


def search_algorithm(src, dsts, G, num_partitions):
    """Greedy marginal-cost Steiner broadcast with Steiner-MST fallback.

    Process destinations in order of increasing shortest-path cost; for each,
    examine several candidate simple paths and pick the one minimizing the
    cost of *new* edges only (edges already on the shared trunk are free).
    This greedily builds a shared delivery tree. Independently build a
    Kou-Markowsky Steiner tree and emit whichever yields a cheaper union
    of edges, so the result is never worse than either strategy alone.
    """
    if src not in G:
        return BroadCastTopology(src, dsts, num_partitions)
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    valid = [d for d in dict.fromkeys(dsts) if d in h]
    # shortest-path distances from src (also serves as fallback ordering)
    try:
        dist, spaths = nx.single_source_dijkstra(h, src, weight=_w)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist, spaths = {}, {}

    # --- Strategy A: greedy marginal-cost routing ---
    greedy_paths = {}
    used = set()
    for dst in sorted(valid, key=lambda d: dist.get(d, float("inf"))):
        best_path, best_marg = None, float("inf")
        try:
            gen = nx.shortest_simple_paths(h, src, dst, weight=_w)
            budget = dist.get(dst, float("inf")) * 2 + 1e9  # prune expensive detours
            for count, p in enumerate(gen):
                if count >= 60 or _tree_cost([p], h) > budget:
                    break
                marg = sum(_w(p[i], p[i + 1], h[p[i]][p[i + 1]])
                           for i in range(len(p) - 1)
                           if (p[i], p[i + 1]) not in used)
                if marg < best_marg:
                    best_path, best_marg = p, marg
                if marg == 0:
                    break  # fully served by existing trunk
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            best_path = spaths.get(dst)
        if best_path is None:
            best_path = spaths.get(dst)
        if best_path is None:
            continue
        greedy_paths[dst] = best_path
        for i in range(len(best_path) - 1):
            used.add((best_path[i], best_path[i + 1]))

    # --- Strategy B: Kou-Markowsky Steiner tree on metric closure ---
    steiner_paths = {}
    terminals = [src] + valid
    try:
        dist_paths = dict(nx.all_pairs_dijkstra(h, weight=_w))
        closure = nx.Graph()
        for i, a in enumerate(terminals):
            for b in terminals[i + 1:]:
                if b in dist_paths[a][0]:
                    closure.add_edge(a, b, weight=dist_paths[a][0][b])
        mst = nx.minimum_spanning_tree(closure, weight="weight")
        shared = nx.DiGraph()
        for a, b in mst.edges():
            pa = dist_paths[a][1].get(b)
            pb = dist_paths[b][1].get(a)
            if pa is not None and (pb is None or dist_paths[a][0][b] <= dist_paths[b][0][a]):
                path = pa
            elif pb is not None:
                path = pb
            else:
                continue
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                shared.add_edge(s, t, cost=_w(s, t, h[s][t]))
        for dst in valid:
            if dst in shared:
                try:
                    steiner_paths[dst] = nx.dijkstra_path(shared, src, dst, weight=_w)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        # Steiner tree must cover all destinations to be a valid candidate
        if set(steiner_paths) != set(greedy_paths):
            steiner_paths = {}
    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
        steiner_paths = {}

    # pick the cheaper complete strategy
    if steiner_paths and len(steiner_paths) == len(greedy_paths):
        cand = [greedy_paths, steiner_paths]
    else:
        cand = [greedy_paths]
    chosen = min(cand, key=lambda ps: _tree_cost(list(ps.values()), h))

    for dst in valid:
        path = chosen.get(dst) or spaths.get(dst)
        if path is None:
            continue
        _emit(bc_topology, dst, path, G, bc_topology.num_partitions)

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
