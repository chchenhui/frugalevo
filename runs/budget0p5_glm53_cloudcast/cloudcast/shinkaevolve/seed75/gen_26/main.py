# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


# ---------------------------------------------------------------------------
# Stage 1: graph preparation
# ---------------------------------------------------------------------------
def prepare_graph(G: nx.DiGraph, src: str) -> nx.DiGraph:
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    return h


# ---------------------------------------------------------------------------
# Stage 2: Steiner-style multicast tree construction (built ONCE, shared)
# ---------------------------------------------------------------------------
def build_multicast_tree(h: nx.DiGraph, src: str, dsts: List[str]) -> Dict:
    """Return parent map {node: parent} rooted at src spanning all dsts."""
    rh = h.reverse(copy=True)
    tree_parent = {src: None}
    in_tree = {src}
    remaining = set(d for d in dsts if d in h)

    while remaining:
        best_dst, best_dist, best_path = None, float("inf"), None
        for d in remaining:
            if d not in rh:
                continue
            try:
                d_dist, d_paths = nx.single_source_dijkstra(rh, d, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            target, tdist = None, float("inf")
            for node in in_tree:
                if node in d_dist and d_dist[node] < tdist:
                    target, tdist = node, d_dist[node]
            if target is not None and tdist < best_dist:
                best_dst, best_dist, best_path = d, tdist, d_paths[target]

        if best_dst is None:
            # unreachable from tree; try direct src path, else skip
            d = remaining.pop()
            try:
                if nx.has_path(h, src, d):
                    p = nx.dijkstra_path(h, src, d, weight="cost")
                    for i in range(len(p) - 1):
                        if p[i + 1] not in in_tree and p[i + 1] != src:
                            tree_parent[p[i + 1]] = p[i]
                            in_tree.add(p[i + 1])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
            continue

        remaining.discard(best_dst)
        # best_path: d -> ... -> tree node. Attach chain into tree.
        for i in range(len(best_path) - 1):
            child, parent = best_path[i], best_path[i + 1]
            if child not in in_tree and child != src:
                tree_parent[child] = parent
                in_tree.add(child)
        in_tree.add(best_dst)
        if best_dst not in tree_parent and len(best_path) >= 2:
            tree_parent[best_dst] = best_path[1]

    return tree_parent


def dst_path_from_tree(tree_parent: Dict, src: str, dst: str):
    """Walk from dst up to src, returning ordered edge list [(s,t),...]."""
    chain = []
    cur = dst
    guard = 0
    while cur != src and guard <= len(tree_parent) + 1:
        par = tree_parent.get(cur)
        if par is None:
            return None  # dst not connected in tree
        chain.append((par, cur))
        cur = par
        guard += 1
    if cur != src:
        return None
    chain.reverse()
    return chain


# ---------------------------------------------------------------------------
# Stage 3: topology writer
# ---------------------------------------------------------------------------
def search_algorithm(src, dsts, G, num_partitions):
    h = prepare_graph(G, src)
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Build one shared multicast tree; every partition reuses it so data
    # traverses each tree edge once instead of once per (partition, dst).
    tree_parent = build_multicast_tree(h, src, dsts)

    # Precompute per-dst edge chains once (fallback to direct shortest path).
    dst_chains = {}
    for dst in dsts:
        chain = dst_path_from_tree(tree_parent, src, dst)
        if chain is None:
            try:
                if nx.has_path(h, src, dst):
                    p = nx.dijkstra_path(h, src, dst, weight="cost")
                    chain = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                chain = None
        dst_chains[dst] = chain

    # Write chains for every partition, reusing precomputed results.
    for dst, chain in dst_chains.items():
        if chain is None:
            continue
        edge_data = [[s, t, G[s][t]] for s, t in chain]
        for j in range(num_partitions):
            for edge in edge_data:
                bc_topology.append_dst_partition_path(dst, j, edge)

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