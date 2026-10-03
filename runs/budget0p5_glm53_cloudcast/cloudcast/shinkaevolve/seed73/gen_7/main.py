# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Filter out any edges without valid cost so Dijkstra/k-shortest work reliably
    valid = nx.DiGraph()
    for u, v, data in h.edges(data=True):
        if data.get("cost") is not None:
            valid.add_edge(u, v, cost=data["cost"], **{k: v for k, v in data.items() if k != "cost"})

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # ---- Steiner tree via shortest-path heuristic ----
    # Build a multicast tree: data crosses each shared link once instead of
    # once per (dst, partition), eliminating redundant transfers.
    tree = nx.DiGraph()
    tree.add_node(src)
    connected = {src}
    # Distances from each node in the current tree to all other nodes
    dist_from_tree = dict(nx.single_source_dijkstra_path_length(valid, src, weight="cost"))
    path_from_tree = dict(nx.single_source_dijkstra_path(valid, src, weight="cost"))

    remaining = [d for d in dsts if d in valid and d != src]
    while remaining:
        # Connect the closest remaining destination to the tree
        best_dst = min(remaining, key=lambda d: dist_from_tree.get(d, float("inf")))
        path = path_from_tree[best_dst]
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not tree.has_edge(u, v):
                tree.add_edge(u, v, cost=valid[u][v]["cost"])
        remaining.remove(best_dst)
        connected.add(best_dst)
        # Recompute reach from the new tree leaves to remaining destinations
        new_leaves = [n for n in tree.nodes() if tree.out_degree(n) == 0]
        best_dist, best_path, best_src_node = float("inf"), None, None
        for leaf in new_leaves:
            if leaf not in valid:
                continue
            d_paths = nx.single_source_dijkstra_path(valid, leaf, weight="cost")
            d_lens = nx.single_source_dijkstra_path_length(valid, leaf, weight="cost")
            for d in remaining:
                if d in d_lens and d_lens[d] < best_dist:
                    best_dist = d_lens[d]
                    best_path = d_paths[d]
                    best_src_node = leaf
        if best_src_node is not None:
            dist_from_tree = {}
            path_from_tree = {}
            for d in remaining:
                if d in nx.single_source_dijkstra_path_length(valid, best_src_node, weight="cost"):
                    pass
            d_lens_all = nx.single_source_dijkstra_path_length(valid, best_src_node, weight="cost")
            d_paths_all = nx.single_source_dijkstra_path(valid, best_src_node, weight="cost")
            dist_from_tree = d_lens_all
            path_from_tree = d_paths_all
        else:
            dist_from_tree = {}
            path_from_tree = {}
            for d in remaining:
                try:
                    dist_from_tree[d], path_from_tree[d] = nx.single_source_dijkstra(valid, d, weight="cost"), None
                except Exception:
                    pass
            # fallback: recompute from every tree node
            dist_from_tree = {}
            path_from_tree = {}
            for tn in tree.nodes():
                if tn not in valid:
                    continue
                d_paths_all = nx.single_source_dijkstra_path(valid, tn, weight="cost")
                d_lens_all = nx.single_source_dijkstra_path_length(valid, tn, weight="cost")
                for d in remaining:
                    if d in d_lens_all and (d not in dist_from_tree or d_lens_all[d] < dist_from_tree[d]):
                        dist_from_tree[d] = d_lens_all[d]
                        path_from_tree[d] = d_paths_all[d]

    # Derive per-dst paths within the tree (cheapest shared structure)
    tree_undirected = tree.to_undirected()

    def tree_path_to(dst):
        try:
            und_path = nx.shortest_path(tree_undirected, src, dst)
        except nx.NetworkXNoPath:
            return None
        return und_path

    for dst in dsts:
        # Partition 0: use the shared Steiner tree path (no redundant edges)
        path = tree_path_to(dst)
        if path is None:
            # fallback to direct dijkstra
            path = nx.dijkstra_path(valid, src, dst, weight="cost")

        def emit(p):
            for i in range(len(p) - 1):
                s, t = p[i], p[i + 1]
                if G.has_edge(s, t):
                    bc_topology.append_dst_partition_path(dst, 0, [s, t, G[s][t]])
                else:
                    bc_topology.append_dst_partition_path(dst, 0, [s, t, G[t][s]])

        emit(path)

        # Partitions 1..k: diversify across alternative parallel paths to
        # balance load across networks while keeping cost low.
        try:
            alts = list(
                nx.shortest_simple_paths(valid, src, dst, weight="cost")
            )
        except nx.NetworkXNoPath:
            alts = [path]
        alts = [p for p in alts if p != path][: num_partitions - 1]
        for j in range(1, num_partitions):
            if j - 1 < len(alts):
                p = alts[j - 1]
            else:
                p = path  # reuse cheapest if no alternatives
            for i in range(len(p) - 1):
                s, t = p[i], p[i + 1]
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

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