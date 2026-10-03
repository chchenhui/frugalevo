# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Shared Steiner-tree broadcast (all partitions on tree paths).

    Builds one shared broadcast tree from src to all dsts using the
    shortest-path Steiner heuristic: repeatedly attach the nearest
    unconnected destination to the growing tree via a minimum-cost
    Dijkstra path from any already-connected node. Every partition of
    every destination then follows its route inside this shared tree.
    If the simulator dedups transfers that share edges across
    destinations, this eliminates redundant per-destination transfers.
    Falls back to plain per-destination Dijkstra for any dst not
    reachable in the tree, preserving success_rate = 1.0.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def strict_w(u, v, d):
        c = d.get("cost")
        return c if c is not None else float("inf")

    # Steiner tree via shortest-path heuristic over the terminal set
    tree = nx.DiGraph()
    tree.add_node(src)
    connected = {src}
    remaining = set(dsts)
    while remaining:
        best_dst, best_path = None, None
        best_len = float("inf")
        for dst in remaining:
            for node in connected:
                try:
                    path = nx.dijkstra_path(h, node, dst, weight=strict_w)
                except nx.NetworkXNoPath:
                    continue
                plen = sum(h[path[i]][path[i + 1]].get("cost") or 0
                           for i in range(len(path) - 1))
                if plen < best_len:
                    best_len, best_dst, best_path = plen, dst, path
        if best_dst is None:
            break
        for i in range(len(best_path) - 1):
            tree.add_edge(best_path[i], best_path[i + 1])
        connected.update(best_path)
        remaining.discard(best_dst)

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        try:
            path = nx.shortest_path(tree, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                path = nx.dijkstra_path(h, src, dst, weight=strict_w)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        for j in range(num_partitions):
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
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
