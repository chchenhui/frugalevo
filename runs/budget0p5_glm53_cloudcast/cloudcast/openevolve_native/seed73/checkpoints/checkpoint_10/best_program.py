# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Shared multicast Steiner-tree routing (shortest-path heuristic).

    Builds an approximate minimum Steiner tree from src to all dsts by
    repeatedly attaching the cheapest (via Dijkstra) unconnected
    destination to the current tree. Each destination's path is then the
    tree path from src. Because shared tree edges are reused across
    destinations, redundant transfers across common trunk links are
    eliminated, reducing total broadcast cost compared to independent
    per-destination shortest paths.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Shortest-path Steiner-tree heuristic:
    # tree_nodes: set of nodes already in the multicast tree
    # tree_pred: predecessor map defining tree edges (child -> parent)
    tree_nodes = {src}
    tree_pred = {}

    remaining = set(dsts)
    while remaining:
        # Multi-source Dijkstra from current tree to all nodes
        dist, paths = nx.multi_source_dijkstra(h, list(tree_nodes), weight="cost")
        # Pick the closest remaining destination
        best_dst = min(remaining, key=lambda d: dist[d])
        path = paths[best_dst]  # path from some tree node to best_dst
        # Attach the path suffix (beyond the first tree node) to the tree
        for i in range(len(path) - 1):
            child = path[i + 1]
            parent = path[i]
            if child not in tree_pred:
                tree_pred[child] = parent
            tree_nodes.add(child)
        remaining.discard(best_dst)

    # Reconstruct each destination's path from src within the tree
    def tree_path_to(dst):
        # Walk up the tree from dst to src, then reverse
        node = dst
        rev = []
        while node != src:
            parent = tree_pred[node]
            rev.append((parent, node))
            node = parent
        return list(reversed(rev))

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        edges = tree_path_to(dst)
        for (s, t) in edges:
            for j in range(bc_topology.num_partitions):
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
