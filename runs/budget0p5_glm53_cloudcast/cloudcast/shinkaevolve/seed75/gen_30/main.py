# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Takahashi-Matsuyama Steiner tree heuristic:
    # grow a shared tree from src, attaching destinations one at a time
    # via cheapest connection to any existing tree node.
    tree_nodes = {src}
    tree_parent = {}  # node -> parent node in tree
    remaining = [d for d in dsts if d != src]

    while remaining:
        best_dst = None
        best_cost = float("inf")
        best_dist = None
        best_paths = None
        for d in remaining:
            try:
                dist, paths = nx.multi_source_dijkstra(h, tree_nodes, d, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if dist < best_cost:
                best_cost = dist
                best_dst = d
                best_paths = paths
        if best_dst is None:
            # unreachable destinations: fall back to any simple path attempt
            remaining = [d for d in remaining if d not in tree_nodes]
            break

        # Walk back from best_dst to the tree, recording parents
        node = best_dst
        while node not in tree_nodes:
            p = best_paths[node][-2]
            tree_parent[node] = p
            tree_nodes.add(node)
            node = p
        remaining.remove(best_dst)

    # Build full src->dst paths in the tree by following parents upward.
    for dst in dsts:
        if dst == src:
            continue
        if dst not in tree_nodes:
            continue
        # reconstruct path from dst up to src
        path = [dst]
        node = dst
        while node != src:
            node = tree_parent[node]
            path.append(node)
        path.reverse()
        for j in range(bc_topology.num_partitions):
            seen_edges = set()
            for i in range(0, len(path) - 1):
                s, t = path[i], path[i + 1]
                if (s, t) in seen_edges:
                    continue
                seen_edges.add((s, t))
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