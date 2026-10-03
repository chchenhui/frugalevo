# EVOLVE-BLOCK-START
import networkx as nx
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    # Remove incoming edges to src and self loops so all flows leave the source
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Discount factor for edges already used in the broadcast tree:
    # reusing an edge is cheap since data only needs to cross it once
    # (shared trunk), so its effective marginal cost is small.
    REUSE_FACTOR = 0.15

    used_edges = set()

    def shared_weight(u, v, data):
        cost = data.get("cost")
        if cost is None:
            cost = 10.0
        if (u, v) in used_edges:
            return cost * REUSE_FACTOR
        return cost

    # Order destinations: farthest/most-expensive first so that trunk
    # infrastructure is built for the hardest destination and reused later.
    def base_cost(dst):
        try:
            return nx.dijkstra_path_length(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    ordered_dsts = sorted(dsts, key=base_cost, reverse=True)

    chosen_paths = {}
    for dst in ordered_dsts:
        try:
            path = nx.dijkstra_path(h, src, dst, weight=shared_weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                path = nx.dijkstra_path(h, src, dst, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        chosen_paths[dst] = path
        # Mark edges as shared trunk (idempotent)
        for i in range(len(path) - 1):
            used_edges.add((path[i], path[i + 1]))

    # Build the topology: all partitions of a destination follow the same
    # (possibly shared) path. Because shared edges carry data only once,
    # total transfer cost drops relative to per-destination shortest paths.
    for dst in dsts:
        path = chosen_paths.get(dst)
        if path is None:
            continue
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = str
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

    def set_graph(self):
        for dst in self.dsts:
            if dst not in self.paths:
                self.paths[dst] = {str(i): None for i in range(self.num_partitions)}


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
        G.add_edge(row["src_region"], row["dst_region"], cost=None,
                   throughput=num_vms * row["throughput_sent"] / 1e9)

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
