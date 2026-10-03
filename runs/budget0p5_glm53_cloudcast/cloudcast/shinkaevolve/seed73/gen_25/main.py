# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    # Parameters
    REUSE_EPSILON = 1e-9    # near-zero marginal cost on already-paid tree edges
    CONGESTION_WEIGHT = 0.05  # mild penalty to spread partitions across parallel links

    # Sanitized copy of G: no in-edges of src, no self loops (read-only).
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Working copy with mutable costs: used for greedy Steiner-tree routing.
    # Edges already paid for by any destination become nearly free, so later
    # destinations and partitions reuse the shared broadcast tree instead of
    # paying for redundant parallel transfers.
    w = h.copy()
    for u, v in w.edges():
        c = w[u][v].get("cost")
        w[u][v]["cost"] = c if c is not None else 1e9

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Per-partition load per (dst, edge) to spread partitions over parallel paths.
    edge_load = {}
    paid_edges = set()

    for dst in dsts:
        if dst == src or dst not in h:
            continue
        for j in range(num_partitions):
            # Apply congestion penalty on top of current (reuse-discounted) costs.
            for (u, v), load in edge_load.items():
                if w.has_edge(u, v):
                    w[u][v]["cost"] += CONGESTION_WEIGHT * load

            try:
                path = nx.dijkstra_path(w, src, dst, weight="cost")
            except (nx.NetworkXNoPath, nx.NetworkXError, nx.NodeNotFound):
                # Retry on the pristine sanitized graph before giving up.
                try:
                    path = nx.dijkstra_path(h, src, dst, weight="cost")
                except (nx.NetworkXNoPath, nx.NetworkXError, nx.NodeNotFound):
                    continue

            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                key = (s, t)
                edge_load[key] = edge_load.get(key, 0) + 1
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
                if key not in paid_edges:
                    paid_edges.add(key)
                    # Edge is now part of the shared tree: marginal cost ~ 0.
                    if w.has_edge(s, t):
                        w[s][t]["cost"] = REUSE_EPSILON

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