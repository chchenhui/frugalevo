# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
import random
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Backfill missing costs with a large-but-finite penalty so pathfinding
    # never crashes on None weights while still avoiding unknown-cost links.
    for u, v, data in h.edges(data=True):
        if data.get("cost") is None:
            data["cost"] = 1e6
        if data.get("throughput") is None:
            data["throughput"] = 1.0

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # --- Stage 1: collect reachable destinations and their baseline costs ---
    reachable = []
    baseline = {}
    for dst in dsts:
        if not nx.has_path(h, src, dst):
            print(f"Destination {dst} is unreachable from {src}")
            continue
        try:
            dist, _ = nx.single_source_dijkstra(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        baseline[dst] = dist
        reachable.append(dst)

    # Process cheaper destinations first: grows the shared backbone where it
    # is cheapest, so expensive far destinations can hook into existing edges.
    reachable.sort(key=lambda d: baseline[d])

    # --- Stage 2: greedy Steiner-tree broadcast with discounted tree reuse ---
    tree_edges = set()
    for dst in reachable:
        def edge_weight(u, v, d):
            c = d["cost"]
            if (u, v) in tree_edges:
                c *= 0.01  # reuse of broadcast backbone is nearly free
            return c

        best_path = nx.dijkstra_path(h, src, dst, weight=edge_weight)
        best_cost = sum(h[best_path[i]][best_path[i + 1]]["cost"]
                        for i in range(len(best_path) - 1)
                        if (best_path[i], best_path[i + 1]) not in tree_edges)

        # Perturbed variants may find a cheaper way to hook into the tree
        rng = random.Random(hash((src, dst)) & 0xFFFFFFFF)
        for _ in range(5):
            ph = h.copy()
            for _, _, d in ph.edges(data=True):
                d["_w"] = d["cost"] * (0.5 + rng.random() * 1.5)
            try:
                p = nx.dijkstra_path(ph, src, dst, weight="_w")
            except nx.NetworkXNoPath:
                continue
            c = sum(h[p[i]][p[i + 1]]["cost"]
                    for i in range(len(p) - 1)
                    if (p[i], p[i + 1]) not in tree_edges)
            if c < best_cost:
                best_path, best_cost = p, c

        # Grow the shared broadcast tree with the chosen path
        for i in range(0, len(best_path) - 1):
            s, t = best_path[i], best_path[i + 1]
            tree_edges.add((s, t))
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