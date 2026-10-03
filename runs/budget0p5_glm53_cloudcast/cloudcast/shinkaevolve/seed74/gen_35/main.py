# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List

REFINEMENT_PASSES = 3
REUSE_DISCOUNT = 0.0


def route_pass(src, dsts, G, num_partitions, used_edges, order, edge_dst_count=None):
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    new_used = set(used_edges)
    if edge_dst_count is None:
        edge_dst_count = {}
    # assume each destination's broadcast stream demands ~1 gbps per edge
    PER_DST_FLOW_GBPS = 1.0
    for dst in order:
        h_disc = G.copy()
        for s, t, data in h_disc.edges(data=True):
            c = data.get("cost")
            if c is None:
                c = 1e9
            if (s, t) in new_used:
                # hard feasibility filter: only keep the reuse discount while
                # aggregate per-destination demand fits the edge throughput
                tput = data.get("throughput")
                count = edge_dst_count.get((s, t), 0)
                if tput is None or count * PER_DST_FLOW_GBPS < tput:
                    c = REUSE_DISCOUNT
                # else: saturated trunk, keep original cost so this dst routes around
            h_disc[s][t]["cost"] = c
        try:
            path = nx.dijkstra_path(h_disc, src, dst, weight="cost")
        except nx.NetworkXPathNotFound:
            path = nx.dijkstra_path(G, src, dst, weight="cost")
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            new_used.add((s, t))
            edge_dst_count[(s, t)] = edge_dst_count.get((s, t), 0) + 1
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
    return bc_topology, new_used, edge_dst_count


def tree_cost(G, used_edges):
    total = 0.0
    for (s, t) in used_edges:
        c = G[s][t].get("cost")
        if c is not None:
            total += c
    return total


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    for s, t, data in h.edges(data=True):
        if data.get("cost") is None:
            h[s][t]["cost"] = 1e9

    # candidate orderings
    base_costs = {}
    for d in dsts:
        try:
            base_costs[d] = nx.shortest_path_length(h, src, d, weight="cost")
        except Exception:
            base_costs[d] = 1e9
    orders = [
        sorted(dsts, key=lambda d: base_costs[d]),
        sorted(dsts, key=lambda d: -base_costs[d]),
        list(dsts),
    ]

    best = None
    best_cost = float("inf")
    for order in orders:
        used = set()
        topo = None
        edge_dst_count = {}
        for _ in range(REFINEMENT_PASSES):
            topo, used, edge_dst_count = route_pass(
                src, dsts, h, num_partitions, used, order, edge_dst_count
            )
        cost = tree_cost(h, used)
        if cost < best_cost:
            best_cost = cost
            best = topo

    return best


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

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
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
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