# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import math
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Construct a low-cost shared directed broadcast topology.

    Candidate multicast trees are grown using marginal edge costs.  Links
    already present in a candidate tree have zero incremental cost, so later
    destinations preferentially reuse transferred data and branch only where
    needed.  Several seed/order strategies are evaluated to avoid dependence
    on one greedy attachment order.
    """
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Preserve destination order in the returned topology while removing
    # duplicates from the optimization problem.
    unique_dsts = list(dict.fromkeys(dsts))
    if not unique_dsts:
        return bc_topology

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    h.remove_edges_from(
        [
            (u, v)
            for u, v, data in h.edges(data=True)
            if data.get("cost") is None
            or not isinstance(data.get("cost"), (int, float))
            or not math.isfinite(float(data["cost"]))
        ]
    )

    missing = [dst for dst in unique_dsts if dst != src and dst not in h]
    if missing:
        raise nx.NetworkXNoPath(f"No broadcast path exists to {missing}")

    def edge_cost(u, v, data):
        return float(data["cost"])

    def tree_cost(tree):
        return sum(edge_cost(u, v, data) for u, v, data in tree.edges(data=True))

    def add_path(tree, path):
        for u, v in zip(path, path[1:]):
            tree.add_edge(u, v, **dict(G[u][v]))

    def marginal_weight(tree):
        def weight(u, v, data):
            return 0.0 if tree.has_edge(u, v) else float(data["cost"])
        return weight

    def grow_tree(seed_order=None, choose_cheapest=True):
        tree = nx.DiGraph()
        tree.add_node(src)
        remaining = set(unique_dsts)
        remaining.discard(src)

        if seed_order:
            for dst in seed_order:
                if dst not in remaining:
                    continue
                try:
                    path = nx.dijkstra_path(h, src, dst, weight="cost")
                except nx.NetworkXNoPath:
                    return None
                add_path(tree, path)
                remaining.remove(dst)

        while remaining:
            weight = marginal_weight(tree)
            candidates = []
            for dst in remaining:
                try:
                    path = nx.dijkstra_path(h, src, dst, weight=weight)
                    incremental = sum(
                        0.0 if tree.has_edge(u, v) else float(h[u][v]["cost"])
                        for u, v in zip(path, path[1:])
                    )
                    candidates.append((incremental, str(dst), dst, path))
                except nx.NetworkXNoPath:
                    continue

            if not candidates:
                raise nx.NetworkXNoPath(
                    f"No broadcast path exists from {src} to {sorted(remaining)}"
                )

            if choose_cheapest:
                _, _, dst, path = min(candidates)
            else:
                by_dst = {dst: path for _, _, dst, path in candidates}
                dst = next(node for node in unique_dsts if node in by_dst)
                path = by_dst[dst]

            add_path(tree, path)
            remaining.remove(dst)

        return tree

    candidates = []

    # Baseline union of independent least-cost routes.
    direct_tree = nx.DiGraph()
    direct_tree.add_node(src)
    try:
        for dst in unique_dsts:
            if dst != src:
                add_path(direct_tree, nx.dijkstra_path(h, src, dst, weight="cost"))
        candidates.append(direct_tree)
    except nx.NetworkXNoPath:
        pass

    # Marginal-cost growth with no seed, every possible seed, and stable
    # destination-order variants.  This is inexpensive for broadcast-sized
    # destination sets and materially improves Steiner-style sharing.
    candidate_specs = [(None, True), (unique_dsts, False), (list(reversed(unique_dsts)), False)]
    candidate_specs.extend(([dst], True) for dst in unique_dsts)

    for seed_order, choose_cheapest in candidate_specs:
        tree = grow_tree(seed_order, choose_cheapest)
        if tree is not None:
            candidates.append(tree)

    if not candidates:
        raise nx.NetworkXNoPath(f"No broadcast path exists from {src} to {unique_dsts}")

    best_tree = min(candidates, key=tree_cost)

    # A selected edge union can contain alternate routes.  Root-to-destination
    # least-cost paths within that union produce deterministic valid paths while
    # retaining the selected shared transfer links.
    for dst in dsts:
        if dst == src:
            edge_path = []
        else:
            try:
                path = nx.dijkstra_path(best_tree, src, dst, weight="cost")
            except nx.NetworkXNoPath:
                # This should only occur for an invalid external graph.
                path = nx.dijkstra_path(h, src, dst, weight="cost")
            edge_path = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]

        for partition in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(dst, partition, list(edge_path))

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src
        self.dsts = dsts
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
    """
    Default graph with capacity constraints and cost info
    nodes: regions, edges: links
    per edge:
        throughput: max tput achievable (gbps)
        cost: $/GB
        flow: actual flow (gbps), must be < throughput, default = 0
    """
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
        G.add_edge(
            row["src_region"],
            row["dst_region"],
            cost=None,
            throughput=num_vms * row["throughput_sent"] / 1e9,
        )

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