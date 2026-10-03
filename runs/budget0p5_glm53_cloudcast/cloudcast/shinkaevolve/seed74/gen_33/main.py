# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions, k_paths=4, discount=0.0, refine_passes=3):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Edge weights with reuse discounting: edges already paid for by a
    # previously routed partition are (nearly) free for subsequent ones.
    def build_discounted_graph(used_edges):
        hd = h.copy()
        for (s, t) in hd.edges():
            base = hd[s][t].get("cost") or 0.0
            if (s, t) in used_edges:
                base = discount * base
            hd[s][t]["cost"] = base
        return hd

    best_topology = None
    best_total = float("inf")
    used_edges = set()

    # Outer refinement loop: re-route with used_edges carried over so that
    # later passes can consolidate traffic onto shared, already-paid trunks.
    for _ in range(refine_passes):
        hd = build_discounted_graph(used_edges)
        bc_topology = BroadCastTopology(src, dsts, num_partitions)
        new_used = set()

        for dst in dsts:
            try:
                base_path = nx.dijkstra_path(hd, src, dst, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            # Every partition follows the cheapest (reuse-discounted) route;
            # routing all partitions the same way avoids paying for multiple
            # distinct trunks to reach the same destination.
            for j in range(num_partitions):
                try:
                    path = nx.dijkstra_path(hd, src, dst, weight="cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    path = base_path
                for i in range(0, len(path) - 1):
                    s, t = path[i], path[i + 1]
                    bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
                    new_used.add((s, t))
                # Update discounted graph so partitions after the first one
                # can exploit the edges this destination already paid for.
                for i in range(0, len(path) - 1):
                    s, t = path[i], path[i + 1]
                    hd[s][t]["cost"] = discount * (h[s][t].get("cost") or 0.0)

        # Evaluate this pass on undiscounted costs (unique edges counted once
        # per destination-partition route, plus discounted reuse).
        total = 0.0
        for dst in dsts:
            for j in range(num_partitions):
                p = bc_topology.paths[dst][str(j)]
                if not p:
                    continue
                for edge in p:
                    total += edge[2].get("cost") or 0.0

        if total < best_total:
            best_total = total
            best_topology = bc_topology

        if new_used == used_edges:
            # Converged: further passes will not change anything.
            break
        used_edges = new_used

    if best_topology is None:
        best_topology = BroadCastTopology(src, dsts, num_partitions)

    return best_topology


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