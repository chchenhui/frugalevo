# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions, k_paths=4, discount=0.0, num_passes=3):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def true_cost(edges):
        return sum(G[s][t].get("cost") or 0.0 for s, t in edges)

    def route_pass(used_edges):
        """Route every (dst, partition) greedily on a graph where edges
        already carrying traffic are (nearly) free, encouraging shared trunks."""
        # Working graph with discounted edge costs.
        w = h.copy()
        for s, t in w.edges():
            c = w[s][t].get("cost") or 0.0
            if (s, t) in used_edges:
                w[s][t]["cost"] = c * discount
            else:
                w[s][t]["cost"] = c + 1e-9  # break ties deterministically

        topo = BroadCastTopology(src, dsts, num_partitions)
        pass_used = set(used_edges)
        for dst in dsts:
            for j in range(num_partitions):
                try:
                    path = nx.dijkstra_path(w, src, dst, weight="cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                for i in range(0, len(path) - 1):
                    s, t = path[i], path[i + 1]
                    topo.append_dst_partition_path(dst, j, [s, t, G[s][t]])
                    pass_used.add((s, t))
        return topo, pass_used

    best_topo = None
    best_edges = None
    best_cost = float("inf")
    used = set()
    for _ in range(num_passes):
        topo, used = route_pass(used)
        cost = true_cost(used)
        if cost < best_cost:
            best_cost = cost
            best_topo = topo
            best_edges = used
        else:
            break  # converged; no improvement from further passes

    if best_topo is None:
        best_topo = BroadCastTopology(src, dsts, num_partitions)
    return best_topo


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