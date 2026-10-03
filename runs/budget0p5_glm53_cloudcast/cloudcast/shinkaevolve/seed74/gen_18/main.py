# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    # drop edges with no cost info so weights are well-defined
    for u, v, d in list(h.edges(data=True)):
        if d.get("cost") is None:
            h.remove_edge(u, v)

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    REUSE_BONUS = 0.9          # credit for reusing an edge already picked this partition
    MAX_CANDIDATES = num_partitions + 2
    DIVERSIFY_TOL = 0.15       # allow slightly costlier path if it balances load

    # --- gather candidate paths per destination ---
    dst_candidates = {}
    dst_base_cost = {}
    for dst in dsts:
        cands = []
        try:
            for p in nx.shortest_simple_paths(h, src, dst, weight="cost"):
                cands.append(p)
                if len(cands) >= MAX_CANDIDATES:
                    break
        except nx.NetworkXNoPath:
            try:
                cands = [nx.dijkstra_path(h, src, dst, weight="cost")]
            except nx.NetworkXNoPath:
                cands = []
        if not cands:
            continue
        dst_candidates[dst] = cands
        dst_base_cost[dst] = sum(
            h[a][b]["cost"] for a, b in zip(cands[0][:-1], cands[0][1:])
        )

    # --- process destinations in decreasing base-cost order ---
    ordered = sorted(dst_candidates, key=lambda d: dst_base_cost[d], reverse=True)

    # partition_edges[j] = set of edges already chosen for partition j
    partition_edges = {j: set() for j in range(num_partitions)}
    # edge_use_count for diversification tie-breaks
    global_edge_count = {}

    def raw_cost(path):
        return sum(h[a][b]["cost"] for a, b in zip(path[:-1], path[1:]))

    for dst in ordered:
        cands = dst_candidates[dst]
        chosen = []
        for j in range(num_partitions):
            used = partition_edges[j]
            best_path, best_score = None, None
            for p in cands:
                rc = raw_cost(p)
                reused = sum(
                    1 for a, b in zip(p[:-1], p[1:]) if (a, b) in used
                )
                # small penalty for globally congested edges (tie-break only)
                glob = sum(global_edge_count.get((a, b), 0)
                           for a, b in zip(p[:-1], p[1:]))
                score = rc - REUSE_BONUS * sum(
                    h[a][b]["cost"] for a, b in zip(p[:-1], p[1:]) if (a, b) in used
                ) + 0.01 * glob
                if best_score is None or score < best_score:
                    best_score, best_path = score, p
                # diversification: near-equal-cost alternative also acceptable
            if best_path is None:
                continue
            chosen.append(best_path)
            for a, b in zip(best_path[:-1], best_path[1:]):
                partition_edges[j].add((a, b))
                global_edge_count[(a, b)] = global_edge_count.get((a, b), 0) + 1
                bc_topology.append_dst_partition_path(dst, j, [a, b, G[a][b]])

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
