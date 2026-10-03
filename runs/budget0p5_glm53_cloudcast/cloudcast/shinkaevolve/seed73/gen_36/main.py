# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    K_PATHS = 6

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def edge_cost(s, t):
        c = h[s][t].get("cost")
        return c if c is not None else 1e9

    def path_cost(path, free_edges, penalty_edges=None):
        c = 0.0
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            if key not in free_edges:
                c += edge_cost(path[i], path[i + 1])
                if penalty_edges and key in penalty_edges:
                    c -= 0.25 * edge_cost(path[i], path[i + 1])
        return c

    def get_candidates(dst):
        cands = []
        try:
            count = 0
            for p in nx.shortest_simple_paths(h, src, dst, weight="cost"):
                cands.append(p)
                count += 1
                if count >= K_PATHS:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        if not cands:
            try:
                cands = [nx.dijkstra_path(h, src, dst, weight="cost")]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
        return cands

    # Pre-compute candidates and baseline costs for each destination
    info = {}
    for dst in dsts:
        if dst == src:
            continue
        cands = get_candidates(dst)
        if not cands:
            continue
        base_cost = min(path_cost(p, set()) for p in cands)
        info[dst] = {"cands": cands, "base": base_cost}

    # Expensive-first ordering: costly destinations choose first
    order = sorted(info.keys(), key=lambda d: -info[d]["base"])

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    shared_edges = set()

    for dst in order:
        cands = info[dst]["cands"]

        # One-step lookahead: build a hypothetical future trunk from
        # edges used by other destinations' current best candidates.
        future_trunk = set(shared_edges)
        for other, oinfo in info.items():
            if other == dst:
                continue
            best_other = min(
                oinfo["cands"],
                key=lambda p: path_cost(p, shared_edges),
            )
            for i in range(len(best_other) - 1):
                future_trunk.add((best_other[i], best_other[i + 1]))
        # Edges likely to be shared by multiple others get a bonus
        from collections import Counter
        counter = Counter()
        for other, oinfo in info.items():
            if other == dst:
                continue
            best_other = min(
                oinfo["cands"],
                key=lambda p: path_cost(p, shared_edges),
            )
            for i in range(len(best_other) - 1):
                counter[(best_other[i], best_other[i + 1])] += 1
        bonus_edges = set(k for k, v in counter.items() if v >= 1)

        # Score: shared edges free, plus lookahead bonus for edges
        # other destinations are also expected to use.
        def score(p):
            c = 0.0
            for i in range(len(p) - 1):
                key = (p[i], p[i + 1])
                if key in shared_edges:
                    continue  # fully free reuse
                ec = edge_cost(p[i], p[i + 1])
                if key in bonus_edges:
                    ec *= 0.75  # likely-shared discount for selection only
                c += ec
            return c

        best_p = min(cands, key=score)

        for i in range(len(best_p) - 1):
            s, t = best_p[i], best_p[i + 1]
            shared_edges.add((s, t))
            for j in range(num_partitions):
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

    def set_graph(self):
        pass


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