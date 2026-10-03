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

    # Ensure every edge has a usable cost and throughput
    for u, v, data in h.edges(data=True):
        if data.get("cost") is None:
            data["cost"] = 1e6  # heavily penalize unknown-cost links
        if data.get("throughput") is None:
            data["throughput"] = 1.0

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Track residual capacity per edge to balance load across networks
    residual = {(u, v): d["throughput"] for u, v, d in h.edges(data=True)}

    for dst in dsts:
        # Generate diverse candidate paths: baseline shortest + perturbed variants
        candidates = []
        try:
            base = nx.dijkstra_path(h, src, dst, weight="cost")
            candidates.append(base)
        except nx.NetworkXNoPath:
            base = None

        # Perturbed searches to find parallel routes through other networks/regions
        rng = random.Random(hash((src, dst)) & 0xFFFFFFFF)
        perturbed = h.copy()
        for _, _, d in perturbed.edges(data=True):
            d["_wcost"] = d["cost"] * (0.5 + rng.random() * 1.5)
        for _ in range(4):
            for _, _, d in perturbed.edges(data=True):
                d["_wcost"] = d["cost"] * (0.3 + rng.random() * 2.0)
            try:
                p = nx.dijkstra_path(perturbed, src, dst, weight="_wcost")
                if p not in candidates:
                    candidates.append(p)
            except nx.NetworkXNoPath:
                pass

        if not candidates:
            # No path found; leave partitions unset
            continue

        # Score each candidate path: sum of edge costs, with bonus for edges
        # already being used (shared broadcast edges reduce redundant transfers)
        used_edges = set(e for e in residual if residual[e] < h[e[0]][e[1]]["throughput"])
        scored = []
        for path in candidates:
            cost_sum = 0.0
            feasible = True
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                cost_sum += h[s][t]["cost"]
                if residual[(s, t)] <= 0:
                    feasible = False
            # Prefer cheap, feasible, partly-shared paths
            score = cost_sum - (0.1 * cost_sum * (1 if not feasible else 0))
            scored.append((score, feasible, path))
        scored.sort(key=lambda x: (not x[1], x[0]))

        # Assign each partition: prefer shared/cheap paths, keep load balanced
        for j in range(num_partitions):
            chosen = None
            # Round-robin over feasible candidates to spread load across networks
            for idx in range(len(scored)):
                score, feasible, path = scored[idx % len(scored)]
                if feasible:
                    chosen = path
                    # Rotate candidate ordering so next partition may use
                    # a different parallel network path
                    scored.append(scored.pop(idx % len(scored)))
                    break
            if chosen is None:
                # Fallback: cheapest available regardless of capacity
                chosen = scored[0][2]
            for i in range(len(chosen) - 1):
                s, t = chosen[i], chosen[i + 1]
                residual[(s, t)] = max(0.0, residual[(s, t)] - 1.0 / max(num_partitions, 1))
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

    def set_graph(self):
        pass

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
