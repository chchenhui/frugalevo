# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
import random
import math
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions, num_samples=200, lam=0.05):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    edge_load = {}

    def edge_cost(s, t):
        c = h[s][t].get("cost")
        return c if c is not None else 0.0

    def edge_tput(s, t):
        tp = h[s][t].get("throughput")
        return tp if tp and tp > 0 else 1.0

    def path_cost(path):
        return sum(edge_cost(path[i], path[i + 1]) for i in range(len(path) - 1))

    def marginal_cost(path):
        # cost + congestion penalty scaled by relative load vs throughput
        c = 0.0
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            load = edge_load.get((s, t), 0)
            c += edge_cost(s, t) * (1.0 + lam * load / edge_tput(s, t))
        return c

    def sample_path(dst, rng):
        # Randomized greedy walk biased toward cheap, uncongested edges
        # using a biased random shortest-path (edge weights perturbed).
        node = src
        path = [src]
        visited = {src}
        guard = 0
        while node != dst:
            guard += 1
            if guard > 200:
                return None
            nexts = [n for n in h[node] if n not in visited]
            if not nexts:
                return None
            weights = []
            for n in nexts:
                base = edge_cost(node, n) + 1e-9
                load = edge_load.get((node, n), 0)
                # convex combination of true cost and noisy cost
                w = (base ** rng.uniform(0.5, 2.0)) * (
                    1.0 + lam * load / edge_tput(node, n)
                )
                weights.append(1.0 / (w + 1e-9))
            # noise for exploration
            weights = [w * rng.uniform(0.5, 2.0) for w in weights]
            node = rng.choices(nexts, weights=weights, k=1)[0]
            path.append(node)
            visited.add(node)
        return path

    def get_candidate_pool(dst):
        pool = {}
        try:
            for p in nx.shortest_simple_paths(h, src, dst, weight="cost"):
                pool[tuple(p)] = path_cost(p)
                if len(pool) >= 30:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        # Monte-Carlo exploration
        rng = random.Random(hash(dst) & 0xffffffff)
        for _ in range(num_samples):
            p = sample_path(dst, rng)
            if p is not None and tuple(p) not in pool:
                pool[tuple(p)] = path_cost(p)
        if not pool:
            # fallback: any path
            try:
                p = nx.dijkstra_path(h, src, dst, weight="cost")
                pool[tuple(p)] = path_cost(p)
            except nx.NetworkXNoPath:
                return [None]
        # prune: keep paths within a cost slack of the best
        best = min(pool.values())
        pruned = [list(p) for p, c in pool.items() if c <= best * 3.0 + 1e-6]
        pruned.sort(key=path_cost)
        return pruned if pruned else [None]

    for dst in dsts:
        pool = get_candidate_pool(dst)
        if pool == [None]:
            # unreachable: fill with None paths to avoid crashing
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [])
            continue

        # Iterative congestion-aware assignment: process partitions one at a
        # time, always picking the path with the lowest *marginal* cost given
        # current edge load.
        for j in range(bc_topology.num_partitions):
            best_path = None
            best_mc = float("inf")
            for p in pool:
                mc = marginal_cost(p)
                if mc < best_mc:
                    best_mc = mc
                    best_path = p
            for i in range(0, len(best_path) - 1):
                s, t = best_path[i], best_path[i + 1]
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
                edge_load[(s, t)] = edge_load.get((s, t), 0) + 1

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