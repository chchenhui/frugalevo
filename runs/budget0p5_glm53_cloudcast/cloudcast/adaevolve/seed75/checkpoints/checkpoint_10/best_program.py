# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def _greedy_tree(h, src, order, cost_attr="cost"):
    """Greedy directed-Steiner heuristic: route terminals in `order`,
    reusing already-built edges for free. Returns {dst: path}."""
    used = set()
    result = {}
    for dst in order:
        def w(u, v, data, used=used):
            return 0.0 if (u, v) in used else data[cost_attr]
        try:
            path = nx.dijkstra_path(h, src, dst, weight=w)
        except Exception:
            try:
                path = nx.shortest_path(h, src, dst)
            except Exception:
                continue
        for i in range(len(path) - 1):
            used.add((path[i], path[i + 1]))
        result[dst] = path
    return result


def search_algorithm(src, dsts, G, num_partitions):
    """Broadcast src's partitions to all dsts with minimal transfer cost.

    Approach: per-partition directed Steiner-tree approximation. For each
    partition, greedily route destinations one-by-one with Dijkstra where
    edges already carrying this partition cost 0 (free reuse => shared
    trunk links are paid once). To mitigate greedy ordering sensitivity,
    several destination orderings are tried (nearest-first, farthest-first,
    given order) and the cheapest tree is kept per partition.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Sanitize missing / NaN costs so Dijkstra is always valid.
    for _, _, d in h.edges(data=True):
        c = d.get("cost")
        if c is None or (isinstance(c, float) and c != c):
            d["cost"] = 1e12

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Base distances from src, used to order destinations for the greedy.
    try:
        dist = nx.single_source_dijkstra_path_length(h, src, weight="cost")
    except Exception:
        dist = {}

    reachable = [d for d in dsts if d in dist] or list(dsts)
    near_first = sorted(reachable, key=lambda d: dist.get(d, float("inf")))
    orders = [near_first, near_first[::-1], list(dsts)]
    # Extra orderings to escape greedy local optima.
    try:
        import random
        rng = random.Random(0)
        for _ in range(min(6, max(1, len(reachable)))):
            o = list(reachable)
            rng.shuffle(o)
            orders.append(o)
    except Exception:
        pass

    def tree_cost(paths):
        seen, c = set(), 0.0
        for p in paths.values():
            for i in range(len(p) - 1):
                e = (p[i], p[i + 1])
                if e not in seen:
                    seen.add(e)
                    c += h[e[0]][e[1]]["cost"]
        return c

    best_paths, best_cost = None, float("inf")
    for order in orders:
        paths = _greedy_tree(h, src, order)
        c = tree_cost(paths)
        if paths and (best_paths is None or len(paths) > len(best_paths) or
                      (len(paths) == len(best_paths) and c < best_cost)):
            best_paths, best_cost = paths, c
    if best_paths is None:
        best_paths = _greedy_tree(h, src, list(dsts))

    # The same optimal tree applies to every partition (identical data size,
    # identical network), so compute once and replicate.
    for j in range(bc_topology.num_partitions):
        for dst, path in best_paths.items():
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
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
