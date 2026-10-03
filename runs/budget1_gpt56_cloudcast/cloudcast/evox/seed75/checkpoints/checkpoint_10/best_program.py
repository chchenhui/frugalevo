# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Find an exact low-cost directed broadcast tree for up to ten destinations.

    The dynamic program minimizes the cost of unique tree edges, so a source to
    relay transfer is paid once and then reused by every destination below that
    relay.  Larger destination sets use shortest paths to retain predictable
    runtime.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    h.remove_edges_from([
        (u, v) for u, v, data in h.edges(data=True)
        if data.get("cost") is None
    ])
    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(dsts))

    def emit(paths):
        for dst, path in paths.items():
            for u, v in zip(path, path[1:]):
                for partition in range(num_partitions):
                    topology.append_dst_partition_path(
                        dst, partition, [u, v, G[u][v]]
                    )
        return topology

    # The exact directed-Steiner DP is exponential only in the number of clouds,
    # not in the number of regions.  The cutoff avoids excessive work on unusual
    # large fan-out configurations.
    if len(terminals) > 10:
        return emit({
            dst: nx.dijkstra_path(h, src, dst, weight="cost")
            for dst in terminals
        })

    nodes = list(h.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    masks = 1 << len(terminals)
    inf = float("inf")
    dp = [[inf] * n for _ in range(masks)]
    choice = [[None] * n for _ in range(masks)]
    reverse = h.reverse(copy=False)

    # A state is first formed by joining two subtrees at a region, then relaxed
    # backwards over links.  This is the standard directed Steiner recurrence.
    import heapq
    for mask in range(1, masks):
        if mask & (mask - 1) == 0:
            terminal = terminals[mask.bit_length() - 1]
            dp[mask][index[terminal]] = 0.0
        else:
            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if other and sub < other:
                    for i in range(n):
                        cost = dp[sub][i] + dp[other][i]
                        if cost < dp[mask][i]:
                            dp[mask][i] = cost
                            choice[mask][i] = ("split", sub, other)
                sub = (sub - 1) & mask

        heap = [(dp[mask][i], nodes[i]) for i in range(n) if dp[mask][i] < inf]
        heapq.heapify(heap)
        while heap:
            cost, child = heapq.heappop(heap)
            child_i = index[child]
            if cost != dp[mask][child_i]:
                continue
            for parent in reverse.neighbors(child):
                parent_i = index[parent]
                candidate = cost + float(h[parent][child]["cost"])
                if candidate < dp[mask][parent_i]:
                    dp[mask][parent_i] = candidate
                    choice[mask][parent_i] = ("edge", child)
                    heapq.heappush(heap, (candidate, parent))

    full = masks - 1
    edges = set()

    def recover(mask, node):
        step = choice[mask][index[node]]
        if step is None:
            return
        if step[0] == "split":
            recover(step[1], node)
            recover(step[2], node)
        else:
            child = step[1]
            edges.add((node, child))
            recover(mask, child)

    recover(full, src)
    tree = nx.DiGraph()
    tree.add_edges_from(edges)

    paths = {}
    for dst in terminals:
        try:
            paths[dst] = nx.shortest_path(tree, src, dst)
        except nx.NetworkXNoPath:
            paths[dst] = nx.dijkstra_path(h, src, dst, weight="cost")
    return emit(paths)


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
