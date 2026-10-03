# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Find an exact shared directed broadcast tree for up to ten destinations.

    The dynamic program stores the minimum-cost subtree rooted at every cloud
    for every destination subset.  Subtrees are merged at relay clouds and
    then propagated backward over incoming links, allowing one ingress path to
    feed multiple destination networks.  Larger requests use independent
    shortest paths to keep the exponential subset search bounded.
    """
    import heapq

    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(dsts))
    if not terminals:
        return topology

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def cost(u, v, data):
        value = data.get("cost")
        return float(value) if value is not None else float("inf")

    # Exact directed Steiner DP is practical for the small broadcast groups
    # used by the optimizer; preserve predictable runtime for unusually large
    # requests.
    if len(terminals) > 10:
        tree = nx.DiGraph()
        for dst in terminals:
            tree.add_edges_from(zip(
                nx.dijkstra_path(h, src, dst, weight=cost),
                nx.dijkstra_path(h, src, dst, weight=cost)[1:],
            ))
    else:
        nodes = list(h.nodes)
        index = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        masks = 1 << len(terminals)
        inf = float("inf")
        dp = [[inf] * n for _ in range(masks)]
        choice = [[None] * n for _ in range(masks)]

        # A singleton starts at its destination.  Reverse relaxation computes
        # the cheapest route from every possible relay to that destination.
        for bit, dst in enumerate(terminals):
            dp[1 << bit][index[dst]] = 0.0

        for mask in range(1, masks):
            # Joining two already-closed subtrees at one relay is the key
            # operation that avoids paying a shared ingress edge twice.
            if mask & (mask - 1):
                sub = (mask - 1) & mask
                while sub:
                    other = mask ^ sub
                    if other and sub < other:
                        for i in range(n):
                            candidate = dp[sub][i] + dp[other][i]
                            if candidate < dp[mask][i]:
                                dp[mask][i] = candidate
                                choice[mask][i] = ("join", sub, other)
                    sub = (sub - 1) & mask

            heap = [(dp[mask][i], i) for i in range(n) if dp[mask][i] < inf]
            heapq.heapify(heap)
            while heap:
                distance, child = heapq.heappop(heap)
                if distance != dp[mask][child]:
                    continue
                child_node = nodes[child]
                for parent_node, _, data in h.in_edges(child_node, data=True):
                    parent = index[parent_node]
                    candidate = distance + cost(parent_node, child_node, data)
                    if candidate < dp[mask][parent]:
                        dp[mask][parent] = candidate
                        choice[mask][parent] = ("edge", child)
                        heapq.heappush(heap, (candidate, parent))

        full = masks - 1
        if dp[full][index[src]] == inf:
            raise nx.NetworkXNoPath("A destination is unreachable from source")

        edges = set()

        def collect(mask, node):
            """Recover the selected Steiner subtree from DP predecessor data."""
            decision = choice[mask][node]
            if decision is None:
                return
            if decision[0] == "join":
                collect(decision[1], node)
                collect(decision[2], node)
            else:
                child = decision[1]
                edges.add((nodes[node], nodes[child]))
                collect(mask, child)

        collect(full, index[src])
        tree = nx.DiGraph()
        tree.add_edges_from(edges)

    for dst in terminals:
        path = nx.dijkstra_path(tree, src, dst, weight=cost)
        for u, v in zip(path, path[1:]):
            for partition in range(num_partitions):
                topology.append_dst_partition_path(
                    dst, partition, [u, v, G[u][v]]
                )

    return topology


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
