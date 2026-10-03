# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Build an exact low-cost directed Steiner broadcast tree for up to ten destinations.

    The subset dynamic program prices a relay edge only once when it is shared
    by multiple receivers.  It therefore finds cheaper shared inter-cloud
    prefixes than independently attaching destinations.  Larger terminal sets
    retain a scalable greedy shared-tree fallback.
    """
    import heapq

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(d for d in dsts if d != src))

    if not terminals:
        for dst in dsts:
            for partition in range(num_partitions):
                topology.set_dst_partition_paths(dst, partition, [])
        return topology

    def greedy_tree():
        reached, parent = {src}, {}
        remaining = set(terminals)
        while remaining:
            distances, paths = nx.multi_source_dijkstra(h, reached, weight="cost")
            dst = min((d for d in remaining if d in distances),
                      key=distances.get)
            for u, v in zip(paths[dst], paths[dst][1:]):
                if v not in reached:
                    reached.add(v)
                    parent[v] = u
            remaining.difference_update(reached)
        return {(u, v) for v, u in parent.items()}

    if len(terminals) > 10:
        selected = greedy_tree()
    else:
        nodes = list(h.nodes)
        index = {v: i for i, v in enumerate(nodes)}
        n = len(nodes)
        count = 1 << len(terminals)
        inf = float("inf")
        dp = [[inf] * n for _ in range(count)]
        choice = [[None] * n for _ in range(count)]

        # Each state is a minimum-cost arborescence rooted at node v which
        # reaches exactly the terminals represented by mask.
        for i, dst in enumerate(terminals):
            dp[1 << i][index[dst]] = 0.0
            choice[1 << i][index[dst]] = ("terminal", None)

        predecessors = [[] for _ in nodes]
        for u, v, data in h.edges(data=True):
            if data.get("cost") is not None:
                predecessors[index[v]].append((index[u], data["cost"]))

        for mask in range(1, count):
            if mask & (mask - 1):
                sub = (mask - 1) & mask
                while sub:
                    other = mask ^ sub
                    if sub < other:
                        for v in range(n):
                            candidate = dp[sub][v] + dp[other][v]
                            if candidate < dp[mask][v]:
                                dp[mask][v] = candidate
                                choice[mask][v] = ("split", (sub, other))
                    sub = (sub - 1) & mask

            # Multi-source Dijkstra on reversed edges implements
            # min_u(cost(v,u) + dp[mask][u]) for every possible relay v.
            heap = [(dp[mask][v], v) for v in range(n) if dp[mask][v] < inf]
            heapq.heapify(heap)
            while heap:
                distance, child = heapq.heappop(heap)
                if distance != dp[mask][child]:
                    continue
                for parent, edge_cost in predecessors[child]:
                    candidate = distance + edge_cost
                    if candidate < dp[mask][parent]:
                        dp[mask][parent] = candidate
                        choice[mask][parent] = ("edge", child)
                        heapq.heappush(heap, (candidate, parent))

        full_mask = count - 1
        root = index[src]
        if dp[full_mask][root] == inf:
            selected = greedy_tree()
        else:
            selected = set()

            def collect(mask, v):
                kind, value = choice[mask][v]
                if kind == "edge":
                    selected.add((nodes[v], nodes[value]))
                    collect(mask, value)
                elif kind == "split":
                    collect(value[0], v)
                    collect(value[1], v)

            collect(full_mask, root)

    tree = nx.DiGraph()
    tree.add_edges_from(selected)
    for dst in dsts:
        if dst == src:
            edges = []
        else:
            path = nx.shortest_path(tree, src, dst)
            edges = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
        for partition in range(num_partitions):
            topology.set_dst_partition_paths(dst, partition, list(edges))
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
