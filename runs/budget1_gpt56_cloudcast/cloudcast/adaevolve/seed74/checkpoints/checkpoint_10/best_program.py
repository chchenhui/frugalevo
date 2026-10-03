# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Construct a minimum-cost directed Steiner-style broadcast tree.

    For up to nine unique destinations, dynamic programming finds a shared
    relay tree: each paid edge can carry the broadcast once before branching
    to multiple clouds.  The resulting tree routes are reused by every data
    partition, preserving parallel partition transfers without paying for
    independent source-to-destination paths.  Larger requests use a cheap
    incremental shared-tree fallback.
    """
    import heapq

    h = G.copy()
    h.remove_edges_from(
        list(h.in_edges(src)) + list(nx.selfloop_edges(h)) +
        [(u, v) for u, v, d in h.edges(data=True) if d.get("cost") is None]
    )
    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(d for d in dsts if d != src))

    def set_routes(edges):
        tree = nx.DiGraph()
        tree.add_edges_from(edges)
        for dst in dsts:
            if dst == src:
                route = []
            else:
                try:
                    nodes = nx.shortest_path(tree, src, dst)
                    route = [[u, v, G[u][v]] for u, v in zip(nodes, nodes[1:])]
                except nx.NetworkXNoPath:
                    nodes = nx.dijkstra_path(h, src, dst, weight="cost")
                    route = [[u, v, G[u][v]] for u, v in zip(nodes, nodes[1:])]
            for partition in range(num_partitions):
                topology.set_dst_partition_paths(dst, partition, list(route))

    # Exact directed-Steiner DP is practical for the small multicast groups
    # used by the optimizer.  dp[mask][node] is the cost to serve mask from
    # node; reverse Dijkstra allows a shared subtree to start at any relay.
    if len(terminals) <= 9:
        n = len(terminals)
        size = 1 << n
        inf = float("inf")
        reverse = {}
        for u, v, data in h.edges(data=True):
            reverse.setdefault(v, []).append((u, data["cost"]))

        dp = [[inf] * len(h) for _ in range(size)]
        nodes = list(h.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        choice = [[None] * len(h) for _ in range(size)]

        for mask in range(1, size):
            dist = dp[mask]
            decisions = choice[mask]

            if mask & (mask - 1) == 0:
                terminal = terminals[mask.bit_length() - 1]
                if terminal in index:
                    dist[index[terminal]] = 0
                    decisions[index[terminal]] = ("terminal",)
            else:
                sub = (mask - 1) & mask
                while sub:
                    other = mask ^ sub
                    if sub < other:
                        left, right = dp[sub], dp[other]
                        for i in range(len(nodes)):
                            value = left[i] + right[i]
                            if value < dist[i]:
                                dist[i] = value
                                decisions[i] = ("split", sub, other)
                    sub = (sub - 1) & mask

            queue = [(dist[i], nodes[i]) for i in range(len(nodes)) if dist[i] < inf]
            heapq.heapify(queue)
            while queue:
                value, v = heapq.heappop(queue)
                vi = index[v]
                if value != dist[vi]:
                    continue
                for u, edge_cost in reverse.get(v, []):
                    ui = index[u]
                    candidate = value + edge_cost
                    if candidate < dist[ui]:
                        dist[ui] = candidate
                        decisions[ui] = ("edge", v)
                        heapq.heappush(queue, (candidate, u))

        full = size - 1
        if src in index and dp[full][index[src]] < inf:
            edges = set()

            def expand(mask, node):
                entry = choice[mask][index[node]]
                if entry[0] == "edge":
                    edges.add((node, entry[1]))
                    expand(mask, entry[1])
                elif entry[0] == "split":
                    expand(entry[1], node)
                    expand(entry[2], node)

            expand(full, src)
            set_routes(edges)
            return topology

    # Scalable fallback: attach the next destination using the least new cost,
    # treating already selected broadcast links as free shared prefixes.
    reached, pending, edges = {src}, set(terminals), set()
    while pending:
        def marginal(u, v, data):
            return 0 if (u, v) in edges else data["cost"]

        distances, paths = nx.multi_source_dijkstra(h, reached, weight=marginal)
        candidates = [d for d in pending if d in distances]
        if not candidates:
            break
        dst = min(candidates, key=distances.get)
        path = paths[dst]
        edges.update(zip(path, path[1:]))
        reached.update(path)
        pending.difference_update(reached)

    set_routes(edges)
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
