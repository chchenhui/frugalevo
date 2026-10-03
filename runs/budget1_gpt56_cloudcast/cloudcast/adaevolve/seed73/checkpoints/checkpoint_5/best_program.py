# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Find a minimum-cost shared directed broadcast tree for small receiver sets.

    The dynamic program solves the directed Steiner-tree recurrence exactly: at
    every relay it may merge two receiver groups, then it propagates that merged
    group backwards over the cheapest incoming routes.  Thus a costly inter-cloud
    hop is paid once and can fan out to several destinations.  Large receiver
    sets use the previous greedy multicast-tree approximation to bound runtime.
    """
    import heapq

    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(d for d in dsts if d != src))

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    h.remove_edges_from([
        (u, v) for u, v, data in h.edges(data=True)
        if data.get("cost") is None or data.get("cost") != data.get("cost")
    ])

    if not terminals:
        for dst in dsts:
            for part in range(num_partitions):
                topology.set_dst_partition_paths(dst, part, [])
        return topology

    tree_edges = set()

    # Exact directed Steiner DP is practical for the small multicast groups used
    # by the benchmark.  Above this limit retain a fast sharing-aware heuristic.
    if len(terminals) <= 13:
        nodes = list(h.nodes)
        reverse = h.reverse(copy=False)
        count = len(terminals)
        full = (1 << count) - 1
        costs, choices = {}, {}

        for mask in range(1, full + 1):
            dist = {v: float("inf") for v in nodes}
            choice = {}
            heap = []

            if mask & (mask - 1) == 0:
                terminal = terminals[mask.bit_length() - 1]
                if terminal in h:
                    dist[terminal] = 0.0
                    choice[terminal] = ("terminal",)
                    heapq.heappush(heap, (0.0, terminal))
            else:
                sub = (mask - 1) & mask
                while sub:
                    other = mask ^ sub
                    if sub < other:
                        left, right = costs[sub], costs[other]
                        for v in nodes:
                            value = left[v] + right[v]
                            if value < dist[v]:
                                dist[v] = value
                                choice[v] = ("merge", sub, other)
                                heapq.heappush(heap, (value, v))
                    sub = (sub - 1) & mask

            # Reverse Dijkstra computes min_u(distance(v,u) + merged_cost(u)).
            while heap:
                value, node = heapq.heappop(heap)
                if value != dist[node]:
                    continue
                for predecessor in reverse.neighbors(node):
                    candidate = value + h[predecessor][node]["cost"]
                    if candidate < dist[predecessor]:
                        dist[predecessor] = candidate
                        choice[predecessor] = ("edge", node)
                        heapq.heappush(heap, (candidate, predecessor))

            costs[mask], choices[mask] = dist, choice

        if src in costs[full] and costs[full][src] < float("inf"):
            visited = set()

            def collect(mask, node):
                state = (mask, node)
                if state in visited:
                    return
                visited.add(state)
                action = choices[mask].get(node)
                if not action or action[0] == "terminal":
                    return
                if action[0] == "edge":
                    tree_edges.add((node, action[1]))
                    collect(mask, action[1])
                else:
                    collect(action[1], node)
                    collect(action[2], node)

            collect(full, src)

    if not tree_edges:
        reached = {src}
        remaining = set(terminals)
        while remaining:
            best = None
            for dst in remaining:
                try:
                    value, path = nx.multi_source_dijkstra(
                        h, list(reached), dst, weight="cost"
                    )
                    candidate = (value, dst, path)
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            if best is None:
                break
            _, dst, path = best
            tree_edges.update(zip(path, path[1:]))
            reached.update(path)
            remaining.remove(dst)

    tree = nx.DiGraph()
    for u, v in tree_edges:
        tree.add_edge(u, v, **G[u][v])

    for dst in dsts:
        try:
            path = [src] if dst == src else nx.dijkstra_path(
                tree, src, dst, weight="cost"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                path = nx.dijkstra_path(h, src, dst, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        edges = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
        for part in range(num_partitions):
            topology.set_dst_partition_paths(dst, part, list(edges))

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
