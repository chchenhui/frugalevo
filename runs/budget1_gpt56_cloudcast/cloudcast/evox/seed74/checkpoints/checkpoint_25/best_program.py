# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Build a minimum-cost shared directed broadcast tree via subset DP.

    For up to seven unique non-source destinations, this uses an exact
    directed-Steiner dynamic program: each state is the cheapest subtree
    rooted at a node that reaches a subset of destinations.  The resulting
    edge union is globally optimized for sharing.  Larger requests use the
    previous fast incremental-tree strategy.
    """
    import heapq

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = [d for d in dict.fromkeys(dsts) if d != src]

    if not terminals:
        return topology

    # Keep the exact solver bounded; its complexity is O(3^k * |V|).
    if len(terminals) <= 7:
        count = len(terminals)
        states = 1 << count
        cost = [dict() for _ in range(states)]
        choice = [dict() for _ in range(states)]

        for mask in range(1, states):
            initial = {}
            decisions = {}

            # A singleton can terminate at its destination at zero extra cost.
            if mask & (mask - 1) == 0:
                terminal = terminals[mask.bit_length() - 1]
                initial[terminal] = 0.0
                decisions[terminal] = ("terminal",)
            else:
                part = (mask - 1) & mask
                while part:
                    other = mask ^ part
                    # Evaluating one side of each split avoids duplicate work.
                    if part < other:
                        for node, left_cost in cost[part].items():
                            right_cost = cost[other].get(node)
                            if right_cost is not None:
                                candidate = left_cost + right_cost
                                if candidate < initial.get(node, float("inf")):
                                    initial[node] = candidate
                                    decisions[node] = ("split", part, other)
                    part = (part - 1) & mask

            # Reverse Dijkstra applies an arbitrary shared prefix before the
            # subtree represented by this mask.
            heap = [(value, node) for node, value in initial.items()]
            heapq.heapify(heap)
            cost[mask] = initial
            choice[mask] = decisions

            while heap:
                value, node = heapq.heappop(heap)
                if value != cost[mask].get(node):
                    continue
                for predecessor in h.predecessors(node):
                    edge_cost = h[predecessor][node].get("cost")
                    if edge_cost is None:
                        continue
                    candidate = value + edge_cost
                    if candidate < cost[mask].get(predecessor, float("inf")):
                        cost[mask][predecessor] = candidate
                        choice[mask][predecessor] = ("edge", node)
                        heapq.heappush(heap, (candidate, predecessor))

        full = states - 1
        if src not in cost[full]:
            raise nx.NetworkXNoPath("A broadcast destination is unreachable")

        selected_edges = set()

        def collect(node, mask):
            decision = choice[mask][node]
            if decision[0] == "terminal":
                return
            if decision[0] == "edge":
                child = decision[1]
                selected_edges.add((node, child))
                collect(child, mask)
            else:
                collect(node, decision[1])
                collect(node, decision[2])

        collect(src, full)
        tree = nx.DiGraph()
        tree.add_edges_from(selected_edges)

        for dst in terminals:
            try:
                path = nx.shortest_path(tree, src, dst)
            except nx.NetworkXNoPath:
                raise nx.NetworkXNoPath("A broadcast destination is unreachable")
            edges = [[a, b, G[a][b]] for a, b in zip(path, path[1:])]
            for partition in range(num_partitions):
                topology.set_dst_partition_paths(dst, partition, edges)
        return topology

    # Scalable fallback for large destination sets.
    reached = {src}
    source_paths = {src: [src]}
    pending = terminals

    while pending:
        distances, extensions = nx.multi_source_dijkstra(h, reached, weight="cost")
        candidates = [dst for dst in pending if dst in distances]
        if not candidates:
            raise nx.NetworkXNoPath("A broadcast destination is unreachable")

        dst = min(candidates, key=lambda node: distances[node])
        extension = extensions[dst]
        attach = extension[0]
        path = source_paths[attach] + extension[1:]

        for index, node in enumerate(extension[1:], 1):
            source_paths.setdefault(node, source_paths[attach] + extension[1:index + 1])

        reached.update(extension)
        pending.remove(dst)

        for a, b in zip(path, path[1:]):
            edge = [a, b, G[a][b]]
            for partition in range(num_partitions):
                topology.append_dst_partition_path(dst, partition, edge)

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
