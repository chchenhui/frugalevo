# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import heapq
import math
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Minimize the cost of the union of broadcast links.  For typical broadcast
    fan-outs this uses an exact directed-Steiner DP, so a relay branch is paid
    for once even when it serves several clouds.
    """
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    if src not in G:
        return bc_topology

    h = nx.DiGraph()
    h.add_nodes_from(G.nodes)
    for u, v, data in G.edges(data=True):
        try:
            cost = float(data.get("cost"))
        except (TypeError, ValueError):
            continue
        if u != v and v != src and math.isfinite(cost) and cost >= 0:
            edge_data = dict(data)
            edge_data["cost"] = cost
            h.add_edge(u, v, **edge_data)

    terminals = list(dict.fromkeys(
        dst for dst in dsts
        if dst != src and dst in h and nx.has_path(h, src, dst)
    ))
    if not terminals:
        return bc_topology

    tree = nx.DiGraph()
    tree.add_node(src)

    if len(terminals) <= 11:
        # dp[mask][node] is the cheapest network rooted at node that reaches
        # all terminals in mask.  Backward Dijkstra permits a common prefix to
        # be shared by arbitrary subsets of destinations.
        nodes = list(h.nodes)
        full_mask = (1 << len(terminals)) - 1
        values = []
        choices = []

        for mask in range(full_mask + 1):
            row = {node: math.inf for node in nodes}
            choice = {node: None for node in nodes}

            if mask & (mask - 1) == 0:
                terminal = terminals[mask.bit_length() - 1]
                row[terminal] = 0.0
                choice[terminal] = ("terminal",)
            else:
                subset = (mask - 1) & mask
                while subset:
                    other = mask ^ subset
                    if subset < other:
                        for node in nodes:
                            candidate = values[subset][node] + values[other][node]
                            if candidate < row[node]:
                                row[node] = candidate
                                choice[node] = ("merge", subset, other)
                    subset = (subset - 1) & mask

            queue = [(value, node) for node, value in row.items()
                     if math.isfinite(value)]
            heapq.heapify(queue)
            while queue:
                current, node = heapq.heappop(queue)
                if current != row[node]:
                    continue
                for predecessor in h.predecessors(node):
                    candidate = current + h[predecessor][node]["cost"]
                    if candidate + 1e-12 < row[predecessor]:
                        row[predecessor] = candidate
                        choice[predecessor] = ("edge", node)
                        heapq.heappush(queue, (candidate, predecessor))

            values.append(row)
            choices.append(choice)

        def add_selected(mask, node, expanded=set()):
            state = (mask, node)
            if state in expanded:
                return
            expanded.add(state)
            decision = choices[mask][node]
            if decision is None or decision[0] == "terminal":
                return
            if decision[0] == "merge":
                add_selected(decision[1], node, expanded)
                add_selected(decision[2], node, expanded)
            else:
                next_node = decision[1]
                tree.add_edge(node, next_node, **dict(h[node][next_node]))
                add_selected(mask, next_node, expanded)

        if math.isfinite(values[full_mask][src]):
            add_selected(full_mask, src)
    else:
        # Preserve predictable runtime for unusually large fan-outs while still
        # reusing every branch that has already been selected.
        remaining = set(terminals)
        while remaining:
            best = None
            for anchor in list(tree.nodes):
                for dst in remaining:
                    try:
                        path = nx.dijkstra_path(h, anchor, dst, weight="cost")
                        cost = nx.path_weight(h, path, weight="cost")
                    except nx.NetworkXNoPath:
                        continue
                    if best is None or cost < best[0]:
                        best = (cost, dst, path)
            if best is None:
                break
            _, dst, path = best
            for u, v in zip(path, path[1:]):
                tree.add_edge(u, v, **dict(h[u][v]))
            remaining.remove(dst)

    for dst in dsts:
        if dst == src or dst not in tree or not nx.has_path(tree, src, dst):
            continue
        path = nx.shortest_path(tree, src, dst)
        edges = [[u, v, dict(G[u][v])] for u, v in zip(path, path[1:])]
        for partition in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(dst, partition, list(edges))

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