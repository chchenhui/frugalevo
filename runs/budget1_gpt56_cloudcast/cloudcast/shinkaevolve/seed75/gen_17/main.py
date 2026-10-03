# EVOLVE-BLOCK-START
import heapq
import math
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


EXACT_TERMINAL_LIMIT = 11
EPSILON = 1e-12


def _safe_cost(data):
    """Return a finite non-negative edge cost, or infinity for invalid links."""
    try:
        value = float(data.get("cost"))
        if math.isfinite(value) and value >= 0:
            return value
    except (TypeError, ValueError):
        pass
    return math.inf


def _path_edges(path):
    return {(u, v) for u, v in zip(path, path[1:])}


def _reverse_closure(graph, initial_costs, initial_decisions):
    """
    Extend DP states upstream using a multi-source Dijkstra traversal over
    reverse edges.  A state rooted at predecessor forwards through `node`.
    """
    distances = dict(initial_costs)
    decisions = dict(initial_decisions)
    queue = []

    for node, distance in distances.items():
        if math.isfinite(distance):
            heapq.heappush(queue, (distance, str(node), node))

    while queue:
        distance, _, node = heapq.heappop(queue)
        if distance > distances.get(node, math.inf) + EPSILON:
            continue

        for predecessor in graph.predecessors(node):
            candidate = distance + _safe_cost(graph[predecessor][node])
            if candidate + EPSILON < distances.get(predecessor, math.inf):
                distances[predecessor] = candidate
                decisions[predecessor] = ("edge", node)
                heapq.heappush(queue, (candidate, str(predecessor), predecessor))

    return distances, decisions


def _exact_steiner_tree(graph, src, terminals):
    """
    Find a minimum-cost directed tree rooted at src reaching all terminals.

    dp[mask][node] is the cost of a subtree rooted at node.  At every cloud,
    subset states can merge and thereby reuse a single upstream transfer.
    """
    nodes = list(graph.nodes())
    count = len(terminals)
    full_mask = (1 << count) - 1
    dp = [None] * (full_mask + 1)
    decisions = [None] * (full_mask + 1)

    for index, terminal in enumerate(terminals):
        mask = 1 << index
        initial = {node: math.inf for node in nodes}
        initial[terminal] = 0.0
        dp[mask], decisions[mask] = _reverse_closure(
            graph, initial, {terminal: ("terminal",)}
        )

    for mask in range(1, full_mask + 1):
        if not (mask & (mask - 1)):
            continue

        initial = {node: math.inf for node in nodes}
        initial_decisions = {}
        subset = (mask - 1) & mask
        while subset:
            other = mask ^ subset
            if other and subset < other:
                for node in nodes:
                    candidate = dp[subset][node] + dp[other][node]
                    if candidate + EPSILON < initial[node]:
                        initial[node] = candidate
                        initial_decisions[node] = ("merge", subset, other)
            subset = (subset - 1) & mask

        dp[mask], decisions[mask] = _reverse_closure(
            graph, initial, initial_decisions
        )

    if not math.isfinite(dp[full_mask].get(src, math.inf)):
        return None

    edges = set()
    active = set()

    def collect(mask, node):
        state = (mask, node)
        if state in active:
            return
        active.add(state)
        decision = decisions[mask].get(node)
        if decision:
            if decision[0] == "edge":
                edges.add((node, decision[1]))
                collect(mask, decision[1])
            elif decision[0] == "merge":
                collect(decision[1], node)
                collect(decision[2], node)
        active.discard(state)

    collect(full_mask, src)
    return edges


def _greedy_shared_tree(graph, src, terminals):
    """Scalable fallback with installed broadcast edges free to reuse."""
    tree_nodes = {src}
    tree_edges = set()
    remaining = set(terminals)

    while remaining:
        distances = {node: math.inf for node in graph.nodes()}
        parents = {}
        queue = [(0.0, str(src), src)]
        distances[src] = 0.0

        while queue:
            distance, _, node = heapq.heappop(queue)
            if distance > distances[node] + EPSILON:
                continue
            for _, nxt, data in graph.out_edges(node, data=True):
                edge_cost = 0.0 if (node, nxt) in tree_edges else _safe_cost(data)
                candidate = distance + edge_cost
                if candidate + EPSILON < distances[nxt]:
                    distances[nxt] = candidate
                    parents[nxt] = node
                    heapq.heappush(queue, (candidate, str(nxt), nxt))

        reachable = [dst for dst in remaining if math.isfinite(distances[dst])]
        if not reachable:
            return None

        destination = min(reachable, key=lambda dst: (distances[dst], str(dst)))
        path = [destination]
        while path[-1] not in tree_nodes:
            parent = parents.get(path[-1])
            if parent is None:
                return None
            path.append(parent)
        path.reverse()
        tree_edges.update(_path_edges(path))
        tree_nodes.update(path)
        remaining.remove(destination)

    return tree_edges


def search_algorithm(src, dsts, G, num_partitions):
    """Construct a low-cost shared tree for concurrent cloud broadcast."""
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    if src not in G:
        return bc_topology

    destinations = list(dict.fromkeys(dsts))
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    h.remove_edges_from([
        (u, v) for u, v, data in h.edges(data=True)
        if not math.isfinite(_safe_cost(data))
    ])

    if src in bc_topology.paths:
        for partition in range(num_partitions):
            bc_topology.set_dst_partition_paths(src, partition, [])

    terminals = []
    for dst in destinations:
        if dst == src or dst not in h:
            continue
        try:
            nx.dijkstra_path(h, src, dst, weight=lambda u, v, d: _safe_cost(d))
            terminals.append(dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not terminals:
        return bc_topology

    if len(terminals) <= EXACT_TERMINAL_LIMIT:
        selected_edges = _exact_steiner_tree(h, src, terminals)
    else:
        selected_edges = _greedy_shared_tree(h, src, terminals)

    if selected_edges is None:
        return bc_topology

    multicast_graph = nx.DiGraph()
    multicast_graph.add_node(src)
    for u, v in selected_edges:
        multicast_graph.add_edge(u, v, **G[u][v])

    for dst in terminals:
        try:
            path = nx.dijkstra_path(
                multicast_graph, src, dst,
                weight=lambda u, v, data: _safe_cost(data)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        edge_path = [
            [u, v, G[u][v]]
            for u, v in zip(path, path[1:])
        ]
        for partition in range(num_partitions):
            bc_topology.set_dst_partition_paths(dst, partition, list(edge_path))

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