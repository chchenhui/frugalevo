# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import heapq
import math
import pandas as pd
from typing import Dict, List


def _valid_cost_graph(src, G):
    """Create a routable copy containing only finite non-negative cost edges."""
    h = G.copy()

    if src not in h:
        return h

    h.remove_edges_from(list(h.in_edges(src)))
    h.remove_edges_from(list(nx.selfloop_edges(h)))

    invalid_edges = []
    for u, v, data in h.edges(data=True):
        try:
            cost = float(data.get("cost"))
            if not math.isfinite(cost) or cost < 0:
                invalid_edges.append((u, v))
        except (TypeError, ValueError):
            invalid_edges.append((u, v))

    h.remove_edges_from(invalid_edges)
    return h


def _add_edges_from_path(tree, path, G):
    """Add a node path to a tree using the original edge attributes."""
    for u, v in zip(path, path[1:]):
        tree.add_edge(u, v, **G[u][v])


def _normalize_tree(src, terminals, tree, G):
    """
    Keep only source-reachable shortest routes inside the selected edge union.

    Greedy expansion can leave cycle edges or redundant relay links. Rebuilding
    the routes from the source over the selected union retains all reachable
    terminals while reducing the final shared-link set.
    """
    normalized = nx.DiGraph()
    normalized.add_node(src)

    if src not in tree:
        return normalized

    try:
        paths = nx.single_source_dijkstra_path(tree, src, weight="cost")
    except (nx.NetworkXError, ValueError):
        return normalized

    for terminal in terminals:
        path = paths.get(terminal)
        if path is not None:
            _add_edges_from_path(normalized, path, G)

    return normalized


def _exact_steiner_tree(src, terminals, h, G):
    """
    Directed Steiner dynamic programming.

    dp[mask][node] is the cheapest subtree rooted at node that reaches every
    terminal represented by mask.  Merging masks models a relay receiving a
    partition once and broadcasting it to multiple downstream branches.
    """
    terminal_count = len(terminals)
    if terminal_count == 0:
        tree = nx.DiGraph()
        tree.add_node(src)
        return tree

    nodes = list(h.nodes)
    full_mask = (1 << terminal_count) - 1
    infinity = float("inf")
    dp = {}
    decisions = {}

    for index, terminal in enumerate(terminals):
        mask = 1 << index
        distance = {node: infinity for node in nodes}
        decision = {}
        distance[terminal] = 0.0
        decision[terminal] = ("terminal",)
        queue = [(0.0, terminal)]

        while queue:
            current_cost, node = heapq.heappop(queue)
            if current_cost != distance[node]:
                continue

            for predecessor in h.predecessors(node):
                candidate = current_cost + h[predecessor][node]["cost"]
                if candidate < distance[predecessor]:
                    distance[predecessor] = candidate
                    decision[predecessor] = ("edge", node)
                    heapq.heappush(queue, (candidate, predecessor))

        dp[mask] = distance
        decisions[mask] = decision

    for mask in range(1, full_mask + 1):
        if mask & (mask - 1) == 0:
            continue

        distance = {node: infinity for node in nodes}
        decision = {}

        subset = (mask - 1) & mask
        while subset:
            other = mask ^ subset
            if other and subset < other:
                left = dp[subset]
                right = dp[other]
                for node in nodes:
                    candidate = left[node] + right[node]
                    if candidate < distance[node]:
                        distance[node] = candidate
                        decision[node] = ("merge", subset, other)
            subset = (subset - 1) & mask

        queue = [(value, node) for node, value in distance.items() if value < infinity]
        heapq.heapify(queue)

        while queue:
            current_cost, node = heapq.heappop(queue)
            if current_cost != distance[node]:
                continue

            for predecessor in h.predecessors(node):
                candidate = current_cost + h[predecessor][node]["cost"]
                if candidate < distance[predecessor]:
                    distance[predecessor] = candidate
                    decision[predecessor] = ("edge", node)
                    heapq.heappush(queue, (candidate, predecessor))

        dp[mask] = distance
        decisions[mask] = decision

    if dp[full_mask].get(src, infinity) == infinity:
        return None

    selected_edges = set()
    visiting = set()

    def add_subtree(mask, node):
        state = (mask, node)
        if state in visiting:
            return

        choice = decisions[mask].get(node)
        if choice is None or choice[0] == "terminal":
            return

        visiting.add(state)
        if choice[0] == "edge":
            next_node = choice[1]
            selected_edges.add((node, next_node))
            add_subtree(mask, next_node)
        else:
            add_subtree(choice[1], node)
            add_subtree(choice[2], node)
        visiting.remove(state)

    add_subtree(full_mask, src)

    tree = nx.DiGraph()
    tree.add_node(src)
    for u, v in selected_edges:
        tree.add_edge(u, v, **G[u][v])

    return _normalize_tree(src, terminals, tree, G)


def _greedy_shared_tree(src, terminals, h, G):
    """
    Scalable marginal-cost broadcast construction.

    Every iteration treats all already-reached nodes as simultaneous transfer
    sources.  Therefore the next destination is attached through the cheapest
    additional path rather than paying again for the common source prefix.
    """
    tree = nx.DiGraph()
    tree.add_node(src)
    pending = set(terminals)

    while pending:
        sources = list(tree.nodes)
        lengths, paths = nx.multi_source_dijkstra(h, sources, weight="cost")

        candidates = [
            (lengths[dst], str(dst), dst, paths[dst])
            for dst in pending
            if dst in lengths
        ]
        if not candidates:
            break

        _, _, dst, path = min(candidates)
        _add_edges_from_path(tree, path, G)

        # Any requested destination encountered along the relay path has
        # already received the partition and needs no additional transfer.
        pending.difference_update(set(path))

    return _normalize_tree(src, terminals, tree, G)


def search_algorithm(src, dsts, G, num_partitions):
    """
    Build a low-cost shared broadcast topology.

    Exact Steiner optimization is used for moderate terminal counts; a
    multi-source marginal-cost expansion is used beyond that limit to preserve
    practical runtime on larger multi-cloud topologies.
    """
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    if src not in G:
        for dst in dsts:
            for partition in range(bc_topology.num_partitions):
                bc_topology.set_dst_partition_paths(dst, partition, [])
        return bc_topology

    h = _valid_cost_graph(src, G)
    unique_terminals = sorted({dst for dst in dsts if dst != src and dst in h}, key=str)

    # The state space grows as O(3^k). Eleven terminals remains practical for
    # profile-sized cloud graphs while providing substantially more exact
    # shared-routing opportunities than independent shortest paths.
    if unique_terminals and len(unique_terminals) <= 11:
        tree = _exact_steiner_tree(src, unique_terminals, h, G)
        if tree is None:
            tree = _greedy_shared_tree(src, unique_terminals, h, G)
    else:
        tree = _greedy_shared_tree(src, unique_terminals, h, G)

    try:
        route_nodes = nx.single_source_dijkstra_path(tree, src, weight="cost")
    except (nx.NetworkXError, ValueError):
        route_nodes = {src: [src]}

    for dst in dsts:
        if dst == src:
            edge_path = []
        else:
            node_path = route_nodes.get(dst)
            if node_path is None:
                edge_path = []
            else:
                edge_path = [
                    [u, v, G[u][v]]
                    for u, v in zip(node_path, node_path[1:])
                ]

        for partition in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(
                dst,
                partition,
                list(edge_path),
            )

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    def __init__(self, src: str, dsts: List[str], num_partitions: int = 4, paths: Dict[str, SingleDstPath] = None):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions

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
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Default graph with capacity constraints and cost info.
    """
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
        G.add_edge(
            row["src_region"],
            row["dst_region"],
            cost=None,
            throughput=num_vms * row["throughput_sent"] / 1e9,
        )

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    no_cost_pairs = []
    for edge in G.edges.data():
        edge_src, edge_dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((edge_src, edge_dst))
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