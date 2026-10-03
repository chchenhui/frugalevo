# EVOLVE-BLOCK-START
import heapq
import math
import os
from typing import Dict, List

import networkx as nx
import pandas as pd


# Exact directed-Steiner optimization remains tractable for common cloud
# broadcast fanouts. Larger terminal sets use a multi-start shared-prefix
# heuristic that scales without independently replicating source transfers.
EXACT_TERMINAL_LIMIT = 13
EPSILON = 1e-12


def _edge_cost(data):
    """Return a finite non-negative transfer price or infinity."""
    try:
        value = float(data.get("cost"))
        return value if math.isfinite(value) and value >= 0 else math.inf
    except (TypeError, ValueError):
        return math.inf


def _priced_graph(src, G):
    """Copy only usable directed links for source-rooted cost optimization."""
    h = G.copy()

    if src in h:
        h.remove_edges_from(list(h.in_edges(src)))
    h.remove_edges_from(list(nx.selfloop_edges(h)))

    h.remove_edges_from([
        (u, v)
        for u, v, data in h.edges(data=True)
        if not math.isfinite(_edge_cost(data))
    ])
    return h


def _edge_union_cost(h, edges):
    """Cost of physical links installed by a shared broadcast topology."""
    return sum(_edge_cost(h[u][v]) for u, v in edges)


def _prune_shared_edges(src, terminals, edges):
    """Remove relay leaves that cannot contribute to any destination path."""
    tree = nx.DiGraph()
    tree.add_edges_from(edges)
    protected = set(terminals)
    protected.add(src)

    changed = True
    while changed:
        changed = False
        for node in list(tree.nodes()):
            if node not in protected and tree.out_degree(node) == 0:
                tree.remove_node(node)
                changed = True

    return set(tree.edges())


def _relax_subset(h, nodes, node_index, seed_costs, seed_choices):
    """
    Reverse Dijkstra relaxation for a directed-Steiner DP subset.

    A solution rooted at u can be extended to a predecessor v by installing
    v->u, so traversing predecessor links computes the least expensive relay
    from which the subset can be broadcast.
    """
    distances = list(seed_costs)
    choices = list(seed_choices)
    heap = [
        (cost, index)
        for index, cost in enumerate(distances)
        if math.isfinite(cost)
    ]
    heapq.heapify(heap)

    while heap:
        current_cost, node_position = heapq.heappop(heap)
        if current_cost > distances[node_position] + EPSILON:
            continue

        node = nodes[node_position]
        for predecessor in h.predecessors(node):
            predecessor_position = node_index[predecessor]
            candidate = current_cost + _edge_cost(h[predecessor][node])

            if candidate + EPSILON < distances[predecessor_position]:
                distances[predecessor_position] = candidate
                choices[predecessor_position] = ("edge", node_position)
                heapq.heappush(heap, (candidate, predecessor_position))

    return distances, choices


def _exact_shared_edges(src, terminals, h):
    """
    Compute a minimum-cost directed Steiner broadcast arborescence.

    dp[mask][v] is the cost to send from relay v to every destination
    represented by mask. A state is built either by merging two subsets at
    the same relay or by extending a subtree through an incoming edge.
    """
    if src not in h:
        raise nx.NodeNotFound(f"Source {src} is not present in the graph")

    nodes = list(h.nodes())
    node_index = {node: index for index, node in enumerate(nodes)}
    terminal_count = len(terminals)
    state_count = 1 << terminal_count
    node_count = len(nodes)
    full_mask = state_count - 1

    dp = [[math.inf] * node_count for _ in range(state_count)]
    decisions = [[None] * node_count for _ in range(state_count)]

    for mask in range(1, state_count):
        costs = [math.inf] * node_count
        choices = [None] * node_count

        if mask & (mask - 1) == 0:
            terminal_position = mask.bit_length() - 1
            terminal = terminals[terminal_position]
            if terminal in node_index:
                position = node_index[terminal]
                costs[position] = 0.0
                choices[position] = ("terminal", terminal_position)
        else:
            anchor = mask & -mask
            subset = (mask - 1) & mask

            while subset:
                other = mask ^ subset
                if other and subset & anchor:
                    left = dp[subset]
                    right = dp[other]
                    for position in range(node_count):
                        candidate = left[position] + right[position]
                        if candidate + EPSILON < costs[position]:
                            costs[position] = candidate
                            choices[position] = ("merge", subset, other)
                subset = (subset - 1) & mask

        dp[mask], decisions[mask] = _relax_subset(
            h, nodes, node_index, costs, choices
        )

    root_position = node_index[src]
    if not math.isfinite(dp[full_mask][root_position]):
        missing = ", ".join(str(dst) for dst in terminals)
        raise nx.NetworkXNoPath(
            f"No priced broadcast route from {src} to: {missing}"
        )

    selected_edges = set()
    visited = set()

    def reconstruct(mask, position):
        state = (mask, position)
        if state in visited:
            return
        visited.add(state)

        decision = decisions[mask][position]
        if decision is None or decision[0] == "terminal":
            return

        if decision[0] == "merge":
            reconstruct(decision[1], position)
            reconstruct(decision[2], position)
        else:
            next_position = decision[1]
            selected_edges.add((nodes[position], nodes[next_position]))
            reconstruct(mask, next_position)

    reconstruct(full_mask, root_position)
    return _prune_shared_edges(src, terminals, selected_edges)


def _shortest_path(h, src, dst):
    """Return a priced shortest path, or None when no directed route exists."""
    try:
        return nx.dijkstra_path(h, src, dst, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _multistart_greedy_shared_edges(src, terminals, h):
    """
    Shared-prefix fallback for large destination sets.

    Installed links receive zero incremental cost, which favors forwarding an
    already delivered stream through existing relays. Multiple initial trunks
    are evaluated because regional fan-out quality depends on the first branch.
    """
    direct_paths = {
        dst: _shortest_path(h, src, dst)
        for dst in terminals
    }
    unreachable = [dst for dst, path in direct_paths.items() if path is None]
    if unreachable:
        missing = ", ".join(str(dst) for dst in unreachable)
        raise nx.NetworkXNoPath(
            f"No priced broadcast route from {src} to: {missing}"
        )

    ordered_seeds = [None] + sorted(
        terminals,
        key=lambda dst: (
            sum(_edge_cost(h[u][v]) for u, v in zip(
                direct_paths[dst], direct_paths[dst][1:]
            )),
            str(dst),
        ),
    )

    best_edges = None
    best_score = None

    for seed in ordered_seeds:
        remaining = set(terminals)
        tree_nodes = {src}
        installed = set()

        if seed is not None:
            for u, v in zip(direct_paths[seed], direct_paths[seed][1:]):
                installed.add((u, v))
                tree_nodes.add(u)
                tree_nodes.add(v)
            remaining.remove(seed)

        while remaining:
            def incremental_weight(u, v, data):
                if (u, v) in installed:
                    return 0.0
                return _edge_cost(data)

            distances, paths = nx.multi_source_dijkstra(
                h, list(tree_nodes), weight=incremental_weight
            )

            reachable = [
                dst for dst in remaining
                if dst in distances and math.isfinite(distances[dst])
            ]
            if not reachable:
                missing = ", ".join(str(dst) for dst in remaining)
                raise nx.NetworkXNoPath(
                    f"No priced broadcast route from {src} to: {missing}"
                )

            dst = min(
                reachable,
                key=lambda node: (
                    distances[node],
                    len(paths[node]),
                    str(node),
                ),
            )

            branch = paths[dst]
            for u, v in zip(branch, branch[1:]):
                installed.add((u, v))
                tree_nodes.add(u)
                tree_nodes.add(v)
            remaining.remove(dst)

        installed = _prune_shared_edges(src, terminals, installed)
        score = (_edge_union_cost(h, installed), len(installed), str(seed))
        if best_score is None or score < best_score:
            best_score = score
            best_edges = installed

    return best_edges or set()


def _paths_from_shared_edges(src, dsts, selected_edges, G):
    """Extract valid source-to-destination paths from the selected edge union."""
    topology_graph = nx.DiGraph()
    topology_graph.add_node(src)

    for u, v in selected_edges:
        topology_graph.add_edge(u, v, cost=_edge_cost(G[u][v]))

    result = {}
    for dst in dict.fromkeys(dsts):
        if dst == src:
            result[dst] = []
            continue

        try:
            node_path = nx.shortest_path(
                topology_graph, src, dst, weight="cost"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise nx.NetworkXNoPath(
                f"Shared broadcast topology does not reach destination {dst}"
            )

        result[dst] = [
            [u, v, G[u][v]]
            for u, v in zip(node_path, node_path[1:])
        ]

    return result


def search_algorithm(src, dsts, G, num_partitions):
    """Build a low-cost shared multicast topology rooted at ``src``."""
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    h = _priced_graph(src, G)

    terminals = list(dict.fromkeys(dst for dst in dsts if dst != src))
    if not terminals:
        selected_edges = set()
    elif len(terminals) <= EXACT_TERMINAL_LIMIT:
        # Exact DP gives the lowest possible physical edge-union cost.
        selected_edges = _exact_shared_edges(src, terminals, h)
    else:
        # Multi-start greedy retains shared-prefix savings at high fanout.
        selected_edges = _multistart_greedy_shared_edges(src, terminals, h)

    paths = _paths_from_shared_edges(src, dsts, selected_edges, G)
    for dst in dsts:
        edge_path = paths[dst]
        for partition in range(num_partitions):
            bc_topology.set_dst_partition_paths(
                dst, partition, list(edge_path)
            )

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]


class BroadCastTopology:
    def __init__(
        self,
        src: str,
        dsts: List[str],
        num_partitions: int = 4,
        paths: Dict[str, SingleDstPath] = None,
    ):
        self.src = src
        self.dsts = dsts
        self.num_partitions = num_partitions

        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {
                dst: {str(i): None for i in range(num_partitions)}
                for dst in dsts
            }

    def set_graph(self):
        """Compatibility hook for callers constructing from existing paths."""
        return self.paths

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
    Default graph with capacity constraints and cost information.

    Edge attributes:
      throughput: aggregate available throughput in Gbps
      cost: transfer cost per GB
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if cost_path is None:
        cost = pd.read_csv(os.path.join(current_dir, "profiles/cost.csv"))
    else:
        cost = pd.read_csv(cost_path)

    if throughput_path is None:
        throughput = pd.read_csv(
            os.path.join(current_dir, "profiles/throughput.csv")
        )
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

    no_cost_pairs = [
        (u, v) for u, v, data in G.edges(data=True)
        if data.get("cost") is None
    ]
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