import networkx as nx
import json
import os
import math
import heapq
import pandas as pd
from typing import Dict, List


def _edge_cost(data):
    value = data.get("cost")
    if value is None:
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def _edge_union_cost(G, edges):
    return sum(_edge_cost(G[u][v]) or 0.0 for u, v in edges)


def _shortest_paths(h, src, dsts):
    paths = {}
    for dst in dsts:
        paths[dst] = nx.dijkstra_path(h, src, dst, weight="cost")
    return paths


def _paths_to_edges(paths):
    result = set()
    for path in paths.values():
        result.update(zip(path, path[1:]))
    return result


def _greedy_tree(h, src, terminals, orders):
    """Bounded fallback for larger terminal sets."""
    best_edges = None
    best_paths = None
    best_cost = float("inf")

    for order in orders:
        used = set()
        paths = {}
        feasible = True

        for dst in order:
            def marginal(u, v, data):
                if (u, v) in used:
                    return 0.0
                c = _edge_cost(data)
                return float("inf") if c is None else c

            try:
                path = nx.dijkstra_path(h, src, dst, weight=marginal)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                feasible = False
                break
            paths[dst] = path
            used.update(zip(path, path[1:]))

        if feasible:
            value = _edge_union_cost(h, used)
            if value < best_cost:
                best_cost = value
                best_edges = used
                best_paths = paths

    return best_edges, best_paths


def _directed_steiner_edges(h, src, terminals):
    """
    Exact directed Steiner DP for a small terminal set.
    dp[mask][v] is represented by a cost dictionary and traceback dictionary.
    """
    terminals = list(dict.fromkeys(terminals))
    k = len(terminals)
    if k == 0:
        return set()
    if k > 7:
        return None

    nodes = list(h.nodes())
    incoming = {v: [] for v in nodes}
    for u, v, data in h.edges(data=True):
        c = _edge_cost(data)
        if c is not None:
            incoming[v].append((u, c))

    full = (1 << k) - 1
    dp = [dict() for _ in range(full + 1)]
    choice = [dict() for _ in range(full + 1)]

    for mask in range(1, full + 1):
        initial = {}
        initial_choice = {}

        if mask & (mask - 1) == 0:
            terminal = terminals[mask.bit_length() - 1]
            initial[terminal] = 0.0
            initial_choice[terminal] = ("terminal",)
        else:
            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if other and sub < other:
                    common = set(dp[sub]).intersection(dp[other])
                    for v in common:
                        value = dp[sub][v] + dp[other][v]
                        if value < initial.get(v, float("inf")):
                            initial[v] = value
                            initial_choice[v] = ("merge", sub)
                sub = (sub - 1) & mask

        # Multi-source reverse Dijkstra computes min_w c(v,w) + initial[w].
        dist = dict(initial)
        trace = dict(initial_choice)
        heap = [(value, v) for v, value in initial.items()]
        heapq.heapify(heap)

        while heap:
            value, v = heapq.heappop(heap)
            if value != dist.get(v):
                continue
            for u, edge_cost in incoming.get(v, []):
                candidate = value + edge_cost
                if candidate < dist.get(u, float("inf")):
                    dist[u] = candidate
                    trace[u] = ("edge", v)
                    heapq.heappush(heap, (candidate, u))

        dp[mask] = dist
        choice[mask] = trace

    if src not in dp[full]:
        return None

    selected = set()
    visiting = set()

    def recover(mask, v):
        state = (mask, v)
        if state in visiting:
            return
        visiting.add(state)
        action = choice[mask][v]
        if action[0] == "edge":
            w = action[1]
            selected.add((v, w))
            recover(mask, w)
        elif action[0] == "merge":
            left = action[1]
            recover(left, v)
            recover(mask ^ left, v)
        visiting.remove(state)

    recover(full, src)
    return selected


def _decomposed_steiner_edges(h, src, terminals):
    """Compose exact local directed Steiner trees over a rooted block-cut tree."""
    terminals = list(dict.fromkeys(terminals))
    if not terminals:
        return set()

    support = h.to_undirected()
    blocks = [set(component) for component in nx.biconnected_components(support)]
    if not blocks:
        return None

    articulation = set(nx.articulation_points(support))
    containing = [i for i, block in enumerate(blocks) if src in block]
    if not containing:
        return None

    block_of = {node: [] for node in support.nodes()}
    for index, block in enumerate(blocks):
        for node in block:
            block_of.setdefault(node, []).append(index)

    root = containing[0]
    parent_block = {root: None}
    parent_vertex = {root: src}
    order = [root]
    queue = [root]

    while queue:
        current = queue.pop(0)
        for vertex in blocks[current].intersection(articulation):
            for neighbor in block_of.get(vertex, []):
                if neighbor not in parent_block:
                    parent_block[neighbor] = current
                    parent_vertex[neighbor] = vertex
                    queue.append(neighbor)
                    order.append(neighbor)

    terminal_blocks = {}
    for terminal in terminals:
        candidates = [i for i in block_of.get(terminal, []) if i in parent_block]
        if not candidates:
            return None
        # Assign articulation destinations to the deepest reachable block.
        terminal_blocks[terminal] = max(
            candidates,
            key=lambda i: sum(1 for _ in nx.shortest_path(
                nx.Graph(), root, i
            )) if False else order.index(i)
        )

    selected = set()
    for index in order:
        block = blocks[index]
        local_source = parent_vertex[index]
        local_terminals = {
            terminal for terminal, owner in terminal_blocks.items()
            if owner == index
        }

        for child in order:
            if parent_block.get(child) == index:
                local_terminals.add(parent_vertex[child])

        if not local_terminals:
            continue
        if local_source not in block:
            return None

        local_graph = h.subgraph(block).copy()
        local_edges = _directed_steiner_edges(
            local_graph, local_source, list(local_terminals)
        )
        if local_edges is None:
            orders = [list(local_terminals)]
            local_edges, _ = _greedy_tree(
                local_graph, local_source, list(local_terminals), orders
            )
        if local_edges is None:
            return None
        selected.update(local_edges)

    candidate = h.edge_subgraph(selected).copy()
    if any(
        terminal not in candidate or
        not nx.has_path(candidate, src, terminal)
        for terminal in terminals
    ):
        return None
    return selected


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Ignore unusable cost edges rather than allowing Dijkstra comparisons with None.
    invalid = [(u, v) for u, v, data in h.edges(data=True) if _edge_cost(data) is None]
    h.remove_edges_from(invalid)

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    unique_dsts = list(dict.fromkeys(dsts))

    try:
        baseline_paths = _shortest_paths(h, src, unique_dsts)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Preserve the original behavior for malformed/unreachable evaluator inputs.
        baseline_paths = {}
        for dst in unique_dsts:
            try:
                baseline_paths[dst] = nx.dijkstra_path(h, src, dst, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                baseline_paths[dst] = []

    baseline_edges = _paths_to_edges(baseline_paths)
    selected_edges = baseline_edges

    steiner_edges = _decomposed_steiner_edges(h, src, unique_dsts)
    if steiner_edges is not None:
        try:
            candidate_graph = h.edge_subgraph(steiner_edges).copy()
            reachable = all(
                dst in candidate_graph and nx.has_path(candidate_graph, src, dst)
                for dst in unique_dsts
            )
        except (nx.NetworkXError, nx.NodeNotFound):
            reachable = False

        if reachable and _edge_union_cost(h, steiner_edges) <= _edge_union_cost(h, baseline_edges):
            selected_edges = steiner_edges

    tree_graph = nx.DiGraph()
    tree_graph.add_nodes_from(h.nodes())
    for u, v in selected_edges:
        tree_graph.add_edge(u, v, **G[u][v])

    final_paths = {}
    for dst in unique_dsts:
        try:
            final_paths[dst] = nx.dijkstra_path(tree_graph, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            final_paths[dst] = baseline_paths.get(dst, [])

    for dst in dsts:
        path = final_paths.get(dst, [])
        edges = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
        for partition in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(dst, partition, list(edges))

    return bc_topology


class SingleDstPath(Dict):
    partition: int
    edges: List[List]


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
        self.paths[dst][str(partition)] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)


def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
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