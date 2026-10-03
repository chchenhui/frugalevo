import heapq
import os
from typing import Dict, List

import networkx as nx
import pandas as pd


def search_algorithm(src, dsts, G, num_partitions):
    """Find a low union-cost directed broadcast topology."""
    topology = BroadCastTopology(src, dsts, num_partitions)
    unique_dsts = list(dict.fromkeys(dsts))
    if not unique_dsts or src not in G:
        return topology

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    invalid = []
    for u, v, data in h.edges(data=True):
        try:
            if data.get("cost") is None or float(data["cost"]) < 0:
                invalid.append((u, v))
        except (TypeError, ValueError):
            invalid.append((u, v))
    h.remove_edges_from(invalid)

    def shortest_length(dst):
        try:
            return nx.dijkstra_path_length(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    reachable = [d for d in unique_dsts if shortest_length(d) != float("inf")]
    terminals = [d for d in reachable if d != src]

    def edge_path(nodes):
        return tuple(zip(nodes, nodes[1:]))

    def project_and_paths(active_edges):
        """Remove unused selected edges by routing only within their union."""
        sub = nx.DiGraph()
        for u, v in active_edges:
            if h.has_edge(u, v):
                sub.add_edge(u, v, **h[u][v])
        paths = {}
        if src not in sub:
            if src in reachable:
                paths[src] = tuple()
            return paths
        try:
            _, node_paths = nx.single_source_dijkstra(sub, src, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            node_paths = {}
        for dst in reachable:
            if dst == src:
                paths[dst] = tuple()
            elif dst in node_paths:
                paths[dst] = edge_path(node_paths[dst])
        return paths

    def exact_subset_steiner():
        """Exact directed Steiner DP for a small terminal set."""
        k = len(terminals)
        if k == 0:
            return {src: tuple()}
        if k > 10:
            return None

        nodes = list(h.nodes())
        reverse = h.reverse(copy=False)
        full = (1 << k) - 1
        dp = {}
        choice = {}

        # Singleton values are all directed distances from every node to terminal.
        for i, terminal in enumerate(terminals):
            mask = 1 << i
            try:
                lengths, rev_paths = nx.single_source_dijkstra(
                    reverse, terminal, weight="cost"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
            values = {}
            picks = {}
            for node, value in lengths.items():
                values[node] = float(value)
                backwards = rev_paths[node]       # terminal ... node in reverse graph
                forward = list(reversed(backwards))
                picks[node] = ("base", edge_path(forward))
            dp[mask] = values
            choice[mask] = picks

        # For each subset, first combine branches at a node, then move the
        # common root backward through directed incoming edges.
        for mask in range(1, full + 1):
            if mask in dp:
                continue
            values = {}
            picks = {}
            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if other and sub < other:
                    left, right = dp[sub], dp[other]
                    for node in set(left).intersection(right):
                        value = left[node] + right[node]
                        if value < values.get(node, float("inf")):
                            values[node] = value
                            picks[node] = ("split", sub, other)
                sub = (sub - 1) & mask

            if not values:
                return None

            heap = []
            serial = 0
            for node, value in values.items():
                heapq.heappush(heap, (value, serial, node))
                serial += 1

            while heap:
                value, _, node = heapq.heappop(heap)
                if value != values.get(node):
                    continue
                for predecessor in h.predecessors(node):
                    candidate = value + float(h[predecessor][node]["cost"])
                    if candidate + 1e-12 < values.get(predecessor, float("inf")):
                        values[predecessor] = candidate
                        picks[predecessor] = ("edge", node)
                        heapq.heappush(heap, (candidate, serial, predecessor))
                        serial += 1

            dp[mask] = values
            choice[mask] = picks

        if src not in dp[full]:
            return None

        active = set()
        visiting = set()

        def recover(mask, node):
            key = (mask, node)
            if key in visiting:
                return
            visiting.add(key)
            item = choice[mask].get(node)
            if item is None:
                visiting.remove(key)
                return
            if item[0] == "base":
                active.update(item[1])
            elif item[0] == "split":
                recover(item[1], node)
                recover(item[2], node)
            else:
                nxt = item[1]
                active.add((node, nxt))
                recover(mask, nxt)
            visiting.remove(key)

        recover(full, src)
        paths = project_and_paths(active)
        return paths if all(dst in paths for dst in reachable) else None

    def greedy_fallback():
        """Bounded union-cost coordinate descent for larger terminal sets."""
        distances = {d: shortest_length(d) for d in reachable}
        orders = [
            sorted(reachable, key=lambda d: (distances[d], str(d))),
            sorted(reachable, key=lambda d: (-distances[d], str(d))),
            list(reachable),
        ]
        best_cost = float("inf")
        best_paths = {}

        for order in orders:
            counts = {}
            paths = {}

            def marginal(u, v, data):
                return 0.0 if counts.get((u, v), 0) else float(data["cost"])

            for dst in order:
                try:
                    nodes = nx.dijkstra_path(h, src, dst, weight=marginal)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                route = edge_path(nodes)
                paths[dst] = route
                for edge in route:
                    counts[edge] = counts.get(edge, 0) + 1

            for _ in range(3):
                changed = False
                for dst in order:
                    old = paths.get(dst)
                    if old is None:
                        continue
                    for edge in old:
                        counts[edge] -= 1
                        if counts[edge] == 0:
                            del counts[edge]
                    try:
                        nodes = nx.dijkstra_path(h, src, dst, weight=marginal)
                        new = edge_path(nodes)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        new = old
                    changed = changed or new != old
                    paths[dst] = new
                    for edge in new:
                        counts[edge] = counts.get(edge, 0) + 1
                if not changed:
                    break

            projected = project_and_paths(counts)
            if not all(dst in projected for dst in reachable):
                projected = paths
            cost = sum(
                float(h[u][v]["cost"])
                for route in projected.values()
                for u, v in route
            )
            union_cost = sum(
                float(h[u][v]["cost"])
                for u, v in {e for route in projected.values() for e in route}
            )
            if union_cost < best_cost:
                best_cost = union_cost
                best_paths = projected

        return best_paths

    best_paths = exact_subset_steiner()
    if best_paths is None:
        best_paths = greedy_fallback()

    for dst in unique_dsts:
        route = best_paths.get(dst)
        if route is None:
            continue
        edges = [[u, v, G[u][v]] for u, v in route]
        for partition in range(topology.num_partitions):
            topology.set_dst_partition_paths(dst, partition, list(edges))
    return topology


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
    cost = pd.read_csv(
        os.path.join(current_dir, "profiles/cost.csv")
        if cost_path is None else cost_path
    )
    throughput = pd.read_csv(
        os.path.join(current_dir, "profiles/throughput.csv")
        if throughput_path is None else throughput_path
    )

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


def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    return search_algorithm(src, dsts, G, num_partitions)