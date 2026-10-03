import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Build and locally improve a minimum union-cost directed broadcast topology."""
    topology = BroadCastTopology(src, dsts, num_partitions)

    # Costless/missing-cost edges must not accidentally receive NetworkX's
    # default unit weight.  Source incoming edges and loops cannot help a
    # source-rooted broadcast.
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bad_edges = []
    for u, v, data in h.edges(data=True):
        cost = data.get("cost")
        try:
            if cost is None or float(cost) < 0:
                bad_edges.append((u, v))
        except (TypeError, ValueError):
            bad_edges.append((u, v))
    h.remove_edges_from(bad_edges)

    # BroadCastTopology is keyed by destination, so duplicate names cannot
    # represent distinct output obligations.
    unique_dsts = list(dict.fromkeys(dsts))
    if not unique_dsts:
        return topology

    def base_distance(dst):
        try:
            return nx.dijkstra_path_length(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    distances = {dst: base_distance(dst) for dst in unique_dsts}
    reachable = [dst for dst in unique_dsts if distances[dst] != float("inf")]

    # Preserve the parent behavior for ordinary reachable instances.  Empty
    # paths are valid for src itself; unreachable destinations retain None.
    if not reachable:
        return topology

    ascending = sorted(reachable, key=lambda d: (distances[d], str(d)))
    descending = list(reversed(ascending))
    input_order = list(reachable)
    orders = []
    for order in (ascending, descending, input_order):
        if order not in orders:
            orders.append(order)

    def route_for_order(order):
        counts = {}
        paths = {}

        def marginal(u, v, data):
            if counts.get((u, v), 0) > 0:
                return 0.0
            return float(data["cost"])

        for dst in order:
            try:
                node_path = nx.dijkstra_path(h, src, dst, weight=marginal)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            edge_path = list(zip(node_path, node_path[1:]))
            paths[dst] = edge_path
            for edge in edge_path:
                counts[edge] = counts.get(edge, 0) + 1

        # Coordinate descent: with all other routes held fixed, Dijkstra with
        # their edges free gives the best union-cost route for this destination.
        for _ in range(3):
            changed = False
            for dst in order:
                old_path = paths.get(dst)
                if old_path is None:
                    continue
                for edge in old_path:
                    counts[edge] -= 1
                    if counts[edge] == 0:
                        del counts[edge]

                try:
                    node_path = nx.dijkstra_path(h, src, dst, weight=marginal)
                    new_path = list(zip(node_path, node_path[1:]))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    new_path = old_path

                if new_path != old_path:
                    changed = True
                paths[dst] = new_path
                for edge in new_path:
                    counts[edge] = counts.get(edge, 0) + 1
            if not changed:
                break

        union_cost = sum(float(h[u][v]["cost"]) for u, v in counts)
        return union_cost, paths

    best_cost = float("inf")
    best_paths = {}
    for order in orders:
        cost, paths = route_for_order(order)
        if cost < best_cost:
            best_cost = cost
            best_paths = paths

    for dst in unique_dsts:
        edge_pairs = best_paths.get(dst)
        if edge_pairs is None:
            continue
        edges = [[u, v, G[u][v]] for u, v in edge_pairs]
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
        partition = str(partition)
        self.paths[dst][partition] = paths

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

    no_cost_pairs = []
    for edge in G.edges.data():
        edge_src, edge_dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((edge_src, edge_dst))
    print("Unable to get costs for: ", no_cost_pairs)

    return G


def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    return search_algorithm(src, dsts, G, num_partitions)