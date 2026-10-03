import networkx as nx
import json
import os
import itertools
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Select the cheapest shared-edge tree from a bounded order portfolio.

    Each candidate attaches destinations in a prescribed order.  Links selected
    earlier in that candidate have zero marginal cost, so a route may reuse an
    already broadcast prefix.  Candidates are compared by the cost of their
    unique directed edge union and the best valid candidate is serialized in
    the established BroadCastTopology format.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)))
    h.remove_edges_from(list(nx.selfloop_edges(h)))
    h.remove_edges_from([
        (u, v) for u, v, data in h.edges(data=True)
        if data.get("cost") is None
    ])

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    if not dsts:
        return bc_topology

    # Preserve caller order while avoiding redundant work for duplicate names.
    unique_dsts = list(dict.fromkeys(dsts))

    def base_distance(dst):
        try:
            return nx.shortest_path_length(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    base_order = sorted(unique_dsts, key=lambda d: (base_distance(d), str(d)))

    def build_for_order(order):
        used = set()
        destination_paths = {}

        for dst in order:
            def marginal_weight(u, v, data):
                if (u, v) in used:
                    return 0.0
                value = data.get("cost")
                return float(value) if value is not None else float("inf")

            try:
                path = nx.dijkstra_path(h, src, dst, weight=marginal_weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None, float("inf")

            destination_paths[dst] = path
            used.update(zip(path, path[1:]))

        cost = sum(float(h[u][v]["cost"]) for u, v in used)
        return destination_paths, cost

    # Generate deterministic, structurally distinct attachment orders.
    orders = []
    seen_orders = set()

    def add_order(order):
        key = tuple(order)
        if key not in seen_orders:
            seen_orders.add(key)
            orders.append(list(order))

    add_order(base_order)
    add_order(list(reversed(base_order)))

    n = len(unique_dsts)
    if n <= 5:
        for order in itertools.permutations(unique_dsts):
            add_order(order)
    else:
        for sequence in (base_order, list(reversed(base_order))):
            for shift in range(n):
                add_order(sequence[shift:] + sequence[:shift])

        # A bounded lexicographic tail gives additional alternatives without
        # random behavior or an unbounded factorial search.
        canonical = sorted(unique_dsts, key=str)
        for order in itertools.islice(itertools.permutations(canonical), 120):
            add_order(order)
            if len(orders) >= 120:
                break

    orders = orders[:120]

    best_paths = None
    best_cost = float("inf")
    for order in orders:
        paths, cost = build_for_order(order)
        if paths is not None and cost < best_cost:
            best_paths = paths
            best_cost = cost

    # Match ordinary shortest-path behavior as a final robust fallback.
    if best_paths is None:
        best_paths = {}
        for dst in unique_dsts:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
            best_paths[dst] = path

    for dst in unique_dsts:
        path = best_paths[dst]
        records = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
        for partition in range(bc_topology.num_partitions):
            for record in records:
                bc_topology.append_dst_partition_path(dst, partition, record)

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


def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    return search_algorithm(src, dsts, G, num_partitions)