import networkx as nx
import os
import heapq
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Minimum-cost directed Steiner broadcast with throughput-aware cost ties."""
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)))
    h.remove_edges_from(list(nx.selfloop_edges(h)))
    h.remove_edges_from([
        (u, v) for u, v, data in h.edges(data=True)
        if data.get("cost") is None
    ])

    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(dsts))
    if not terminals:
        return topology

    def edge_delay(data):
        throughput = data.get("throughput")
        try:
            throughput = float(throughput)
        except (TypeError, ValueError):
            throughput = 0.0
        return 1.0 / max(throughput, 1e-12)

    def greedy_paths():
        used = set()
        result = {}
        distances = {}
        for dst in terminals:
            try:
                distances[dst] = nx.shortest_path_length(
                    h, src, dst, weight="cost"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distances[dst] = float("inf")

        for dst in sorted(terminals, key=lambda d: (distances[d], str(d))):
            def marginal(u, v, data):
                return 0.0 if (u, v) in used else float(data["cost"])

            try:
                path = nx.dijkstra_path(h, src, dst, weight=marginal)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path = nx.dijkstra_path(h, src, dst, weight="cost")
            result[dst] = path
            used.update(zip(path, path[1:]))
        return result

    paths = None

    # A modest increase remains practical for typical cloud-region graphs while
    # retaining a strict bounded exponential search.
    if len(terminals) <= 11 and src in h and all(dst in h for dst in terminals):
        nodes = list(h.nodes)
        pos = {node: i for i, node in enumerate(nodes)}
        terminal_bit = {node: 1 << i for i, node in enumerate(terminals)}
        n_nodes = len(nodes)
        size = 1 << len(terminals)
        inf = (float("inf"), float("inf"))

        dp = [[inf] * n_nodes for _ in range(size)]
        choice = [[None] * n_nodes for _ in range(size)]

        for dst in terminals:
            dp[terminal_bit[dst]][pos[dst]] = (0.0, 0.0)

        reverse_edges = [[] for _ in range(n_nodes)]
        for u, v, data in h.edges(data=True):
            reverse_edges[pos[v]].append((
                pos[u],
                float(data["cost"]),
                edge_delay(data),
                v,
            ))

        for mask in range(1, size):
            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if sub < other:
                    for i in range(n_nodes):
                        left, right = dp[sub][i], dp[other][i]
                        candidate = (left[0] + right[0], left[1] + right[1])
                        if candidate < dp[mask][i]:
                            dp[mask][i] = candidate
                            choice[mask][i] = ("merge", sub)
                sub = (sub - 1) & mask

            heap = [
                (dp[mask][i], i)
                for i in range(n_nodes)
                if dp[mask][i][0] < float("inf")
            ]
            heapq.heapify(heap)

            while heap:
                label, vi = heapq.heappop(heap)
                if label != dp[mask][vi]:
                    continue
                for ui, cost, delay, successor in reverse_edges[vi]:
                    candidate = (label[0] + cost, label[1] + delay)
                    if candidate < dp[mask][ui]:
                        dp[mask][ui] = candidate
                        choice[mask][ui] = ("edge", successor)
                        heapq.heappush(heap, (candidate, ui))

        full = size - 1
        if dp[full][pos[src]][0] < float("inf"):
            tree_edges = set()

            def recover(mask, node):
                item = choice[mask][pos[node]]
                if item is None:
                    return
                if item[0] == "merge":
                    recover(item[1], node)
                    recover(mask ^ item[1], node)
                else:
                    nxt = item[1]
                    tree_edges.add((node, nxt))
                    recover(mask, nxt)

            recover(full, src)
            tree = nx.DiGraph()
            tree.add_edges_from(tree_edges)

            if all(
                dst in tree and nx.has_path(tree, src, dst)
                for dst in terminals
            ):
                paths = {
                    dst: nx.shortest_path(tree, src, dst)
                    for dst in terminals
                }

    if paths is None:
        paths = greedy_paths()

    for dst in terminals:
        records = [
            [u, v, G[u][v]]
            for u, v in zip(paths[dst], paths[dst][1:])
        ]
        for partition in range(topology.num_partitions):
            for record in records:
                topology.append_dst_partition_path(dst, partition, record)

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
        (u, v)
        for u, v, data in G.edges(data=True)
        if data.get("cost") is None
    ]
    print("Unable to get costs for: ", no_cost_pairs)
    return G


def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    return search_algorithm(src, dsts, G, num_partitions)