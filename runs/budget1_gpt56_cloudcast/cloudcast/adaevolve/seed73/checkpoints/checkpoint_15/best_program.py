# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Solve a directed shared-edge broadcast Steiner MILP, with greedy fallback.

    A binary variable installs each network edge once.  One continuous unit-flow
    commodity per reachable destination is constrained to use installed edges,
    so the objective charges a shared relay link only once even when it serves
    many destinations.  The resulting selected subgraph is converted into one
    valid source-to-destination path per receiver and reused by all partitions.
    """
    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(d for d in dsts if d != src))

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    h.remove_edges_from([
        (u, v) for u, v, data in h.edges(data=True)
        if data.get("cost") is None
        or data.get("cost") != data.get("cost")
        or data.get("cost") < 0
    ])

    if not terminals:
        for dst in dsts:
            for part in range(num_partitions):
                topology.set_dst_partition_paths(dst, part, [])
        return topology

    # Do not make an infeasible MILP for destinations that cannot be reached.
    reachable = []
    for dst in terminals:
        try:
            if src in h and dst in h and nx.has_path(h, src, dst):
                reachable.append(dst)
        except nx.NodeNotFound:
            pass

    tree_edges = set()
    edges = list(h.edges())
    m, k = len(edges), len(reachable)

    # Exact directed-Steiner formulation:
    # y[e] is binary; f[t,e] carries terminal t's unit demand.  f <= y lets
    # all terminal commodities share the same installed edge at no extra cost.
    if m and k:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix

            nodes = list(h.nodes())
            node_index = {node: i for i, node in enumerate(nodes)}
            n = len(nodes)
            variables = m * (k + 1)
            rows = k * n + k * m
            A = lil_matrix((rows, variables), dtype=float)
            lower = np.empty(rows)
            upper = np.empty(rows)
            row = 0

            for ti, terminal in enumerate(reachable):
                offset = m + ti * m
                for node in nodes:
                    for ei, (u, v) in enumerate(edges):
                        if u == node:
                            A[row, offset + ei] += 1.0
                        if v == node:
                            A[row, offset + ei] -= 1.0
                    demand = 1.0 if node == src else -1.0 if node == terminal else 0.0
                    lower[row] = demand
                    upper[row] = demand
                    row += 1

                for ei in range(m):
                    A[row, offset + ei] = 1.0
                    A[row, ei] = -1.0
                    lower[row] = -np.inf
                    upper[row] = 0.0
                    row += 1

            # A tiny throughput preference only breaks equal-cost solutions.
            objective = np.array([
                float(h[u][v]["cost"]) +
                1e-10 / max(float(h[u][v].get("throughput", 1.0)), 1e-12)
                for u, v in edges
            ] + [0.0] * (k * m))

            result = milp(
                c=objective,
                integrality=np.array([1] * m + [0] * (k * m)),
                bounds=Bounds(np.zeros(variables), np.ones(variables)),
                constraints=LinearConstraint(A.tocsr(), lower, upper),
                options={"time_limit": 90.0, "mip_rel_gap": 0.0},
            )

            # HiGHS can return a feasible incumbent when its time limit expires.
            if result.x is not None:
                selected = {
                    edge for ei, edge in enumerate(edges)
                    if result.x[ei] >= 0.5
                }
                candidate = nx.DiGraph()
                candidate.add_edges_from(selected)
                if all(nx.has_path(candidate, src, dst) for dst in reachable):
                    tree_edges = selected
        except (ImportError, ValueError, RuntimeError):
            pass

    # Robust fallback if SciPy is unavailable, the model times out before an
    # incumbent, or a solver result cannot be reconstructed into valid paths.
    if not tree_edges:
        reached, remaining = {src}, set(reachable)
        while remaining:
            best = None
            for dst in remaining:
                try:
                    value, path = nx.multi_source_dijkstra(
                        h, list(reached), dst, weight="cost"
                    )
                    if best is None or value < best[0]:
                        best = (value, dst, path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
            if best is None:
                break
            _, dst, path = best
            tree_edges.update(zip(path, path[1:]))
            reached.update(path)
            remaining.remove(dst)

    tree = nx.DiGraph()
    for u, v in tree_edges:
        tree.add_edge(u, v, **G[u][v])

    for dst in dsts:
        try:
            path = [src] if dst == src else nx.dijkstra_path(
                tree, src, dst, weight="cost"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                path = nx.dijkstra_path(h, src, dst, weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        path_edges = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
        for part in range(num_partitions):
            topology.set_dst_partition_paths(dst, part, list(path_edges))

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
