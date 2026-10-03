# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Solve shared broadcast routing as a binary directed-Steiner MILP.

    Edge variables pay once for each activated network link, while a binary
    flow commodity for every destination enforces a source-to-destination
    route using only activated links.  The resulting shared topology is then
    reused by every partition.  A marginal-cost greedy tree remains available
    only when SciPy has no feasible MILP incumbent.
    """
    import math
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(d for d in dsts if d != src))

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    h.remove_edges_from([
        (u, v) for u, v, data in h.edges(data=True)
        if data.get("cost") is None
        or not math.isfinite(float(data["cost"]))
        or float(data["cost"]) < 0
    ])

    reachable = set(nx.descendants(h, src)) | {src} if src in h else set()
    terminals = [dst for dst in terminals if dst in reachable]

    def install(routes):
        for dst in dsts:
            route = routes.get(dst, [])
            for partition in range(num_partitions):
                topology.set_dst_partition_paths(dst, partition, list(route))

    if not terminals:
        install({dst: [] for dst in dsts})
        return topology

    # Vertices outside both the source-reachable region and a terminal's
    # reverse-reachable region cannot occur in a useful multicast topology.
    useful = set()
    for dst in terminals:
        useful.add(dst)
        useful.update(nx.ancestors(h, dst))
    h = h.subgraph(reachable & useful).copy()
    edges = list(h.edges())
    nodes = list(h.nodes())
    if src not in h or not edges:
        install({})
        return topology

    node_id = {node: i for i, node in enumerate(nodes)}
    edge_count, commodity_count = len(edges), len(terminals)
    variable_count = edge_count * (commodity_count + 1)

    # Objective is the union-of-links broadcast cost.  The tiny throughput
    # tie-break selects more robust links only when primary costs are tied.
    costs = np.zeros(variable_count)
    for e, (u, v) in enumerate(edges):
        data = h[u][v]
        throughput = float(data.get("throughput", 0) or 0)
        costs[e] = float(data["cost"]) + 1e-10 / max(throughput, 1e-12)

    rows, cols, values, lower, upper = [], [], [], [], []
    row = 0

    # Commodity flow conservation: outflow - inflow is +1 at source and -1
    # at that commodity's destination.
    for k, dst in enumerate(terminals):
        offset = edge_count * (k + 1)
        for node in nodes:
            for e, (u, v) in enumerate(edges):
                if u == node:
                    rows.append(row)
                    cols.append(offset + e)
                    values.append(1)
                if v == node:
                    rows.append(row)
                    cols.append(offset + e)
                    values.append(-1)
            demand = 1 if node == src else -1 if node == dst else 0
            lower.append(demand)
            upper.append(demand)
            row += 1

    # A commodity can traverse an edge only if the broadcast activates it.
    for k in range(commodity_count):
        offset = edge_count * (k + 1)
        for e in range(edge_count):
            rows.extend((row, row))
            cols.extend((offset + e, e))
            values.extend((1, -1))
            lower.append(-np.inf)
            upper.append(0)
            row += 1

    matrix = coo_matrix((values, (rows, cols)), shape=(row, variable_count)).tocsr()
    result = milp(
        c=costs,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options={"time_limit": 120, "presolve": True},
    )

    routes = {}
    if result.x is not None:
        # Extract each explicitly selected commodity path.  This avoids keeping
        # zero-cost cycles or unused activated links in the returned topology.
        for k, dst in enumerate(terminals):
            offset = edge_count * (k + 1)
            path_graph = nx.DiGraph()
            for e, (u, v) in enumerate(edges):
                if result.x[offset + e] > 0.5:
                    path_graph.add_edge(u, v)
            try:
                path = nx.shortest_path(path_graph, src, dst)
                routes[dst] = [
                    [u, v, G[u][v]] for u, v in zip(path, path[1:])
                ]
            except nx.NetworkXNoPath:
                routes = {}
                break

    if len(routes) != len(terminals):
        # Feasible fallback for installations where the solver is interrupted
        # before producing an incumbent.
        selected, reached, pending = set(), {src}, set(terminals)
        while pending:
            def marginal(u, v, data):
                return 0 if (u, v) in selected else float(data["cost"])

            distances, paths = nx.multi_source_dijkstra(h, reached, weight=marginal)
            candidates = [dst for dst in pending if dst in distances]
            if not candidates:
                break
            dst = min(candidates, key=distances.get)
            path = paths[dst]
            selected.update(zip(path, path[1:]))
            reached.update(path)
            pending.difference_update(reached)

        tree = nx.DiGraph()
        tree.add_edges_from(selected)
        routes = {}
        for dst in terminals:
            try:
                path = nx.shortest_path(tree, src, dst)
                routes[dst] = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
            except nx.NetworkXNoPath:
                routes[dst] = []

    routes[src] = []
    install(routes)
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
