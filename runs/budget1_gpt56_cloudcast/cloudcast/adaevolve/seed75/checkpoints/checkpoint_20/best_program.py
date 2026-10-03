# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Solve a fixed-charge directed broadcast Steiner problem as a MILP.

    Binary edge variables charge each priced inter-cloud link once, while one
    continuous unit-flow commodity per destination guarantees that every
    receiver is connected from the source through activated links.
    """
    import math
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix

    topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(d for d in dsts if d != src))
    if not terminals:
        for dst in dsts:
            for partition in range(num_partitions):
                topology.set_dst_partition_paths(dst, partition, [])
        return topology

    # Missing prices are unusable: NetworkX otherwise silently treats them as
    # unit-weight links, which is inappropriate for monetary optimization.
    h = G.copy()
    h.remove_edges_from(
        list(h.in_edges(src)) + list(nx.selfloop_edges(h)) +
        [(u, v) for u, v, data in h.edges(data=True)
         if data.get("cost") is None or not math.isfinite(float(data["cost"]))]
    )
    reachable = nx.descendants(h, src) | {src}
    useful = {src} | set(terminals)
    for terminal in terminals:
        if terminal not in reachable:
            raise nx.NetworkXNoPath("No priced route from %s to %s" % (src, terminal))
        useful.update(nx.ancestors(h, terminal) & reachable)
    h = h.subgraph(useful).copy()

    nodes = list(h.nodes)
    node_id = {node: i for i, node in enumerate(nodes)}
    arcs = list(h.edges())
    costs = np.array([float(h[u][v]["cost"]) for u, v in arcs])
    n, m, k = len(nodes), len(arcs), len(terminals)

    # Variables are x[e] followed by f[terminal,e].  The conservation rows
    # enforce one source-to-terminal transfer for every destination; f <= x
    # permits a single paid edge to broadcast data for many commodities.
    rows, cols, data = [], [], []
    rhs = np.zeros(k * n)
    for t, terminal in enumerate(terminals):
        base = t * n
        rhs[base + node_id[src]] = 1.0
        rhs[base + node_id[terminal]] = -1.0
        flow_base = m + t * m
        for e, (u, v) in enumerate(arcs):
            rows.extend((base + node_id[u], base + node_id[v]))
            cols.extend((flow_base + e, flow_base + e))
            data.extend((1.0, -1.0))

    # f[t,e] - x[e] <= 0.  This is the fixed-charge activation coupling.
    link_start = k * n
    for t in range(k):
        flow_base = m + t * m
        for e in range(m):
            row = link_start + t * m + e
            rows.extend((row, row))
            cols.extend((e, flow_base + e))
            data.extend((-1.0, 1.0))

    total_rows = k * n + k * m
    matrix = coo_matrix(
        (data, (rows, cols)), shape=(total_rows, m + k * m)
    ).tocsr()
    lower = np.concatenate((rhs, np.full(k * m, -np.inf)))
    upper = np.concatenate((rhs, np.zeros(k * m)))
    objective = np.concatenate((costs, np.zeros(k * m)))
    integrality = np.concatenate((np.ones(m), np.zeros(k * m)))

    selected = None
    try:
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(np.zeros(m + k * m), np.ones(m + k * m)),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"time_limit": 120},
        )
        if result.x is not None:
            candidate = {arcs[e] for e in range(m) if result.x[e] > 0.5}
            candidate_graph = nx.DiGraph()
            candidate_graph.add_edges_from(candidate)
            if all(nx.has_path(candidate_graph, src, terminal)
                   for terminal in terminals):
                # Retain only useful source-to-terminal routes.  This removes
                # any harmless activated cycles from an interrupted MILP solve.
                selected = set()
                for terminal in terminals:
                    path = nx.shortest_path(candidate_graph, src, terminal,
                                            weight=lambda u, v, d: h[u][v]["cost"])
                    selected.update(zip(path, path[1:]))
    except Exception:
        selected = None

    # A robust feasible fallback also protects callers without a compatible
    # SciPy HiGHS installation or instances stopped before an incumbent exists.
    if selected is None:
        selected = set()
        for terminal in terminals:
            path = nx.shortest_path(h, src, terminal, weight="cost")
            selected.update(zip(path, path[1:]))

    tree = nx.DiGraph()
    tree.add_edges_from(selected)
    for dst in dsts:
        if dst == src:
            edges = []
        else:
            path = nx.shortest_path(tree, src, dst, weight=lambda u, v, d: G[u][v]["cost"])
            edges = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
        for partition in range(num_partitions):
            topology.set_dst_partition_paths(dst, partition, list(edges))
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
