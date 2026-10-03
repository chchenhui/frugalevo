# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Metric-closure relay arborescence broadcast.

    Computes all-pairs shortest paths over h restricted to {src} ∪ dsts,
    builds a metric-closure digraph C whose edge weights are pairwise
    shortest-path costs, then computes a minimum-cost directed arborescence
    rooted at src on C. Each destination's delivery chain is the
    concatenation of closure-edge paths from src along the arborescence,
    permitting cheap dst→dst relay hops instead of forcing all traffic
    through src's egress. Every (dst, partition) receives a contiguous
    src→dst edge chain in G. Falls back to per-destination Dijkstra on any
    exception.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    def emit(dst, pairs):
        for s, t in pairs:
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    def dijkstra_pairs(dst):
        p = nx.dijkstra_path(h, src, dst, weight="cost")
        return [(p[i], p[i + 1]) for i in range(len(p) - 1)]

    routes = None
    try:
        """Exact directed-Steiner ILP via scipy.optimize.milp (HiGHS).

        Variables: binary y_e per directed edge of h (edge used, paid once —
        the true shared-edge multicast objective) plus binary flow variables
        f_{e,d}: one unit of src→dst flow per reachable dst d, with f_{e,d} ≤ y_e.
        Flow conservation (net outflow = +1 at src, -1 at dst, 0 elsewhere)
        plus the linking constraints guarantee every dst lies on a directed
        src-path in the y-subgraph (flow decomposition), so no reverse-flow
        family is needed — critically, h has no in-edges to src, so any
        dst→src flow family would be infeasible. Per-dst chains are Dijkstra
        paths restricted to the optimal y-subgraph. Falls back to the
        incumbent per-dst Dijkstra on any exception or solver failure.
        """
        import numpy as np
        from scipy import sparse
        from scipy.optimize import milp, LinearConstraint, Bounds

        edges = list(h.edges())
        nE = len(edges)
        eidx = {e: i for i, e in enumerate(edges)}
        costs = np.array([
            float(h[u][v]["cost"]) if h[u][v].get("cost") is not None else 1e6
            for u, v in edges
        ])

        reach = [d for d in dsts if d != src and nx.has_path(h, src, d)]
        D = len(reach)
        if D == 0:
            raise ValueError("no reachable dsts")

        # Variable layout: y[0:nE], f_{d,e} at nE + d*nE + e.
        nvar = nE * (1 + D)

        def vid_f(d, i):
            return nE + d * nE + i

        # Equality constraints: net outflow = b_v for each (dst-flow, node).
        nodes = list(h.nodes())
        rows, cols, vals, b = [], [], [], []
        ri = 0
        for d in range(D):
            for v in nodes:
                for e in h.out_edges(v):
                    rows.append(ri); cols.append(vid_f(d, eidx[e])); vals.append(1.0)
                for e in h.in_edges(v):
                    rows.append(ri); cols.append(vid_f(d, eidx[e])); vals.append(-1.0)
                b.append(1.0 if v == src else (-1.0 if v == reach[d] else 0.0))
                ri += 1
        A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(ri, nvar))

        # Linking constraints: f_{e,d} - y_e <= 0.
        rows2, cols2, vals2 = [], [], []
        ri2 = 0
        for d in range(D):
            for i in range(nE):
                rows2 += [ri2, ri2]
                cols2 += [vid_f(d, i), i]
                vals2 += [1.0, -1.0]
                ri2 += 1
        A_ub = sparse.csr_matrix((vals2, (rows2, cols2)), shape=(ri2, nvar))

        c_obj = np.zeros(nvar)
        c_obj[:nE] = costs
        constraints = [
            LinearConstraint(A_eq, np.array(b, dtype=float), np.array(b, dtype=float)),
            LinearConstraint(A_ub, -np.inf * np.ones(ri2), np.zeros(ri2)),
        ]
        res = milp(
            c=c_obj,
            constraints=constraints,
            integrality=np.ones(nvar),
            bounds=Bounds(0, 1),
            options={"time_limit": 60},
        )
        if res.x is None or not np.all(np.isfinite(res.x)):
            raise RuntimeError("milp produced no solution")

        # Extract optimal shared-edge subgraph and per-dst chains on it.
        y = res.x[:nE] > 0.5
        sub = nx.DiGraph()
        sub.add_nodes_from(h.nodes())
        for i, (u, v) in enumerate(edges):
            if y[i]:
                sub.add_edge(u, v, **h[u][v])

        routes = {}
        for dst in dsts:
            try:
                p = nx.dijkstra_path(sub, src, dst, weight="cost")
                routes[dst] = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
            except Exception:
                routes[dst] = dijkstra_pairs(dst)
    except Exception:
        routes = None

    if routes is None:
        for dst in dsts:
            emit(dst, dijkstra_pairs(dst))
        return bc_topology

    for dst in dsts:
        emit(dst, routes[dst])

    return bc_topology


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
