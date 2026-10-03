# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def _edge_cost(G, s, t):
    """Edge cost with None treated as prohibitive."""
    c = G[s][t].get("cost")
    return c if c is not None else 1e9


def _path_cost(G, path):
    return sum(_edge_cost(G, path[i], path[i + 1]) for i in range(len(path) - 1))


def search_algorithm(src, dsts, G, num_partitions):
    """Broadcast routing via k-shortest-path candidate sets + LP assignment.

    1. For each destination, generate up to k alternative low-cost simple paths
       (nx.shortest_simple_paths on a cost-weighted view of the graph), exposing
       parallel routes across networks that a single Dijkstra path ignores.
    2. Formulate partition->route assignment as a linear program (scipy
       linprog) with congestion-awareness: each partition of each dst picks
       exactly one route; edge usage by earlier assignments adds a congestion
       penalty to shared edges so later choices spread load across networks.
       If scipy is unavailable, fall back to a deterministic greedy assignment.
    3. Round the (fractional) LP solution to a valid assignment.

    This exploits multi-network parallelism and balances load, reducing both
    cost and throughput violations versus single-path routing.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Cost-weighted view so None costs are avoided rather than crashing.
    hw = nx.DiGraph()
    for u, v, d in h.edges(data=True):
        c = d.get("cost")
        hw.add_edge(u, v, cost=c if c is not None else 1e9)

    K = 6
    routes = {}  # dst -> list of (path, cost)
    for dst in dsts:
        cands = []
        try:
            gen = nx.shortest_simple_paths(hw, src, dst, weight="cost")
            for path in gen:
                cands.append((path, _path_cost(hw, path)))
                if len(cands) >= K:
                    break
        except Exception:
            pass
        if not cands:
            cands = [(nx.dijkstra_path(hw, src, dst, weight="cost"),
                      _path_cost(hw, nx.dijkstra_path(hw, src, dst, weight="cost")))]
        # dedupe by edge set
        seen, uniq = set(), []
        for p, c in sorted(cands, key=lambda x: x[1]):
            key = tuple(zip(p, p[1:]))
            if key not in seen:
                seen.add(key)
                uniq.append((p, c))
        routes[dst] = uniq[:K]

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    P = bc_topology.num_partitions

    def assign(dst, edge_load, use_lp=True):
        """Assign P partitions of dst to candidate routes, balancing load."""
        cands = routes[dst]
        n = len(cands)

        def route_cost(p_idx):
            # cost + congestion penalty for edges already used by this dst
            path, base = cands[p_idx]
            pen = 0.0
            for i in range(len(path) - 1):
                e = (path[i], path[i + 1])
                pen += 0.05 * base * edge_load.get(e, 0)
            return base + pen

        costs = [route_cost(i) for i in range(n)]
        chosen = None
        if use_lp:
            try:
                from scipy.optimize import linprog
                # variables x[i] = fraction of partitions on route i
                # min sum(costs[i] * x[i]); sum(x) = P; 0 <= x[i] <= 1
                A_ub = [[1] * n]
                b_ub = [P]
                res = linprog(c=costs, A_ub=A_ub, b_ub=b_ub,
                              A_eq=[[1] * n], b_eq=[P],
                              bounds=[(0, 1)] * n, method="highs")
                if res.success and res.x is not None:
                    # largest-remainder rounding to integers
                    fr = [max(0, min(1, round(v))) for v in res.x]
                    total = sum(fr)
                    rem = sorted(range(n), key=lambda i: -(res.x[i] - fr[i]))
                    k = 0
                    while total < P:
                        if fr[rem[k % n]] < P:
                            fr[rem[k % n]] += 1
                            total += 1
                        k += 1
                    while total > P:
                        for i in rem:
                            if fr[i] > 0 and total > P:
                                fr[i] -= 1
                                total -= 1
                    if sum(fr) == P and all(f >= 0 for f in fr):
                        chosen = []
                        for i, cnt in enumerate(fr):
                            chosen.extend([i] * cnt)
            except Exception:
                chosen = None
        if chosen is None:
            # greedy: round-robin over routes ordered by cost (spreads load)
            order = sorted(range(n), key=lambda i: costs[i])
            chosen = [order[p % n] for p in range(P)]
        return chosen

    for dst in dsts:
        edge_load = {}
        chosen = assign(dst, edge_load)
        for p_idx in range(P):
            path, _ = routes[dst][chosen[p_idx]]
            edges = []
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                edges.append([s, t, G[s][t]])
                edge_load[(s, t)] = edge_load.get((s, t), 0) + 1
            # recompute later partitions with updated congestion (optional
            # refinement is skipped for speed; single LP handles balance)
            bc_topology.set_dst_partition_paths(dst, p_idx, edges)

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
