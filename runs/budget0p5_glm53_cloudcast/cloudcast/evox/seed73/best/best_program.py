# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Exact-marginal-cost greedy broadcast routing + deep re-route polish.

    Score = cost of the UNION of directed edges used (shared edges paid
    once), so each destination's true marginal price is its Dijkstra
    shortest-path cost with already-purchased edges priced ~0. We build
    greedily under many destination orderings (cheap-first, expensive-first,
    fewest-hops, plus several seeded random shuffles), polish every
    candidate with a re-route-each-dst-against-others local search run to
    convergence, and also seed candidates from the undirected Steiner tree
    and MST backbones (edges discounted to ~0). Cheapest fully-reaching
    candidate wins; per-dst star fallback keeps success_rate = 1.0.
    """
    import random

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def path_edges(p):
        return {(p[i], p[i + 1]) for i in range(len(p) - 1)} if p else set()

    def union_edges(paths, skip=None):
        e = set()
        for d, p in paths.items():
            if p and d != skip:
                e |= path_edges(p)
        return e

    def ucost(e):
        return sum(h[a][b].get("cost") or 0 for a, b in e)

    def union_cost(paths):
        return ucost(union_edges(paths))

    def make_disc_w(owned):
        def w(u_, v_, d):
            c = d.get("cost")
            if c is None:
                return float("inf")
            return 0.0 if (u_, v_) in owned else c
        return w

    def dijkstra_paths(weight):
        out = {}
        for d in dsts:
            try:
                out[d] = nx.dijkstra_path(h, src, d, weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                out[d] = None
        return out

    def strict_w(u_, v_, d):
        c = d.get("cost")
        return c if c is not None else float("inf")

    # candidate 1: plain per-destination shortest-path star
    star = dijkstra_paths(strict_w)

    def greedy_build(order):
        paths, owned = {}, set()
        for d in order:
            try:
                p = nx.dijkstra_path(h, src, d, weight=make_disc_w(owned))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                p = star.get(d)
            if p:
                paths[d] = p
                owned |= path_edges(p)
        return paths

    def improve(paths, rounds=12):
        """Re-route each dst against the union of all other paths until
        no strict union-cost reduction is found (convergence)."""
        for _ in range(rounds):
            gained = False
            for d in list(paths):
                if not paths.get(d):
                    continue
                others = union_edges(paths, skip=d)
                try:
                    p = nx.dijkstra_path(h, src, d, weight=make_disc_w(others))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if p:
                    old = path_edges(paths[d])
                    if ucost(others | path_edges(p)) < ucost(others | old):
                        paths[d] = p
                        gained = True
            if not gained:
                break
        return paths

    reachable = [d for d in dsts if star.get(d)]
    scost = {d: ucost(path_edges(star[d])) for d in reachable}

    cands = [star]

    # undirected cheapest-direction projection of the multi-network graph
    und = nx.Graph()
    for a, b, d in h.edges(data=True):
        c = d.get("cost")
        if c is None:
            continue
        if not und.has_edge(a, b) or c < und[a][b]["cost"]:
            und.add_edge(a, b, cost=c)

    terminals = [src] + list(dsts)
    if und.number_of_edges() > 0 and all(n in und for n in terminals):
        backbones = []
        try:
            st = nx.steiner_tree(und, terminals, weight="cost")
            if st is not None:
                backbones.append(st)
        except Exception:
            pass
        try:
            backbones.append(nx.minimum_spanning_tree(und, weight="cost"))
        except Exception:
            pass
        for bb in backbones:
            if bb.number_of_edges() == 0:
                continue
            pairs = {frozenset((a, b)) for a, b in bb.edges()}

            def disc(a, b, d, _pairs=pairs):
                c = d.get("cost")
                if c is None:
                    return float("inf")
                return 1e-6 * c if frozenset((a, b)) in _pairs else c

            cands.append(improve(dijkstra_paths(disc)))

    # exact-marginal-cost greedy construction under many orderings
    base_orders = [
        sorted(reachable, key=lambda d: scost[d]),
        sorted(reachable, key=lambda d: -scost[d]),
        sorted(reachable, key=lambda d: len(star[d])),
        reachable,
    ]
    rng = random.Random(1234)
    for _ in range(20):
        shuf = list(reachable)
        rng.shuffle(shuf)
        base_orders.append(shuf)
    for od in base_orders:
        if not od:
            continue
        gp = greedy_build(od)
        # backfill unreachable dsts with star fallbacks
        for d in dsts:
            if d not in gp and star.get(d):
                gp[d] = star[d]
        cands.append(improve(gp))

    best, best_score = None, None
    for cand in cands:
        reach = sum(1 for d in dsts if cand.get(d))
        score = (-reach, union_cost(cand))
        if best_score is None or score < best_score:
            best, best_score = cand, score
    if best is None:
        best = {d: None for d in dsts}

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        path = best.get(dst) or star.get(dst)
        if not path:
            continue
        for j in range(num_partitions):
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

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
