# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List

DEFAULT_FALLBACK_COST = 1e9
TRUNK_DISCOUNT = 0.3
K_CANDIDATES = 6


class PathPlanner:
    """Modular planner: valid-graph Dijkstra + bounded simple-path alternatives
    + full-graph candidate discovery (None costs filled with median)."""

    def __init__(self, G, num_partitions, k_candidates=K_CANDIDATES):
        self.G = G
        self.num_partitions = num_partitions
        self.k_candidates = k_candidates
        self._valid_graph = self._build_valid_graph()
        self._used_edges = set()

    def _build_valid_graph(self):
        """Copy of G restricted to edges with known cost (avoids None-weight errors)."""
        h = self.G.copy()
        h.remove_edges_from(list(nx.selfloop_edges(h)))
        drop = [(u, v) for u, v, d in h.edges(data=True) if d.get("cost") is None]
        h.remove_edges_from(drop)
        return h

    def _routing_graph(self, src):
        """Valid graph without in-edges of src (no re-entering the source)."""
        h = self._valid_graph.copy()
        h.remove_edges_from(list(h.in_edges(src)))
        return h

    def _build_full_graph(self, src):
        """Full graph (None-cost edges included) with median-filled weights,
        used only to discover candidate routes."""
        h = self.G.copy()
        h.remove_edges_from(list(nx.selfloop_edges(h)))
        h.remove_edges_from(list(h.in_edges(src)))
        costs = sorted(d["cost"] for _, _, d in h.edges(data=True)
                       if d.get("cost") is not None)
        median_cost = costs[len(costs) // 2] if costs else 1.0
        for _, _, d in h.edges(data=True):
            if d.get("cost") is None:
                d["cost"] = median_cost
        return h

    def dijkstra_path(self, src, dst):
        try:
            return nx.dijkstra_path(self._routing_graph(src), src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def candidate_paths(self, src, dst, baseline):
        """Baseline plus bounded low-cost simple-path alternatives (valid graph),
        plus a full-graph candidate, deduplicated."""
        candidates = [baseline]
        h = self._routing_graph(src)
        try:
            for p in nx.shortest_simple_paths(h, src, dst, weight="cost"):
                if p not in candidates:
                    candidates.append(p)
                if len(candidates) >= self.k_candidates:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        # full-graph candidate (may use None-cost edges)
        try:
            fp = nx.dijkstra_path(self._build_full_graph(src), src, dst, weight="cost")
            if fp not in candidates:
                candidates.append(fp)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return candidates

    def valid_path_cost(self, path):
        """True accounting cost: None-cost edges contribute 0 (uncosted links)."""
        total = 0.0
        for i in range(len(path) - 1):
            c = self.G[path[i]][path[i + 1]].get("cost")
            if c is not None:
                total += c
        return total

    def discounted_score(self, path):
        """Cost with trunk-reuse discount for edges already used by other dsts."""
        score = 0.0
        for i in range(len(path) - 1):
            c = self.G[path[i]][path[i + 1]].get("cost")
            if c is None:
                c = 0.0
            if (path[i], path[i + 1]) in self._used_edges:
                c *= (1.0 - TRUNK_DISCOUNT)
            score += c
        return score

    def select(self, src, dst, baseline, baseline_cost):
        """Pick candidate minimizing discounted score, but never adopt a path
        whose true cost exceeds the baseline's true cost."""
        candidates = self.candidate_paths(src, dst, baseline)
        best, best_score = baseline, self.discounted_score(baseline)
        for p in candidates:
            if p is baseline:
                continue
            if self.valid_path_cost(p) > baseline_cost:
                continue  # never worse than baseline
            s = self.discounted_score(p)
            if s < best_score:
                best, best_score = p, s
        for i in range(len(best) - 1):
            self._used_edges.add((best[i], best[i + 1]))
        return best


def search_algorithm(src, dsts, G, num_partitions):
    """
    Plan broadcast routes from src to each dst. Destinations are processed in
    descending order of baseline (Dijkstra) cost so that expensive routes build
    shared trunks first; later destinations get a reuse discount on shared
    edges, reducing redundant transfers while never exceeding their baseline.
    """
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    planner = PathPlanner(G, num_partitions)

    # Baselines for every destination (skip unreachable).
    baselines = {}
    for dst in dsts:
        p = planner.dijkstra_path(src, dst)
        if p is not None:
            baselines[dst] = p
        else:
            # retry on the raw graph (cost filtering may have disconnected dst)
            try:
                p = nx.dijkstra_path(G, src, dst, weight=lambda u, v, d: d.get("cost") or DEFAULT_FALLBACK_COST)
                baselines[dst] = p
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    # Most expensive baseline first: builds shared trunks early.
    ordered = sorted(baselines.keys(),
                     key=lambda d: planner.valid_path_cost(baselines[d]),
                     reverse=True)

    chosen = {}
    for dst in ordered:
        baseline = baselines[dst]
        baseline_cost = planner.valid_path_cost(baseline)
        chosen[dst] = planner.select(src, dst, baseline, baseline_cost)

    # Emit in the original dsts order to preserve output structure.
    for dst in dsts:
        if dst not in chosen:
            continue
        best = chosen[dst]
        for i in range(len(best) - 1):
            s, t = best[i], best[i + 1]
            edge_data = G[s][t]
            for j in range(num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, edge_data])

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