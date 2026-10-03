# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


class PathPlanner:
    """Modular planner that builds candidate routes per destination
    and greedily selects the cheapest valid one."""

    def __init__(self, G, num_partitions, k_candidates=6):
        self.G = G
        self.num_partitions = num_partitions
        self.k_candidates = k_candidates
        self._valid_graph = self._build_valid_graph()

    def _build_valid_graph(self):
        """Copy of G restricted to edges with known cost (avoids None-weight errors)."""
        h = self.G.copy()
        h.remove_edges_from(list(nx.selfloop_edges(h)))
        drop = [(u, v) for u, v, d in h.edges(data=True) if d.get("cost") is None]
        h.remove_edges_from(drop)
        return h

    def dijkstra_path(self, src, dst, blocked_in_src=True):
        h = self._valid_graph
        if blocked_in_src:
            h = h.copy()
            h.remove_edges_from(list(h.in_edges(src)))
        try:
            return nx.dijkstra_path(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def candidate_paths(self, src, dst):
        """Dijkstra path plus k cheap simple-path alternatives."""
        base = self.dijkstra_path(src, dst)
        if base is None:
            return []
        candidates = [base]
        h = self._valid_graph.copy()
        h.remove_edges_from(list(h.in_edges(src)))
        try:
            for p in nx.shortest_simple_paths(h, src, dst, weight="cost"):
                if p not in candidates:
                    candidates.append(p)
                if len(candidates) >= self.k_candidates:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return candidates

    def path_cost(self, path):
        total = 0.0
        for i in range(len(path) - 1):
            total += self.G[path[i]][path[i + 1]].get("cost") or 0.0
        return total

    def select(self, src, dst):
        """Return the cheapest valid candidate; Dijkstra result as fallback."""
        cands = self.candidate_paths(src, dst)
        if not cands:
            # retry on full graph in case cost filtering removed needed edges
            return self.dijkstra_path(src, dst) or nx.shortest_path(self.G, src, dst)
        best = min(cands, key=self.path_cost)
        return best


def search_algorithm(src, dsts, G, num_partitions):
    """
    Plan broadcast routes from src to each dst. For each destination, a set of
    candidate paths (Dijkstra + low-cost alternatives) is evaluated and the
    cheapest is chosen; all partitions reuse that route. Falls back to the
    classic Dijkstra route when no strictly cheaper candidate exists.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    planner = PathPlanner(G, num_partitions)

    for dst in dsts:
        # Reference cost of the plain Dijkstra route
        try:
            dij = nx.dijkstra_path(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        best = planner.select(src, dst)
        if best is None or planner.path_cost(best) > planner.path_cost(dij):
            best = dij  # safe fallback: never worse than baseline

        for i in range(0, len(best) - 1):
            s, t = best[i], best[i + 1]
            for j in range(bc_topology.num_partitions):
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
