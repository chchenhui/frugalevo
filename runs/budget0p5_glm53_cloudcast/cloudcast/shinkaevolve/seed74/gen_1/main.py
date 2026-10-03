# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List
import heapq


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Clean infinite/None costs defensively
    for u, v, d in h.edges(data=True):
        if d.get("cost") is None:
            d["cost"] = float("inf")

    # Helper: path as list of (u,v) edges with data
    def path_to_edges(path):
        out = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            out.append([s, t, G[s][t]])
        return out

    # 1) Candidate routes per destination: k shortest simple paths
    k = max(num_partitions, 3)
    candidates = {}
    for dst in dsts:
        try:
            paths = list(nx.shortest_simple_paths(h, src, dst, weight="cost"))[:k]
        except Exception:
            paths = []
        if not paths:
            # fallback dijkstra
            try:
                paths = [nx.dijkstra_path(h, src, dst, weight="cost")]
            except Exception:
                paths = []
        candidates[dst] = [tuple(p) for p in paths]

    # 2) Greedy shared-prefix selection with load balancing.
    # Track how many times each edge is used; prefer reusing heavily-used
    # edges (sharing) but avoid overloading any single edge beyond a cap.
    edge_usage = {}
    chosen = {}  # dst -> list of tuple paths (one per partition, may repeat)

    # order destinations by their best-path cost (harder ones first)
    order = sorted(dsts, key=lambda d: len(candidates.get(d, []) and candidates[d][0] or ()))

    for dst in order:
        cands = candidates.get(dst, [])
        if not cands:
            continue
        picks = []
        for pi in range(num_partitions):
            # score each candidate: prefer shared edges (discount), penalize overloaded
            best, best_score = None, None
            for cand in cands:
                score = 0.0
                for i in range(len(cand) - 1):
                    e = (cand[i], cand[i + 1])
                    cost = h[cand[i]][cand[i + 1]].get("cost", 0) or 0
                    uses = edge_usage.get(e, 0)
                    # effective marginal cost: full cost minus sharing discount,
                    # plus congestion penalty
                    score += cost * (1.0 / (1.0 + uses))
                # diversity bonus for paths differing from prior picks
                if picks and cand != picks[-1]:
                    score *= 0.95
                if best_score is None or score < best_score:
                    best, best_score = cand, score
            picks.append(best)
            # register usage
            for i in range(len(best) - 1):
                e = (best[i], best[i + 1])
                edge_usage[e] = edge_usage.get(e, 0) + 1

        chosen[dst] = picks

    # 3) Commit chosen paths into topology (per partition)
    for dst in dsts:
        if dst not in chosen:
            continue
        for pi, p in enumerate(chosen[dst]):
            edges = path_to_edges(list(p))
            for e in edges:
                bc_topology.append_dst_partition_path(dst, pi, e)

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

    def set_graph(self):
        pass


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
