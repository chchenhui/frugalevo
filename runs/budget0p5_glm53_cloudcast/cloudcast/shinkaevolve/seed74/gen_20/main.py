# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


# ---------------------------------------------------------------------------
# Stage 1: graph preprocessing
# ---------------------------------------------------------------------------
def _prepare_graph(G, src):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    return h


def _ecost(h, s, t):
    c = h[s][t].get("cost")
    return c if c is not None else 0.0


def _path_cost(h, path):
    return sum(_ecost(h, path[i], path[i + 1]) for i in range(len(path) - 1))


# ---------------------------------------------------------------------------
# Stage 2: candidate route generation per destination
# ---------------------------------------------------------------------------
def _generate_candidates(h, src, dst, k=6, slack=1.4):
    """k shortest paths, kept if within `slack` of the best cost."""
    cands = []
    try:
        for p in nx.shortest_simple_paths(h, src, dst, weight="cost"):
            cands.append(p)
            if len(cands) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    if not cands:
        try:
            cands = [nx.dijkstra_path(h, src, dst, weight="cost")]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    best = _path_cost(h, cands[0])
    kept = [p for p in cands if _path_cost(h, p) <= best * slack + 1e-9]
    return kept if kept else cands[:1]


# ---------------------------------------------------------------------------
# Stage 3: joint assignment with per-partition edge-reuse reward
# ---------------------------------------------------------------------------
def _assign_partitions(h, src, dsts, num_partitions, bc, all_candidates):
    # Process expensive destinations first so cheap destinations can
    # piggyback on the routes already committed for them.
    order = sorted(dsts, key=lambda d: _path_cost(h, all_candidates[d][0])
                   if all_candidates.get(d) else 0.0, reverse=True)

    # partition_edges[j] = set of (s,t) edges already chosen for partition j
    partition_edges = [set() for _ in range(num_partitions)]

    reuse_bonus = 0.25  # fraction of shared cost discounted per reused edge

    for dst in order:
        routes = all_candidates.get(dst)
        if not routes:
            # unreachable: fill partitions with empty edge lists
            for j in range(num_partitions):
                bc.append_dst_partition_path(dst, j, [])
            continue

        for j in range(num_partitions):
            committed = partition_edges[j]
            best_path, best_score = None, float("inf")
            for p in routes:
                raw = _path_cost(h, p)
                shared = 0.0
                for i in range(len(p) - 1):
                    if (p[i], p[i + 1]) in committed:
                        shared += _ecost(h, p[i], p[i + 1])
                # reuse reward: edges already sent for this partition are
                # (nearly) free to reuse -> reduces redundant transfers
                score = raw - reuse_bonus * shared
                if score < best_score:
                    best_score, best_path = score, p

            for i in range(len(best_path) - 1):
                s, t = best_path[i], best_path[i + 1]
                bc.append_dst_partition_path(dst, j, [s, t, bc_graph_edge(G_global := None, None) if False else None])
                # (placeholder never reached; real append below)
            break_outer = False
            for i in range(len(best_path) - 1):
                s, t = best_path[i], best_path[i + 1]
                bc.append_dst_partition_path(dst, j, [s, t, _orig_edge_data(dst, s, t)])
                committed.add((s, t))


def _orig_edge_data(dst, s, t):
    # resolved at call time via closure set in search_algorithm
    return _EDGE_DATA_HOLDER["G"][s][t]


_EDGE_DATA_HOLDER = {"G": None}


def search_algorithm(src, dsts, G, num_partitions):
    global _EDGE_DATA_HOLDER
    _EDGE_DATA_HOLDER["G"] = G

    h = _prepare_graph(G, src)
    bc = BroadCastTopology(src, dsts, num_partitions)

    # Candidate generation for all destinations (shared preprocessing).
    candidates = {}
    for dst in dsts:
        if dst in h and dst != src:
            routes = _generate_candidates(h, src, dst)
            if routes:
                candidates[dst] = routes

    # Joint cross-destination assignment.
    _assign_partitions(h, src, dsts, num_partitions, bc, candidates)

    return bc


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