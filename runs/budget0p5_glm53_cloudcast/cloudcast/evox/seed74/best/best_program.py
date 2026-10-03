# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Cost-sharing iterated Dijkstra per partition with ordering search.

    For each partition, route destinations one at a time where edges already
    installed in this partition have zero incremental cost (greedy Steiner-ish
    tree). Two improvements over plain cost-sharing:
      1. Destinations are also tried in distance-sorted order (closest first),
         which installs cheap shared edges early so later destinations can
         reuse them for free.
      2. Both orderings (input order and sorted order) are evaluated per
         partition by their true incremental edge cost, and the cheaper
         assignment is kept, so ordering never regresses.
    Relay chaining (routing a dst via an already-reached dst) is kept.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Precompute plain shortest-path distances for ordering heuristic
    try:
        dist = nx.single_source_dijkstra_path_length(h, src, weight="cost")
    except Exception:
        dist = {}

    def real_edge_cost(s, t):
        c = h[s][t].get("cost")
        return c if c is not None else 0.0

    def build_partition(order):
        """Greedy cost-sharing tree for one partition over dst order.
        Returns (total_incremental_cost, {dst: path})."""
        installed = set()
        reached = {}
        total = 0.0

        def edge_cost(s, t):
            return 0 if (s, t) in installed else real_edge_cost(s, t)

        for dst in order:
            best_path, best_cost = None, None

            # Option A: direct route from src with cost-sharing discount
            try:
                path = nx.dijkstra_path(
                    h, src, dst,
                    weight=lambda s, t, d: edge_cost(s, t))
                inc = sum(real_edge_cost(path[i], path[i + 1])
                          for i in range(len(path) - 1)
                          if (path[i], path[i + 1]) not in installed)
                best_path, best_cost = path, inc
            except nx.NetworkXNoPath:
                pass

            # Option B: relay via an already-reached destination
            for mid, mid_path in reached.items():
                if mid == dst or not h.has_edge(mid, dst):
                    continue
                relay_path = mid_path + [dst]
                inc = sum(real_edge_cost(relay_path[i], relay_path[i + 1])
                          for i in range(len(relay_path) - 1)
                          if (relay_path[i], relay_path[i + 1]) not in installed)
                if best_cost is None or inc < best_cost:
                    best_path, best_cost = relay_path, inc

            if best_path is None:
                continue  # unreachable destination

            total += best_cost
            for i in range(len(best_path) - 1):
                installed.add((best_path[i], best_path[i + 1]))
            reached[dst] = best_path
        return total, reached

    # Candidate orderings: input, closest-first, farthest-first, plus
    # deterministic shuffles for extra Steiner-tree exploration diversity
    uniq = [d for d in dict.fromkeys(dsts) if d != src]
    orders = [list(uniq)]
    sorted_order = sorted(uniq, key=lambda d: dist.get(d, float("inf")))
    if sorted_order and sorted_order != orders[0]:
        orders.append(sorted_order)
    far_order = sorted(uniq, key=lambda d: -dist.get(d, float("inf")))
    if far_order and far_order not in orders:
        orders.append(far_order)
    # a few deterministic pseudo-random shuffles for extra diversity
    import random
    for seed in range(1, 9):
        shuffled = list(uniq)
        random.Random(seed).shuffle(shuffled)
        if shuffled not in orders:
            orders.append(shuffled)

    for partition in range(num_partitions):
        best_total, best_reached = None, None
        for order in orders:
            total, reached = build_partition(order)
            if best_total is None or total < best_total:
                best_total, best_reached = total, reached

        # Local search: pairwise swaps on the best order until no improvement.
        # Greedy cost-sharing is very order-sensitive; small swaps often unlock
        # cheaper shared-edge structures (poor-man's Steiner refinement).
        if best_reached is not None and len(uniq) > 2:
            base_order = list(uniq)
            improved = True
            rounds = 0
            while improved and rounds < 6:
                improved = False
                rounds += 1
                for i in range(len(base_order)):
                    for j in range(i + 1, len(base_order)):
                        cand = list(base_order)
                        cand[i], cand[j] = cand[j], cand[i]
                        total, reached = build_partition(cand)
                        if total < best_total:
                            best_total, best_reached = total, reached
                            base_order = cand
                            improved = True
                # Re-seed shuffles around current best for extra exploration
                if improved:
                    for seed in (21, 22):
                        shuffled = list(base_order)
                        random.Random(seed + rounds).shuffle(shuffled)
                        total, reached = build_partition(shuffled)
                        if total < best_total:
                            best_total, best_reached = total, reached
                            base_order = shuffled

        for dst, path in (best_reached or {}).items():
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                bc_topology.append_dst_partition_path(dst, partition, [s, t, G[s][t]])

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
