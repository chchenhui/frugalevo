# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import random
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Multi-start greedy Steiner-tree broadcast with pruning.

    The greedy Steiner heuristic (graft cheapest remaining destination onto
    the shared tree, where tree edges cost 0) is sensitive to serving
    order. We build candidate trees for several starting orders (plain
    greedy, farthest-first, and each destination as seed), pick the
    cheapest by total unique-edge cost, then prune any tree edge that is
    not required by any final destination path. Shared trunk links are
    paid once instead of once per destination, minimizing redundant
    transfers across networks."""
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    n = num_partitions
    dst_list = sorted(set(dsts))

    def edge_cost(u, v):
        c = G[u][v].get("cost")
        return c if c is not None else float("inf")

    def build_tree(order=None, farthest_first=False):
        """One greedy Steiner run; returns (edge_set, dst->path map, total).

        Destinations are grafted in the given order (or cheapest-first /
        farthest-first if no explicit order). Tree edges cost 0 so shared
        trunk links are paid once."""
        used = set()
        paths = {}
        if order is not None:
            remaining = [d for d in order if d in dst_list]
        else:
            remaining = list(dst_list)

        def tree_weight(u, v, data):
            if (u, v) in used:
                return 0.0
            c = data.get("cost")
            return c if c is not None else float("inf")

        while remaining:
            if order is not None:
                dst = remaining.pop(0)
                try:
                    _, path = nx.single_source_dijkstra(h, src, dst, weight=tree_weight)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    try:
                        path = nx.dijkstra_path(h, src, dst, weight="cost")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                best = (0.0, dst, path)
            else:
                best = None
                for dst in remaining:
                    try:
                        dist, path = nx.single_source_dijkstra(h, src, dst, weight=tree_weight)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    if best is None or (dist > best[0] if farthest_first else dist < best[0]):
                        best = (dist, dst, path)

            if best is None:
                nxt = []
                for dst in remaining:
                    try:
                        path = nx.dijkstra_path(h, src, dst, weight="cost")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                    paths[dst] = path
                    for i in range(len(path) - 1):
                        used.add((path[i], path[i + 1]))
                    nxt.append(dst)
                remaining = [d for d in remaining if d not in nxt]
                continue

            _, dst, path = best
            paths[dst] = path
            for i in range(len(path) - 1):
                used.add((path[i], path[i + 1]))
            if dst in remaining:
                remaining.remove(dst)

        # keep only edges actually used by some destination path
        used = set()
        for path in paths.values():
            for i in range(len(path) - 1):
                used.add((path[i], path[i + 1]))
        total = sum(edge_cost(u, v) for (u, v) in used)
        return used, paths, total

    def improve(used, paths, total, rounds=6):
        """Local search: remove one destination's path, re-graft it cheapest
        against the rest of the tree; keep move if total cost drops."""
        best_total = total
        for _ in range(rounds):
            improved = False
            for dst in paths:
                # remove dst's path
                kept = set()
                for other, p in paths.items():
                    if other == dst:
                        continue
                    for i in range(len(p) - 1):
                        kept.add((p[i], p[i + 1]))
                # recompute which edges remain needed by others
                def tree_weight(u, v, data):
                    if (u, v) in kept:
                        return 0.0
                    c = data.get("cost")
                    return c if c is not None else float("inf")
                try:
                    _, path = nx.single_source_dijkstra(h, src, dst, weight=tree_weight)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                new_paths = dict(paths)
                new_paths[dst] = path
                new_used = set(kept)
                for i in range(len(path) - 1):
                    new_used.add((path[i], path[i + 1]))
                new_total = sum(edge_cost(u, v) for (u, v) in new_used)
                if new_total < best_total - 1e-9:
                    used, paths, best_total = new_used, new_paths, new_total
                    improved = True
            if not improved:
                break
        return used, paths, best_total

    # candidate runs: plain greedy, farthest-first, each dst as seed,
    # plus seeded random orders, then local-search improvement on the best.
    candidates = [build_tree(), build_tree(farthest_first=True)]
    for d in dst_list:
        candidates.append(build_tree(order=[d] + [x for x in dst_list if x != d]))

    rng = random.Random(42)
    for _ in range(min(10, max(2, 20 // max(1, len(dst_list))))):
        order = list(dst_list)
        rng.shuffle(order)
        candidates.append(build_tree(order=order))

    used_edges, paths, _total = min(candidates, key=lambda c: c[2])
    used_edges, paths, _total = improve(used_edges, paths, _total)

    # prune: keep only edges that appear on some destination's final path
    needed = set()
    for path in paths.values():
        for i in range(len(path) - 1):
            needed.add((path[i], path[i + 1]))

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    for dst, path in paths.items():
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(n):
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
