# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Broadcast src to all dsts via an approximate directed Steiner tree.

    Builds a greedy Steiner tree trying multiple attachment orders
    (closest terminal first, farthest first, and a shuffled order),
    prunes non-destination leaves from each candidate, and keeps the
    cheapest tree. Shared trunk edges are paid once per partition
    instead of once per destination, and pruning removes wasted
    branches. All partitions follow routes inside the shared tree."""
    import random

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = set(d for d in dsts if d != src)

    def build_tree(order):
        """Greedy Steiner: attach terminals in the given order."""
        tree_edges = set()
        tree_nodes = {src}
        remaining = list(order)
        while remaining:
            best = None  # (dist, path)
            for r in remaining:
                try:
                    dist, path = nx.multi_source_dijkstra(
                        h, tree_nodes, r, weight="cost")
                except Exception:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, path)
            if best is None:
                # fallback: plain Dijkstra per dst on original graph
                for d in remaining:
                    try:
                        path = nx.dijkstra_path(h, src, d, weight="cost")
                    except Exception:
                        path = nx.dijkstra_path(G, src, d, weight="cost")
                    for i in range(len(path) - 1):
                        tree_edges.add((path[i], path[i + 1]))
                break
            path = best[1]
            for i in range(len(path) - 1):
                tree_edges.add((path[i], path[i + 1]))
            tree_nodes.update(path)
            remaining = [d for d in remaining if d not in tree_nodes]
        return prune(tree_edges, terminals)

    def prune(tree_edges, terminals):
        """Iteratively drop edges to leaves that are not destinations."""
        edges = set(tree_edges)
        changed = True
        while changed:
            changed = False
            out_deg = {}
            in_deg = {}
            for s, t in edges:
                out_deg[s] = out_deg.get(s, 0) + 1
                in_deg[t] = in_deg.get(t, 0) + 1
            for s in list(out_deg):
                if s != src and s not in terminals and out_deg[s] > 0 and s not in in_deg:
                    edges = {e for e in edges if e[0] != s}
                    changed = True
        return edges

    def tree_cost(edges):
        return sum(G[s][t].get("cost") or 0 for s, t in edges)

    def refine(edges):
        """Local search on the edge set: try deleting each edge and
        reconnecting any cut destinations via the cheapest path from
        the surviving tree; accept strictly cheaper solutions."""
        best = set(edges)
        best_cost = tree_cost(best)
        improved = True
        while improved and len(best) > 1:
            improved = False
            for e in sorted(best):
                cand = best - {e}
                # nodes reachable from src using only candidate edges
                adj = {}
                for s, t in cand:
                    adj.setdefault(s, []).append(t)
                reach = {src}
                frontier = [src]
                while frontier:
                    u = frontier.pop()
                    for v in adj.get(u, ()):
                        if v not in reach:
                            reach.add(v)
                            frontier.append(v)
                missing = [d for d in terminals if d not in reach]
                if missing:
                    try:
                        _, paths = nx.multi_source_dijkstra(h, reach)
                    except Exception:
                        continue
                    add = set()
                    ok = True
                    for d in missing:
                        if d not in paths:
                            ok = False
                            break
                        p = paths[d]
                        for i in range(len(p) - 1):
                            add.add((p[i], p[i + 1]))
                    if not ok:
                        continue
                    cand |= add
                c = tree_cost(cand)
                if c < best_cost - 1e-9:
                    best, best_cost = cand, c
                    improved = True
        return best

    # Candidate orders: closest-first, farthest-first, shuffled (by direct dist)
    try:
        dists = {d: nx.dijkstra_path_length(h, src, d, weight="cost")
                 for d in terminals}
    except Exception:
        dists = {d: 0 for d in terminals}
    order_close = sorted(terminals, key=lambda d: dists[d])
    order_far = list(reversed(order_close))

    candidates = [build_tree(order_close)]
    if order_far != order_close:
        candidates.append(build_tree(order_far))
    order_rand = list(terminals)
    random.shuffle(order_rand)
    if order_rand not in (order_close, order_far):
        candidates.append(build_tree(order_rand))

    # Additional seeded random orders: each order yields a different
    # greedy Steiner tree; we keep whichever pruned tree is cheapest.
    for seed in range(min(8, len(terminals))):
        rng = random.Random(seed)
        order = list(terminals)
        rng.shuffle(order)
        if order not in (order_close, order_far):
            candidates.append(build_tree(order))

    # Rotations of the closest-first order: deterministic variants
    # that often attach the same terminals through different trunks.
    base = list(order_close)
    for k in range(1, min(4, len(base))):
        rot = base[k:] + base[:k]
        if rot not in (order_close, order_far):
            candidates.append(build_tree(rot))

    # Fallback: independent shortest paths, in case trees miss a dst
    fallback_edges = set()
    for d in terminals:
        try:
            path = nx.dijkstra_path(h, src, d, weight="cost")
        except Exception:
            path = nx.dijkstra_path(G, src, d, weight="cost")
        for i in range(len(path) - 1):
            fallback_edges.add((path[i], path[i + 1]))
    candidates.append(fallback_edges)

    # Local-search refinement of every candidate before selection.
    candidates = [refine(c) for c in candidates]

    tree_edges = min(candidates, key=tree_cost)

    # --- Emit per-partition paths along the shared tree ---
    tree = h.copy()
    tree.remove_edges_from(
        [e for e in list(tree.edges) if e not in tree_edges])
    tree.add_edges_from(
        [(s, t, dict(G[s][t])) for (s, t) in tree_edges if not tree.has_edge(s, t)])

    for dst in dsts:
        if dst == src:
            continue
        if dst in tree and nx.has_path(tree, src, dst):
            path = nx.dijkstra_path(tree, src, dst, weight="cost")
            edges = [[path[i], path[i + 1], G[path[i]][path[i + 1]]]
                     for i in range(len(path) - 1)]
        else:
            try:
                path = nx.dijkstra_path(h, src, dst, weight="cost")
            except Exception:
                path = nx.dijkstra_path(G, src, dst, weight="cost")
            edges = [[path[i], path[i + 1], G[path[i]][path[i + 1]]]
                     for i in range(len(path) - 1)]
        for j in range(num_partitions):
            for e in edges:
                bc_topology.append_dst_partition_path(dst, j, e)

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
