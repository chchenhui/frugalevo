# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import random
import pandas as pd
from typing import Dict, List


def _greedy_tree(h, src, order):
    """Greedy directed-Steiner heuristic: route terminals in `order`,
    reusing already-built edges for free (shared trunks paid once)."""
    used, res = set(), {}
    for dst in order:
        def w(u, v, data, used=used):
            return 0.0 if (u, v) in used else data["cost"]
        try:
            path = nx.dijkstra_path(h, src, dst, weight=w)
        except Exception:
            try:
                path = nx.shortest_path(h, src, dst)
            except Exception:
                continue
        for i in range(len(path) - 1):
            used.add((path[i], path[i + 1]))
        res[dst] = path
    return res


def _tree_cost(h, paths):
    """Cost of a set of paths counting each distinct edge once."""
    seen, c = set(), 0.0
    for p in paths.values():
        for i in range(len(p) - 1):
            e = (p[i], p[i + 1])
            if e not in seen:
                seen.add(e)
                c += h[e[0]][e[1]]["cost"]
    return c


def _refine(h, src, paths, rounds=8):
    """Local search: reroute one destination at a time while all edges used
    by the *other* destinations cost 0. Each accepted move strictly lowers
    total tree cost, escaping greedy ordering local optima."""
    keys = list(paths)
    for _ in range(rounds):
        changed = False
        for dst in keys:
            others = set()
            for d, p in paths.items():
                if d != dst:
                    for i in range(len(p) - 1):
                        others.add((p[i], p[i + 1]))
            old = paths[dst]
            old_c = sum(h[old[i]][old[i + 1]]["cost"] for i in range(len(old) - 1)
                        if (old[i], old[i + 1]) not in others)

            def w(u, v, data, others=others):
                return 0.0 if (u, v) in others else data["cost"]

            try:
                new = nx.dijkstra_path(h, src, dst, weight=w)
            except Exception:
                continue
            new_c = sum(h[new[i]][new[i + 1]]["cost"] for i in range(len(new) - 1)
                        if (new[i], new[i + 1]) not in others)
            if new_c < old_c - 1e-12:
                paths[dst] = new
                changed = True
        if not changed:
            break
    return paths


def search_algorithm(src, dsts, G, num_partitions):
    """Broadcast src's partitions to all dsts with minimal transfer cost.

    Approach: directed Steiner-tree approximation. A greedy heuristic
    routes destinations one-by-one with Dijkstra, making edges already in
    the tree free (shared trunk links paid once => no redundant transfers).
    Many destination orderings are tried (nearest/farthest-first, given,
    plus randomized shuffles), each tree is then improved by a local-search
    refinement pass that reroutes single destinations against the rest of
    the tree, and the cheapest refined tree wins. The winning tree is
    replicated for every partition since network and data are identical.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Sanitize missing / NaN costs so Dijkstra is always valid.
    for _, _, d in h.edges(data=True):
        c = d.get("cost")
        if c is None or (isinstance(c, float) and c != c):
            d["cost"] = 1e12

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    try:
        dist = nx.single_source_dijkstra_path_length(h, src, weight="cost")
    except Exception:
        dist = {}

    reachable = [d for d in dsts if d in dist] or list(dsts)
    near_first = sorted(reachable, key=lambda d: dist.get(d, float("inf")))
    orders = [near_first, near_first[::-1], list(dsts)]
    rng = random.Random(1234)
    for _ in range(min(40, max(4, 8 * len(reachable)))):
        o = list(reachable)
        rng.shuffle(o)
        orders.append(o)

    best_paths, best_cost = None, float("inf")
    seen_trees = set()
    for order in orders:
        paths = _greedy_tree(h, src, order)
        if not paths:
            continue
        # Skip refining duplicate greedy trees (many orders converge).
        sig = frozenset((p[i], p[i + 1]) for p in paths.values()
                        for i in range(len(p) - 1))
        if sig in seen_trees:
            continue
        seen_trees.add(sig)
        paths = _refine(h, src, paths)
        c = _tree_cost(h, paths)
        if best_paths is None or len(paths) > len(best_paths) or \
                (len(paths) == len(best_paths) and c < best_cost):
            best_paths, best_cost = paths, c
    if best_paths is None:
        best_paths = _greedy_tree(h, src, list(dsts))

    for j in range(bc_topology.num_partitions):
        for dst, path in best_paths.items():
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
