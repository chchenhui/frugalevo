# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Broadcast via best-of-many Steiner trees: build greedy cheapest-attachment
    Steiner trees under several destination orderings, score each tree by the
    sum of per-dst in-tree path costs (keeping edge sharing intact), and emit
    the cheapest tree's per-dst edge lists reused across all partitions."""
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    dist_cache = {}

    def sp_from(node):
        if node not in dist_cache:
            dist, paths = nx.single_source_dijkstra(h, node, weight="cost")
            dist_cache[node] = (dist, paths)
        return dist_cache[node]

    def build_tree(order):
        """Greedy Steiner: attach dsts strictly in the given order, each via
        the cheapest path from any node already in the tree."""
        tree_nodes = {src}
        tree = nx.DiGraph()
        for dst in order:
            best = None  # (cost, path)
            for node in tree_nodes:
                dist, paths = sp_from(node)
                if dst in dist and (best is None or dist[dst] < best[0]):
                    best = (dist[dst], paths[dst])
            if best is None:
                dist, paths = sp_from(src)
                if dst not in dist:
                    continue
                best = (dist[dst], paths[dst])
            path = best[1]
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                if not tree.has_edge(s, t):
                    tree.add_edge(s, t, data=G[s][t])
                tree_nodes.update([s, t])
        return tree

    def tree_score(tree):
        """Broadcast pays each unique tree edge once: total cost = sum of
        unique edge costs (shared trunk edges are not double-counted)."""
        total = 0.0
        for s, t in tree.edges():
            data = tree[s][t].get("data", {})
            c = data.get("cost")
            if c is None:
                return None
            total += c
        return total

    # candidate orderings: natural, sorted, reversed, greedy-cheapest-first,
    # plus seeded randomized restarts (random orders often beat fixed ones)
    import random
    direct_dist, _ = sp_from(src)
    orders = [
        list(dsts),
        sorted(dsts),
        sorted(dsts, reverse=True),
        sorted(dsts, key=lambda d: direct_dist.get(d, float("inf"))),
        sorted(dsts, key=lambda d: -direct_dist.get(d, float("inf"))),
    ]
    rng = random.Random(0)
    for _ in range(20):
        o = list(dsts)
        rng.shuffle(o)
        orders.append(o)

    best_tree, best_total = None, None
    for order in orders:
        tree = build_tree(order)
        total = tree_score(tree)
        if total is not None and (best_total is None or total < best_total):
            best_tree, best_total = tree, total

    # fallback: pure greedy (original behavior) if all orderings failed
    if best_tree is None:
        best_tree = build_tree(list(dsts))

    # Prune: keep only edges lying on some src->dst path in the winning tree
    try:
        used_edges = set()
        for dst in dsts:
            p = nx.shortest_path(best_tree, src, dst, weight="cost")
            for i in range(len(p) - 1):
                used_edges.add((p[i], p[i + 1]))
        best_tree.remove_edges_from(
            [e for e in list(best_tree.edges()) if e not in used_edges]
        )
    except Exception:
        pass

    # Extract each dst's cheapest path within the winning tree
    for dst in dsts:
        try:
            path = nx.shortest_path(best_tree, src, dst, weight="cost")
        except Exception:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
        edge_list = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            data = G[s][t] if G.has_edge(s, t) else best_tree[s][t].get("data", {})
            edge_list.append([s, t, data])
        # reuse the same edge list for all partitions (no redundant transfers)
        for j in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(dst, j, list(edge_list))

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
