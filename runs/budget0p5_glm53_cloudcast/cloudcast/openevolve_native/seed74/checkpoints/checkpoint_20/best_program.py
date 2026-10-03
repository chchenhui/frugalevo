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
        """Greedy Steiner: attach dsts in the given priority order, each via
        the cheapest path from any node already in the tree."""
        tree_nodes = {src}
        tree = nx.DiGraph()
        remaining = set(dsts)
        while remaining:
            best = None  # (cost, path)
            ordered = [d for d in order if d in remaining]
            for dst in (ordered[:1] + [d for d in remaining if d not in ordered[:1]]):
                for node in tree_nodes:
                    dist, paths = sp_from(node)
                    if dst in dist and (best is None or dist[dst] < best[0]):
                        best = (dist[dst], paths[dst])
                if best is not None and ordered and dst == ordered[0]:
                    break
            if best is None:
                dist, paths = sp_from(src)
                dst = next(iter(remaining))
                if dst not in dist:
                    remaining.discard(dst)
                    continue
                best = (dist[dst], paths[dst])
            path = best[1]
            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                if not tree.has_edge(s, t):
                    tree.add_edge(s, t, data=G[s][t])
                tree_nodes.update([s, t])
            remaining.discard(path[-1])
        return tree

    def tree_score(tree):
        """Total cost = sum of each unique tree edge's cost (broadcast pays
        each edge once per partition, so shared edges are not double-counted)."""
        if tree.number_of_nodes() == 0:
            return None
        try:
            dist = nx.single_source_dijkstra_path_length(tree, src, weight="cost")
        except Exception:
            return None
        for dst in dsts:
            if dst not in dist:
                return None
        total = 0.0
        for u, v in tree.edges():
            data = tree[u][v].get("data") or {}
            c = data.get("cost")
            if c is None:
                c = G[u][v].get("cost") if G.has_edge(u, v) else 0
            total += c if c is not None else 0
        return total

    direct_dist, _ = sp_from(src)
    orders = [
        list(dsts),
        sorted(dsts),
        sorted(dsts, reverse=True),
        sorted(dsts, key=lambda d: direct_dist.get(d, float("inf"))),
        sorted(dsts, key=lambda d: -direct_dist.get(d, float("inf"))),
    ]

    best_tree, best_total = None, None
    for order in orders:
        tree = build_tree(order)
        total = tree_score(tree)
        if total is not None and (best_total is None or total < best_total):
            best_tree, best_total = tree, total

    # candidate: union of per-dst shortest paths (edges paid once when shared)
    _, spaths = sp_from(src)
    union_tree = nx.DiGraph()
    union_ok = True
    for dst in dsts:
        if dst not in spaths:
            union_ok = False
            break
        path = spaths[dst]
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            if not union_tree.has_edge(s, t):
                union_tree.add_edge(s, t, data=G[s][t])
    if union_ok and union_tree.number_of_edges() > 0:
        total = tree_score(union_tree)
        if total is not None and (best_total is None or total < best_total):
            best_tree, best_total = union_tree, total

    if best_tree is None:
        best_tree = build_tree(list(dsts))

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
