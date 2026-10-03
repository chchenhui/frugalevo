# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def _edge_cost(G, s, t):
    c = G[s][t].get("cost")
    return c if c is not None else 0.0


def search_algorithm(src, dsts, G, num_partitions):
    """Best-of-candidates broadcast routing.

    Generates several candidate routing solutions:
      1. Independent per-destination shortest paths (cheap direct routes).
      2. Approximate Steiner tree (multi-source Dijkstra heuristic) that
         maximizes edge sharing to eliminate redundant transfers.
      3. Steiner tree built by attaching destinations in a random-ish
         (sorted by distance) order for a potentially different tree.
    Each candidate is evaluated under the cost model where every distinct
    edge used is paid once per partition; the cheapest candidate is used
    for all partitions, minimizing total broadcast cost.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def sp(src_node, dst):
        try:
            return nx.shortest_path(h, src_node, dst, weight="cost")
        except nx.NetworkXNoPath:
            return None

    def steiner_tree(order_key):
        tree_nodes = {src}
        tree_pred = {}
        remaining = set(dsts)
        while remaining:
            dist, paths = nx.multi_source_dijkstra(h, list(tree_nodes), weight="cost")
            if not all(d in dist for d in remaining):
                return None
            best_dst = min(remaining, key=lambda d: (order_key(d), dist[d]))
            path = paths[best_dst]
            for i in range(len(path) - 1):
                child, parent = path[i + 1], path[i]
                if child not in tree_pred:
                    tree_pred[child] = parent
                tree_nodes.add(child)
            remaining.discard(best_dst)

        def tree_path_to(dst):
            node = dst
            rev = []
            while node != src:
                rev.append((tree_pred[node], node))
                node = tree_pred[node]
            return list(reversed(rev))

        return {dst: tree_path_to(dst) for dst in dsts}

    candidates = []

    # Candidate 1: independent shortest paths
    indep = {}
    ok = True
    for dst in dsts:
        p = sp(src, dst)
        if p is None:
            ok = False
            break
        indep[dst] = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
    if ok:
        candidates.append(indep)

    # Candidates 2 & 3: Steiner trees with different attach orders
    t1 = steiner_tree(lambda d: 0)
    if t1:
        candidates.append(t1)
    t2 = steiner_tree(lambda d: hash(d) % 1000)
    if t2:
        candidates.append(t2)

    if not candidates:
        return BroadCastTopology(src, dsts, num_partitions)

    def candidate_cost(paths):
        edges = set()
        for dst in dsts:
            for (s, t) in paths[dst]:
                edges.add((s, t))
        return sum(_edge_cost(G, s, t) for (s, t) in edges)

    best = min(candidates, key=candidate_cost)

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        for (s, t) in best[dst]:
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
