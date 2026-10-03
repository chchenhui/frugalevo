# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def _path_cost(h, path):
    """Total cost of a node path (list of nodes) in graph h."""
    return sum(h[path[i]][path[i + 1]]["cost"] for i in range(len(path) - 1))


def _candidate_sph(src, dsts, h):
    """Candidate 1: greedy shortest-path Steiner-tree heuristic.

    Repeatedly attaches the cheapest (via multi-source Dijkstra from the
    current tree) unconnected destination, so shared trunk edges are
    reused across destinations.
    Returns dict dst -> node path from src, or None if any dst unreachable.
    """
    tree_nodes = {src}
    tree_pred = {}
    remaining = set(dsts)
    while remaining:
        dist, paths = nx.multi_source_dijkstra(h, list(tree_nodes), weight="cost")
        reachable = [d for d in remaining if d in dist]
        if not reachable:
            return None
        best_dst = min(reachable, key=lambda d: dist[d])
        path = paths[best_dst]
        for i in range(len(path) - 1):
            child, parent = path[i + 1], path[i]
            if child not in tree_pred:
                tree_pred[child] = parent
            tree_nodes.add(child)
        remaining.discard(best_dst)

    out = {}
    for dst in dsts:
        node, rev = dst, []
        while node != src:
            rev.append(node)
            node = tree_pred[node]
        out[dst] = [src] + list(reversed(rev))
    return out


def _candidate_sph_dst_order(src, dsts, h):
    """Candidate 2: Steiner heuristic attaching destinations in input order
    (cheapest-first attachment from the tree at each step)."""
    tree_nodes = {src}
    tree_pred = {}
    for dst in dsts:
        if dst in tree_nodes:
            continue
        dist, paths = nx.multi_source_dijkstra(h, list(tree_nodes), weight="cost")
        if dst not in dist:
            return None
        path = paths[dst]
        for i in range(len(path) - 1):
            child, parent = path[i + 1], path[i]
            if child not in tree_pred:
                tree_pred[child] = parent
            tree_nodes.add(child)
    out = {}
    for dst in dsts:
        node, rev = dst, []
        while node != src:
            rev.append(node)
            node = tree_pred[node]
        out[dst] = [src] + list(reversed(rev))
    return out


def _candidate_independent(src, dsts, h):
    """Candidate 3: independent per-destination shortest paths."""
    out = {}
    for dst in dsts:
        try:
            out[dst] = nx.shortest_path(h, src, dst, weight="cost")
        except nx.NetworkXNoPath:
            return None
    return out


def _candidate_reuse_discount(src, dsts, h, discount=0.5):
    """Candidate 4: incremental routing with a reuse discount.

    Routes destinations one at a time on a graph where edges already in the
    broadcast tree are discounted (they only add marginal cost when shared).
    This finds trees that partially share trunks where full Steiner sharing
    is not optimal, often beating both pure Steiner and pure independent
    routing under broadcast billing.
    """
    used = set()
    out = {}
    for dst in dsts:
        hh = h.copy()
        for u, v in used:
            if hh.has_edge(u, v):
                hh[u][v]["cost"] *= discount
        try:
            p = nx.shortest_path(hh, src, dst, weight="cost")
        except nx.NetworkXNoPath:
            return None
        out[dst] = p
        for i in range(len(p) - 1):
            used.add((p[i], p[i + 1]))
    return out


def search_algorithm(src, dsts, G, num_partitions):
    """Multi-strategy broadcast routing with cost-based candidate selection.

    Generates several candidate routings (greedy Steiner tree, ordered
    Steiner tree, and independent shortest paths), evaluates each by the
    total per-partition path cost, and emits the cheapest one. This is
    robust to whether shared edges are billed once or per-use, and
    balances load by preferring cheaper multi-network routes.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    # Drop edges with unknown cost so they never pollute routing
    h.remove_edges_from([(u, v) for u, v, d in h.edges(data=True) if d.get("cost") is None])

    candidates = [
        _candidate_sph(src, dsts, h),
        _candidate_sph_dst_order(src, dsts, h),
        _candidate_independent(src, dsts, h),
        _candidate_reuse_discount(src, dsts, h, 0.5),
        _candidate_reuse_discount(src, dsts, h, 0.0),
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        # Fallback: direct edges only (or empty) so we still return a topology
        candidates = [{dst: [src, dst] if h.has_edge(src, dst) else [src] for dst in dsts}]

    def total_cost(paths):
        """Broadcast cost: each distinct edge is paid once per partition,
        no matter how many destinations traverse it (shared multicast links)."""
        edges = set()
        for d in dsts:
            p = paths[d]
            for i in range(len(p) - 1):
                edges.add((p[i], p[i + 1]))
        return num_partitions * sum(h[u][v]["cost"] for u, v in edges)

    best = min(candidates, key=total_cost)

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        nodes = best[dst]
        for i in range(len(nodes) - 1):
            s, t = nodes[i], nodes[i + 1]
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
