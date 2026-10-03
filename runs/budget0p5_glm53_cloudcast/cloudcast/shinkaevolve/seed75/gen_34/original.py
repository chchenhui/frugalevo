# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Build a multicast (Steiner-like) tree rooted at src spanning all dsts.
    # Data is transferred once per tree edge instead of once per (dst, edge),
    # eliminating redundant transfers over shared path prefixes.
    # The tree is identical across partitions (nothing is partition-dependent),
    # so we build it ONCE and reuse it for every partition.

    # Cache the reversed graph once: it never changes during tree construction.
    rh = h.reverse(copy=True)
    # Cache Dijkstra results per start node in the reversed graph, since
    # edge costs are never modified during the search.
    dijkstra_cache = {}

    def rev_dijkstra(d):
        if d not in dijkstra_cache:
            if d not in rh:
                dijkstra_cache[d] = (None, None)
            else:
                try:
                    dijkstra_cache[d] = nx.single_source_dijkstra(rh, d, weight="cost")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    dijkstra_cache[d] = ({}, {})
        return dijkstra_cache[d]

    # tree as a map: node -> parent node (rooted at src)
    tree_parent = {src: None}
    reached = {src}
    remaining = set(dsts)

    while remaining:
        # find the destination closest (in cost) to the current tree
        best_dst, best_dist, best_target, best_path = None, float("inf"), None, None
        for d in remaining:
            # shortest path from d to any node already in the tree
            # (run Dijkstra from d on the reversed graph; cached)
            d_dist, d_paths = rev_dijkstra(d)
            if not d_dist:
                continue
            # choose the cheapest tree node reachable from d
            target, tdist = None, float("inf")
            for node in reached:
                if node in d_dist and d_dist[node] < tdist:
                    target, tdist = node, d_dist[node]
            if target is not None and tdist < best_dist:
                # path from d to target in original graph
                rev_path = nx.dijkstra_path(rh, d, target, weight="cost")
                best_dst, best_dist, best_target, best_path = d, tdist, target, rev_path

        if best_dst is None:
            # unreachable destination: fall back to direct attempt (may fail)
            best_dst = remaining.pop()
            try:
                best_path = nx.dijkstra_path(h, src, best_dst, weight="cost")[::-1]
            except nx.NetworkXNoPath:
                continue
        else:
            remaining.discard(best_dst)

        # best_path goes from best_dst ... -> best_target (reverse-graph path).
        # Add edges into the tree: parent of path[i] is path[i+1]
        for i in range(len(best_path) - 1):
            child, parent = best_path[i], best_path[i + 1]
            if child not in reached:
                tree_parent[child] = parent
                reached.add(child)
        reached.add(best_dst)
        if best_dst not in tree_parent and len(best_path) >= 2:
            tree_parent[best_dst] = best_path[1]

    for j in range(num_partitions):
        # per-destination path reconstruction (tree is shared across partitions)
        for dst in dsts:
            if dst not in tree_parent and dst != src:
                continue
            # walk from dst up to src
            chain = []
            cur = dst
            guard = 0
            while cur != src and guard <= len(tree_parent) + 1:
                par = tree_parent.get(cur)
                if par is None:
                    break
                chain.append((par, cur))
                cur = par
                guard += 1
            chain.reverse()
            for s, t in chain:
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