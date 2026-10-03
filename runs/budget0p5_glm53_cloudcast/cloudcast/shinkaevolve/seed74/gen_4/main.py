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

    # Build a low-cost shared broadcast tree (Steiner-tree heuristic):
    # start from the shortest path to the first destination, then grow the
    # tree by repeatedly connecting each remaining destination to the tree
    # via its cheapest path. Shared trunk edges are reused across all
    # destinations, reducing redundant transfers and overall cost.
    tree_nodes = {src}
    tree_edges = set()

    # precompute shortest paths from source (used for first dst)
    try:
        dist, spaths = nx.single_source_dijkstra(h, src, weight="cost")
        # order destinations by shortest-path cost (closest first) so that
        # later destinations can piggyback on the already-built tree
        sorted_dsts = sorted(dsts, key=lambda d: dist.get(d, float("inf")))

        for dst in sorted_dsts:
            if dst in tree_nodes:
                path = None
            else:
                # find cheapest way to attach dst to current tree
                best_path = None
                best_cost = float("inf")
                # run Dijkstra from dst on reversed graph to reach tree nodes
                rh = h.reverse(copy=True)
                rdist, rpaths = nx.single_source_dijkstra(rh, dst, weight="cost")
                for tn in tree_nodes:
                    if tn in rdist and rdist[tn] < best_cost:
                        best_cost = rdist[tn]
                        best_path = list(reversed(rpaths[tn]))
                path = best_path

            if path is None and dst not in tree_nodes:
                path = spaths.get(dst)

            if not path or path[0] != src:
                # fall back to plain shortest path from src
                path = nx.dijkstra_path(h, src, dst, weight="cost")

            for i in range(len(path) - 1):
                s, t = path[i], path[i + 1]
                if s in tree_nodes and t in tree_nodes and (s, t) in tree_edges:
                    continue
                tree_nodes.add(t)
                tree_edges.add((s, t))
                for j in range(bc_topology.num_partitions):
                    bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
    except Exception:
        # robust fallback: independent shortest paths per destination
        for dst in dsts:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
            for i in range(0, len(path) - 1):
                s, t = path[i], path[i + 1]
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