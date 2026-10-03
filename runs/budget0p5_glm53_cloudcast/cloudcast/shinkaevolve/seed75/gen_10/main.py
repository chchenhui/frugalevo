# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h))))

    # Clean out edges with missing cost data (avoid routing through unknown-cost links)
    for u, v in list(h.edges()):
        if h[u][v].get("cost") is None:
            h.remove_edge(u, v)

    # --- Step 1: Build a shared low-cost multicast overlay (Steiner-like) ---
    # Compute pairwise shortest-path distances from src to all dsts.
    dist, paths_from_src = nx.single_source_dijkstra(h, src, weight="cost")

    # Build a metric closure over {src} U dsts, then a MST on it to share links.
    terminals = [src] + [d for d in dsts if d in dist]
    closure = nx.Graph()
    # Precompute pairwise shortest paths between terminals
    pair_paths = {}
    all_pairs_dists = {}
    for t in terminals:
        d_t, p_t = nx.single_source_dijkstra(h, t, weight="cost")
        all_pairs_dists[t] = d_t
        for u in terminals:
            if u == t or u not in d_t:
                continue
            closure.add_edge(t, u, weight=d_t[u])
            pair_paths[(t, u)] = p_t[u]

    mst = nx.minimum_spanning_tree(closure, weight="weight")

    # Expand the MST back into the original graph -> overlay edges used by broadcast tree
    overlay_edges = set()
    tree_paths = {}  # (root, dst) -> path in original graph
    for t, u, data in mst.edges(data=True):
        p = pair_paths.get((t, u)) or list(reversed(pair_paths[(u, t)]))
        tree_paths[(t, u)] = p
        tree_paths[(u, t)] = list(reversed(p))
        for i in range(len(p) - 1):
            overlay_edges.add((p[i], p[i + 1]))

    # For each terminal, find its path in the overlay tree from src
    # by walking the tree structure.
    tree_adj = {t: [] for t in terminals}
    for t, u in mst.edges():
        tree_adj[t].append(u)
        tree_adj[u].append(t)

    overlay_path_to = {src: [src]}
    visited = {src}
    queue = [src]
    while queue:
        node = queue.pop(0)
        for nb in tree_adj[node]:
            if nb in visited:
                continue
            visited.add(nb)
            seg = tree_paths[(node, nb)]
            overlay_path_to[nb] = overlay_path_to[node] + seg[1:]
            queue.append(nb)

    # --- Step 2: Build the BroadCastTopology with partition-level load balancing ---
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        # Base overlay path (shared, low-cost)
        base_path = overlay_path_to.get(dst) or paths_from_src.get(dst)
        if base_path is None:
            # Fallback: direct dijkstra on full graph
            base_path = nx.dijkstra_path(G, src, dst, weight="cost")

        # Find k alternative edge-disjoint paths to balance load across networks
        try:
            alt_paths = list(nx.shortest_simple_paths(h, src, dst, weight="cost"))
        except Exception:
            alt_paths = []

        # Deduplicate and cap the number of alternatives
        candidate_paths = []
        seen = set()
        for p in [base_path] + alt_paths:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                candidate_paths.append(p)
            if len(candidate_paths) >= num_partitions:
                break

        for j in range(num_partitions):
            # Round-robin assignment of partitions across candidate paths,
            # prioritizing the shared overlay path first (cheapest aggregate).
            if len(candidate_paths) == 1 or j < len(candidate_paths) - 1:
                path = base_path if j == 0 else candidate_paths[j % len(candidate_paths)]
            else:
                path = candidate_paths[j % len(candidate_paths)]
            if len(candidate_paths) == 1:
                path = base_path
            for i in range(0, len(path) - 1):
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