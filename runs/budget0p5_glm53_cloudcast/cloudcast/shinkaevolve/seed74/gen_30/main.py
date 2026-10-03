# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions, k_paths=4):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Track how many partitions are already assigned to each edge so we can
    # spread load across parallel network paths.
    edge_load = {}

    def path_cost(path):
        return sum(h[path[i]][path[i + 1]].get("cost") or 0.0
                   for i in range(len(path) - 1))

    for dst in dsts:
        try:
            path_generator = nx.shortest_simple_paths(h, src, dst, weight="cost")
            candidates = []
            for p in path_generator:
                candidates.append(p)
                if len(candidates) >= k_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            candidates = [nx.dijkstra_path(h, src, dst, weight="cost")]

        if not candidates:
            candidates = [nx.dijkstra_path(h, src, dst, weight="cost")]

        # Score each candidate path: cheap and lightly-loaded edges are better.
        def score(path):
            c = path_cost(path)
            load_penalty = 0.0
            for i in range(len(path) - 1):
                load_penalty += edge_load.get((path[i], path[i + 1]), 0)
            # avoid division by zero
            return (c + 1e-9) / (1.0 + load_penalty)

        candidates = sorted(candidates, key=score)

        # Greedy round-robin assignment of partitions over the ranked paths,
        # balancing load across the parallel routes.
        for j in range(bc_topology.num_partitions):
            path = candidates[j % len(candidates)]
            for i in range(0, len(path) - 1):
                s, t = path[i], path[i + 1]
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
                edge_load[(s, t)] = edge_load.get((s, t), 0) + 1

    # ---- Local-search improvement pass ----
    # For each destination, try to reroute it on a graph where edges already
    # used by other destinations are discounted (shared broadcast trunks are
    # effectively amortized). Keep the swap only if the total unique-edge cost
    # of this destination decreases.
    def edge_cost(s, t):
        c = h[s][t].get("cost")
        if c is None:
            # unpriced links: treat as expensive to avoid free rides
            return max((h[u][v].get("cost") or 0.0) for u, v in h.edges() if h.number_of_edges() > 0) + 1.0
        return c

    # Record edges used per destination
    def dst_edges(dst):
        used = set()
        for j in range(num_partitions):
            for e in bc_topology.paths.get(dst, {}).get(str(j)) or []:
                used.add((e[0], e[1]))
        return used

    all_dst_edges = {d: dst_edges(d) for d in dsts}

    for dst in dsts:
        other_edges = set()
        for d in dsts:
            if d != dst:
                other_edges |= all_dst_edges[d]

        # Build a modified graph discounting edges shared with other dsts
        h_mod = h.copy()
        discount = 0.5
        for u, v in h_mod.edges():
            base = edge_cost(u, v)
            if (u, v) in other_edges:
                h_mod[u][v]["cost"] = base * (1.0 - discount)
            else:
                h_mod[u][v]["cost"] = base

        try:
            new_path = nx.dijkstra_path(h_mod, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        def total_edge_cost(edges):
            return sum(edge_cost(s, t) for s, t in edges)

        cur_cost = total_edge_cost(all_dst_edges[dst])
        new_edges = set()
        for i in range(len(new_path) - 1):
            new_edges.add((new_path[i], new_path[i + 1]))
        new_cost = total_edge_cost(new_edges)

        if new_cost < cur_cost:
            # Re-assign all partitions of this dst to the new path
            for j in range(num_partitions):
                segs = []
                for i in range(len(new_path) - 1):
                    s, t = new_path[i], new_path[i + 1]
                    segs.append([s, t, G[s][t]])
                bc_topology.set_dst_partition_paths(dst, j, segs)
            all_dst_edges[dst] = new_edges

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