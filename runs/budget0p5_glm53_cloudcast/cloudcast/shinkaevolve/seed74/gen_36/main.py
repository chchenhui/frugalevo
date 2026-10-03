# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions, k_paths=4, discount=0.25, refine_passes=3):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def edge_cost(s, t):
        return h[s][t].get("cost") or 0.0

    def build_discounted_graph(used_edges, disc):
        hd = h.copy()
        for (s, t) in hd.edges():
            c = edge_cost(s, t)
            if (s, t) in used_edges:
                c = disc * c
            hd[s][t]["cost"] = c
        return hd

    def topology_total_cost(topo):
        total = 0.0
        for dst in dsts:
            for j in range(num_partitions):
                p = topo.paths.get(dst, {}).get(str(j))
                if not p:
                    continue
                for e in p:
                    total += e[2].get("cost") or 0.0
        return total

    # Baseline pass 0: plain Dijkstra, no discount (always feasible fallback).
    discount_schedule = [0.0, discount, max(discount, 0.5)]
    best_topology = None
    best_total = float("inf")
    used_edges = set()

    for pass_idx in range(max(refine_passes, len(discount_schedule))):
        disc = discount_schedule[min(pass_idx, len(discount_schedule) - 1)]
        hd = build_discounted_graph(used_edges, disc)
        bc_topology = BroadCastTopology(src, dsts, num_partitions)
        new_used = set()
        local_used = set()

        for dst in dsts:
            # Enumerate up to k candidate paths on the discounted graph.
            candidates = []
            try:
                gen = nx.shortest_simple_paths(hd, src, dst, weight="cost")
                for p in gen:
                    candidates.append(p)
                    if len(candidates) >= k_paths:
                        break
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                candidates = []
            if not candidates:
                try:
                    candidates = [nx.dijkstra_path(hd, src, dst, weight="cost")]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

            # Pick path minimizing incremental cost: edges already used in
            # this pass (by this dst's earlier partitions or other dsts)
            # only cost the discounted price.
            def incremental_cost(path):
                tot = 0.0
                seen_local = set()
                for i in range(len(path) - 1):
                    e = (path[i], path[i + 1])
                    c = edge_cost(*e)
                    if e in local_used:
                        c = disc * c
                    elif e in seen_local:
                        c = disc * c
                    tot += c
                    seen_local.add(e)
                return tot

            best_path = min(candidates, key=incremental_cost)

            # All partitions of this destination follow the same consolidated
            # route: the trunk is paid once, subsequent partitions reuse it.
            segs = []
            for i in range(len(best_path) - 1):
                s, t = best_path[i], best_path[i + 1]
                segs.append([s, t, G[s][t]])
                local_used.add((s, t))
                new_used.add((s, t))
            for j in range(num_partitions):
                bc_topology.set_dst_partition_paths(dst, j, list(segs))

        total = topology_total_cost(bc_topology)
        if total < best_total:
            best_total = total
            best_topology = bc_topology

        if new_used == used_edges and pass_idx >= 1:
            break
        used_edges = new_used

    if best_topology is None:
        best_topology = BroadCastTopology(src, dsts, num_partitions)

    return best_topology


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