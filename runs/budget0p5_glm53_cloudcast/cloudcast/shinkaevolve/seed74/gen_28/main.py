# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import heapq
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions, discount=0.0):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    # ensure every edge has a finite positive cost for pathfinding
    for s, t, data in h.edges(data=True):
        if data.get("cost") is None:
            h[s][t]["cost"] = 1e9
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    used_edges = set()

    def edge_cost(s, t):
        # edges already paid for are (nearly) free for later destinations
        return discount if (s, t) in used_edges else h[s][t]["cost"]

    def shortest_with_discount(dst):
        """Manual Dijkstra with discounted weights on already-paid edges."""
        dist = {src: 0.0}
        prev = {}
        pq = [(0.0, src)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break
            for v in h[u]:
                nd = d + edge_cost(u, v)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dst not in visited:
            return None, float("inf")
        path = [dst]
        while path[-1] != src:
            if path[-1] not in prev:
                return None, float("inf")
            path.append(prev[path[-1]])
        path.reverse()
        return path, dist.get(dst, float("inf"))

    # baseline undiscounted shortest-path cost per destination
    orig_cost = {}
    for d in dsts:
        try:
            orig_cost[d] = nx.shortest_path_length(h, src, d, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            orig_cost[d] = float("inf")

    remaining = [d for d in dsts if orig_cost[d] != float("inf")]
    unreachable = [d for d in dsts if orig_cost[d] == float("inf")]

    while remaining:
        # Greedy dynamic ordering: lock in the destination with the
        # largest immediate savings from already-shared (paid) edges.
        best_dst = None
        best_saving = -float("inf")
        best_info = None
        for d in remaining:
            path, dcost = shortest_with_discount(d)
            if path is None:
                continue
            saving = orig_cost[d] - dcost
            if saving > best_saving:
                best_saving = saving
                best_dst = d
                best_info = (path, dcost)
        if best_dst is None:
            unreachable.extend(remaining)
            break
        path, _ = best_info
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            used_edges.add((s, t))
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(best_dst, j, [s, t, G[s][t]])
        remaining = [d for d in remaining if d != best_dst]

    for d in unreachable:
        for j in range(bc_topology.num_partitions):
            bc_topology.append_dst_partition_path(d, j, [])

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