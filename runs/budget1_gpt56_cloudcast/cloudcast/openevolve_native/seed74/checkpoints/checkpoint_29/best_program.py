# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Compute an exact low-cost directed multicast tree for up to 12 targets.

    The dynamic program minimizes the cost of the union of links used to reach
    all destinations.  It explicitly considers where a broadcast tree should
    branch, rather than greedily committing to the first destination path.
    Each partition then reuses the resulting shared multicast tree.
    """
    import heapq

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(dsts))

    # A missing price is not a usable transfer route.  Keeping only priced
    # links also prevents NetworkX from treating None as an accidental weight.
    h = nx.DiGraph()
    h.add_nodes_from(G.nodes)
    for u, v, data in G.edges(data=True):
        cost = data.get("cost")
        if cost is not None and u != v:
            h.add_edge(u, v, cost=float(cost))

    if src not in h or not terminals:
        return bc_topology

    # For unusually large terminal sets, shortest paths are a predictable,
    # bounded-time fallback. Typical broadcast configurations have few targets,
    # where the exact Steiner DP below is substantially better than greedy.
    if len(terminals) > 12:
        _, paths = nx.single_source_dijkstra(h, src, weight="cost")
        for dst in dsts:
            if dst not in paths:
                continue
            path = paths[dst]
            edge_path = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
            for partition in range(num_partitions):
                bc_topology.set_dst_partition_paths(dst, partition, list(edge_path))
        return bc_topology

    nodes = list(h.nodes)
    full_mask = (1 << len(terminals)) - 1
    inf = float("inf")
    dp = [{node: inf for node in nodes} for _ in range(full_mask + 1)]
    split = [{} for _ in range(full_mask + 1)]
    next_hop = [{} for _ in range(full_mask + 1)]

    # dp[mask][v] is the cheapest directed tree rooted at v spanning mask.
    # A singleton starts at its terminal; reverse Dijkstra extends that tree
    # backwards over possible incoming links.
    for mask in range(1, full_mask + 1):
        if mask & (mask - 1) == 0:
            terminal = terminals[mask.bit_length() - 1]
            if terminal in h:
                dp[mask][terminal] = 0.0
        else:
            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if sub < other:
                    for node in nodes:
                        candidate = dp[sub][node] + dp[other][node]
                        if candidate < dp[mask][node]:
                            dp[mask][node] = candidate
                            split[mask][node] = (sub, other)
                sub = (sub - 1) & mask

        # Multi-source reverse Dijkstra implements
        # dp[mask][v] = min(v->w cost + dp[mask][w]).
        heap = []
        serial = 0
        for node in nodes:
            if dp[mask][node] < inf:
                heapq.heappush(heap, (dp[mask][node], serial, node))
                serial += 1

        while heap:
            distance, _, child = heapq.heappop(heap)
            if distance != dp[mask][child]:
                continue
            for parent in h.predecessors(child):
                candidate = distance + h[parent][child]["cost"]
                if candidate < dp[mask][parent]:
                    dp[mask][parent] = candidate
                    next_hop[mask][parent] = child
                    heapq.heappush(heap, (candidate, serial, parent))
                    serial += 1

    if dp[full_mask][src] == inf:
        return bc_topology

    selected_edges = set()
    visited_states = set()

    def collect(mask, node):
        """Recover the selected Steiner-tree links from DP decisions."""
        state = (mask, node)
        if state in visited_states:
            return
        visited_states.add(state)

        if node in next_hop[mask]:
            child = next_hop[mask][node]
            selected_edges.add((node, child))
            collect(mask, child)
        elif node in split[mask]:
            left, right = split[mask][node]
            collect(left, node)
            collect(right, node)

    collect(full_mask, src)

    tree = nx.DiGraph()
    tree.add_edges_from(selected_edges)
    for dst in dsts:
        try:
            path = nx.shortest_path(tree, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        edge_path = [[u, v, G[u][v]] for u, v in zip(path, path[1:])]
        for partition in range(num_partitions):
            bc_topology.set_dst_partition_paths(dst, partition, list(edge_path))

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
