# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Find a minimum-cost shared broadcast tree with directed Steiner DP.

    For up to twelve distinct receivers, dynamic programming minimizes the
    cost of the union of all links rather than independently charging common
    broadcast prefixes.  The resulting shared edge set is converted back into
    a valid source-to-receiver path for every partition.  Larger receiver sets
    use shortest paths to keep runtime predictably bounded.
    """
    import heapq

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    h.remove_edges_from(
        [(u, v) for u, v, data in h.edges(data=True)
         if data.get("cost") is None]
    )

    terminals = list(dict.fromkeys(dst for dst in dsts if dst != src))
    direct_paths = {
        dst: nx.dijkstra_path(h, src, dst, weight="cost")
        for dst in terminals
    }

    # Directed Steiner DP is exact but exponential in terminal count.  Twelve
    # terminals is inexpensive for the small cloud-region graphs used here.
    if len(terminals) <= 12 and terminals:
        nodes = list(h.nodes())
        inf = float("inf")
        states = 1 << len(terminals)
        dp = [{node: inf for node in nodes} for _ in range(states)]
        action = [{} for _ in range(states)]

        for mask in range(1, states):
            # Joining two already-optimized terminal groups at the same node.
            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if other and sub < other:
                    for node in nodes:
                        value = dp[sub][node] + dp[other][node]
                        if value < dp[mask][node]:
                            dp[mask][node] = value
                            action[mask][node] = ("join", sub, other)
                sub = (sub - 1) & mask

            if mask & (mask - 1) == 0:
                terminal = terminals[mask.bit_length() - 1]
                dp[mask][terminal] = 0.0
                action[mask][terminal] = ("terminal",)

            # Relax through arbitrary directed paths.  Running Dijkstra on the
            # reversed graph computes min_u(cost(v -> u) + dp[mask][u]).
            queue = []
            for node in nodes:
                if dp[mask][node] < inf:
                    heapq.heappush(queue, (dp[mask][node], node))

            while queue:
                value, node = heapq.heappop(queue)
                if value != dp[mask][node]:
                    continue
                for predecessor in h.predecessors(node):
                    candidate = value + float(h[predecessor][node]["cost"])
                    if candidate < dp[mask][predecessor]:
                        dp[mask][predecessor] = candidate
                        action[mask][predecessor] = ("edge", node)
                        heapq.heappush(queue, (candidate, predecessor))

        all_mask = states - 1
        used_edges = set()

        def collect(mask, node):
            choice = action[mask].get(node)
            if choice is None or choice[0] == "terminal":
                return
            if choice[0] == "edge":
                nxt = choice[1]
                used_edges.add((node, nxt))
                collect(mask, nxt)
            else:
                collect(choice[1], node)
                collect(choice[2], node)

        collect(all_mask, src)
        tree = nx.DiGraph()
        for u, v in used_edges:
            tree.add_edge(u, v)

        # DP reconstruction is a shared directed subgraph; its individual
        # root-to-terminal routes are the paths required by BroadCastTopology.
        try:
            selected_paths = {
                dst: nx.shortest_path(tree, src, dst) for dst in terminals
            }
        except nx.NetworkXNoPath:
            selected_paths = direct_paths
    else:
        selected_paths = direct_paths

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        if dst == src:
            continue
        for u, v in zip(selected_paths[dst], selected_paths[dst][1:]):
            for partition in range(num_partitions):
                bc_topology.append_dst_partition_path(
                    dst, partition, [u, v, G[u][v]]
                )
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
