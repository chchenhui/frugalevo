# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Build a shared multicast tree instead of independently routing every
    destination from the source.  Every new destination is attached through
    the cheapest path from any node that already receives the broadcast.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    remaining = set(dsts)
    remaining.discard(src)
    tree_nodes = {src}
    root_paths = {src: [src]}

    while remaining:
        distances, paths = nx.multi_source_dijkstra(h, tree_nodes, weight="cost")
        reachable = [dst for dst in remaining if dst in distances]
        if not reachable:
            raise nx.NetworkXNoPath(
                "No broadcast path from %s to destinations %s"
                % (src, sorted(remaining))
            )

        dst = min(reachable, key=lambda node: distances[node])
        connector = paths[dst]
        attachment = connector[0]
        full_path = root_paths[attachment] + connector[1:]

        # A connector may pass through other requested destinations.  They
        # already receive the same broadcast and therefore need no separate
        # upstream transfer.
        for index, node in enumerate(connector):
            node_path = root_paths[attachment] + connector[1:index + 1]
            if node not in root_paths:
                root_paths[node] = node_path
            tree_nodes.add(node)

            if node in remaining:
                remaining.remove(node)
                for partition in range(bc_topology.num_partitions):
                    for edge_index in range(len(node_path) - 1):
                        s, t = node_path[edge_index], node_path[edge_index + 1]
                        bc_topology.append_dst_partition_path(
                            node, partition, [s, t, G[s][t]]
                        )

        root_paths[dst] = full_path

    # A destination colocated with the source has no network edges, but its
    # partition paths should still be represented as completed routes.
    if src in dsts:
        for partition in range(bc_topology.num_partitions):
            bc_topology.set_dst_partition_paths(src, partition, [])

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