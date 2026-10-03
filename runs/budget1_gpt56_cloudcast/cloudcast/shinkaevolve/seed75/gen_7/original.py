# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Build a shared directed multicast tree rooted at ``src``.

    Rather than independently sending every destination's data from the source,
    each iteration attaches the cheapest remaining destination to any node
    already reached by the broadcast.  The resulting prefixes are shared by
    all downstream destinations.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    # Links without a price cannot safely participate in cost minimization.
    h.remove_edges_from(
        [(u, v) for u, v, data in h.edges(data=True) if data.get("cost") is None]
    )

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    tree_nodes = {src}
    tree_paths = {src: [src]}
    remaining = set(dsts)

    while remaining:
        distances, paths = nx.multi_source_dijkstra(
            h, tree_nodes, weight="cost"
        )
        reachable = [dst for dst in remaining if dst in distances]
        if not reachable:
            missing = ", ".join(str(dst) for dst in remaining)
            raise nx.NetworkXNoPath(
                f"No priced broadcast route from {src} to: {missing}"
            )

        # Cheapest incremental branch first maximizes reuse of the current tree.
        dst = min(reachable, key=lambda node: (distances[node], str(node)))
        branch = paths[dst]
        attachment = branch[0]
        full_path = tree_paths[attachment] + branch[1:]

        # Record source paths for all newly reached branch nodes, allowing
        # later destinations to attach at any of them at zero prefix cost.
        for index, node in enumerate(branch[1:], start=1):
            tree_nodes.add(node)
            tree_paths.setdefault(node, tree_paths[attachment] + branch[1:index + 1])

        for s, t in zip(full_path, full_path[1:]):
            edge = [s, t, G[s][t]]
            for partition in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, partition, edge)

        remaining.remove(dst)

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