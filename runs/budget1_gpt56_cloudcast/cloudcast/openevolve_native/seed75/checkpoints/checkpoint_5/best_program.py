# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Select a low-cost shared broadcast tree from several relay-path seeds.

    The algorithm charges an edge only once while constructing a tree.  It
    evaluates independent shortest paths plus up to three low-cost initial
    paths for every destination, then greedily attaches remaining destinations
    using zero marginal cost for links already present in the broadcast tree.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(dsts))

    if not terminals:
        return bc_topology

    def edge_cost(u, v, data):
        """Return a numeric link price, excluding links without a price."""
        value = data.get("cost")
        return float(value) if value is not None else float("inf")

    def tree_cost(edges):
        """Return the distinct-link cost of a candidate broadcast tree."""
        return sum(edge_cost(u, v, h[u][v]) for u, v in edges)

    def build_tree(seed_path):
        """Attach terminals by repeatedly choosing the cheapest marginal path."""
        edges = set(zip(seed_path, seed_path[1:]))
        attached = {dst for dst in terminals if dst in seed_path}

        while len(attached) < len(terminals):
            def marginal_cost(u, v, data):
                return 0.0 if (u, v) in edges else edge_cost(u, v, data)

            tree_nodes = {src}
            for u, v in edges:
                tree_nodes.add(u)
                tree_nodes.add(v)

            lengths, paths = nx.multi_source_dijkstra(
                h, list(tree_nodes), weight=marginal_cost
            )
            choices = [
                (lengths[dst], dst, paths[dst])
                for dst in terminals
                if dst not in attached and dst in paths
            ]
            if not choices:
                raise nx.NetworkXNoPath("A destination is unreachable from source")

            _, _, path = min(choices, key=lambda item: item[0])
            edges.update(zip(path, path[1:]))
            attached.update(dst for dst in terminals if dst in path)

        return edges

    shortest_paths = {}
    baseline = set()
    for dst in terminals:
        path = nx.dijkstra_path(h, src, dst, weight=edge_cost)
        shortest_paths[dst] = [path]
        baseline.update(zip(path, path[1:]))

    candidates = [baseline]

    # A slightly longer first branch can be globally cheaper when later
    # destinations reuse it.  Limit alternatives to keep search bounded.
    for dst in terminals:
        try:
            alternatives = nx.shortest_simple_paths(h, src, dst, weight=edge_cost)
            for index, path in enumerate(alternatives):
                if index >= 3:
                    break
                shortest_paths[dst].append(path)
        except nx.NetworkXNoPath:
            raise

        for path in shortest_paths[dst]:
            candidates.append(build_tree(path))

    best_edges = min(candidates, key=tree_cost)
    tree_graph = nx.DiGraph()
    tree_graph.add_edges_from(best_edges)

    for dst in terminals:
        path = nx.dijkstra_path(tree_graph, src, dst, weight=edge_cost)
        for u, v in zip(path, path[1:]):
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
