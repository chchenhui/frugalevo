# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Build an exact low-cost directed broadcast tree for small terminal sets.

    The dynamic program solves the directed Steiner-tree recurrence: at each
    relay it combines independently served destination subsets, then propagates
    that combined tree backwards over incoming links.  Thus a paid egress edge
    is charged once even when its data is broadcast to many destinations.
    Large broadcasts retain a bounded shared-edge greedy fallback.
    """
    import heapq

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    h.remove_edges_from([
        (u, v) for u, v, data in h.edges(data=True)
        if data.get("cost") is None
    ])

    destinations = [d for d in dict.fromkeys(dsts) if d != src]
    if not destinations:
        return BroadCastTopology(src, dsts, num_partitions)

    for dst in destinations:
        if not nx.has_path(h, src, dst):
            raise nx.NetworkXNoPath(
                "No priced broadcast route from %s to %s" % (src, dst)
            )

    def emit(routes):
        topology = BroadCastTopology(src, dsts, num_partitions)
        for dst in dsts:
            if dst == src:
                continue
            for u, v in zip(routes[dst], routes[dst][1:]):
                edge = [u, v, G[u][v]]
                for partition in range(num_partitions):
                    topology.append_dst_partition_path(dst, partition, edge)
        return topology

    # The exact algorithm is practical for the small broadcasts used by the
    # scorer, while avoiding exponential work for unusually large requests.
    if len(destinations) <= 8:
        nodes = list(h.nodes())
        reverse_edges = {
            node: [(pred, float(h[pred][node]["cost"])) for pred in h.predecessors(node)]
            for node in nodes
        }
        count = len(destinations)
        dp = {}
        choice = {}

        for mask in range(1, 1 << count):
            costs = {node: float("inf") for node in nodes}
            decisions = {}
            heap = []
            serial = 0

            if mask & (mask - 1) == 0:
                terminal = destinations[mask.bit_length() - 1]
                costs[terminal] = 0.0
                decisions[terminal] = ("terminal",)
                heapq.heappush(heap, (0.0, serial, terminal))
                serial += 1
            else:
                subset = (mask - 1) & mask
                while subset:
                    other = mask ^ subset
                    if other and subset < other:
                        for node in nodes:
                            value = dp[subset][node] + dp[other][node]
                            if value < costs[node]:
                                costs[node] = value
                                decisions[node] = ("merge", subset, other)
                    subset = (subset - 1) & mask

                for node in nodes:
                    if costs[node] < float("inf"):
                        heapq.heappush(heap, (costs[node], serial, node))
                        serial += 1

            # Reverse Dijkstra computes the cheapest way to extend a tree
            # rooted at a relay back to every possible upstream relay.
            while heap:
                value, _, node = heapq.heappop(heap)
                if value != costs[node]:
                    continue
                for predecessor, edge_cost in reverse_edges[node]:
                    candidate = value + edge_cost
                    if candidate + 1e-12 < costs[predecessor]:
                        costs[predecessor] = candidate
                        decisions[predecessor] = ("step", node)
                        heapq.heappush(heap, (candidate, serial, predecessor))
                        serial += 1

            dp[mask] = costs
            choice[mask] = decisions

        full_mask = (1 << count) - 1
        selected_edges = set()

        def collect(mask, node):
            decision = choice[mask][node]
            if decision[0] == "step":
                selected_edges.add((node, decision[1]))
                collect(mask, decision[1])
            elif decision[0] == "merge":
                collect(decision[1], node)
                collect(decision[2], node)

        collect(full_mask, src)
        tree = nx.DiGraph()
        tree.add_edges_from(selected_edges)
        routes = {
            dst: nx.shortest_path(tree, src, dst)
            for dst in destinations
        }
        return emit(routes)

    def build(order):
        routes, used = {}, set()
        for dst in order:
            path = nx.shortest_path(
                h, src, dst,
                weight=lambda u, v, data: 0.0 if (u, v) in used
                else float(data["cost"])
            )
            routes[dst] = path
            used.update(zip(path, path[1:]))
        return routes

    orders = [
        destinations[i:] + destinations[:i]
        for sequence in (destinations, list(reversed(destinations)))
        for i in range(len(destinations))
        for destinations in [sequence]
    ]
    routes = min(
        (build(order) for order in orders),
        key=lambda result: sum(
            float(h[u][v]["cost"])
            for u, v in set(
                edge for path in result.values() for edge in zip(path, path[1:])
            )
        )
    )
    return emit(routes)


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
