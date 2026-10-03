# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Find a shared directed Steiner broadcast network rooted at ``src``.

    For normal broadcast fan-outs this uses a subset dynamic program: each
    state is the cheapest network rooted at a node that reaches a subset of
    destinations.  States can either merge at a relay or traverse an outgoing
    link to a cheaper downstream state.  This finds globally shared relays
    rather than committing to the next locally cheapest destination path.
    """
    import heapq

    h = G.copy()
    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    if src not in h:
        return bc_topology

    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    invalid_edges = []
    for u, v, data in h.edges(data=True):
        try:
            cost = float(data.get("cost"))
            if cost < 0 or cost != cost:
                invalid_edges.append((u, v))
            else:
                data["cost"] = cost
        except (TypeError, ValueError):
            invalid_edges.append((u, v))
    h.remove_edges_from(invalid_edges)

    terminals = list(dict.fromkeys(dst for dst in dsts if dst != src and dst in h))
    nodes = list(h.nodes)

    # Exact Steiner DP is inexpensive for the usual small fan-out, while the
    # greedy fallback prevents exponential planning time for large workloads.
    if len(terminals) <= 12:
        terminal_bits = {dst: 1 << index for index, dst in enumerate(terminals)}
        state_count = 1 << len(terminals)
        dp = [{node: float("inf") for node in nodes} for _ in range(state_count)]
        choice = [{} for _ in range(state_count)]

        for mask in range(1, state_count):
            if mask & (mask - 1) == 0:
                terminal = terminals[mask.bit_length() - 1]
                dp[mask][terminal] = 0.0
                choice[mask][terminal] = ("terminal",)
            else:
                subset = (mask - 1) & mask
                while subset:
                    other = mask ^ subset
                    if subset < other:
                        for node in nodes:
                            candidate = dp[subset][node] + dp[other][node]
                            if candidate < dp[mask][node]:
                                dp[mask][node] = candidate
                                choice[mask][node] = ("merge", subset, other)
                    subset = (subset - 1) & mask

            # Multi-source reverse Dijkstra propagates each merged subtree to
            # every upstream relay that can send data into it.
            queue = [
                (distance, node)
                for node, distance in dp[mask].items()
                if distance != float("inf")
            ]
            heapq.heapify(queue)
            while queue:
                distance, node = heapq.heappop(queue)
                if distance != dp[mask][node]:
                    continue
                for predecessor, _, edge in h.in_edges(node, data=True):
                    candidate = distance + edge["cost"]
                    if candidate < dp[mask][predecessor]:
                        dp[mask][predecessor] = candidate
                        choice[mask][predecessor] = ("edge", node)
                        heapq.heappush(queue, (candidate, predecessor))

        full_mask = state_count - 1
        selected = nx.DiGraph()

        def collect_edges(mask, node, visited):
            state = (mask, node)
            if state in visited or node not in choice[mask]:
                return
            visited.add(state)
            decision = choice[mask][node]
            if decision[0] == "merge":
                collect_edges(decision[1], node, visited)
                collect_edges(decision[2], node, visited)
            elif decision[0] == "edge":
                successor = decision[1]
                selected.add_edge(node, successor, **G[node][successor])
                collect_edges(mask, successor, visited)

        if dp[full_mask].get(src, float("inf")) != float("inf"):
            collect_edges(full_mask, src, set())
    else:
        selected = nx.DiGraph()
        selected.add_node(src)
        pending = set(terminals)
        while pending:
            lengths, paths = nx.multi_source_dijkstra(
                h, list(selected.nodes), weight="cost"
            )
            candidates = [
                (lengths[dst], dst, paths[dst])
                for dst in pending
                if dst in lengths
            ]
            if not candidates:
                break
            _, _, path = min(candidates, key=lambda item: item[0])
            for u, v in zip(path, path[1:]):
                selected.add_edge(u, v, **G[u][v])
            pending.difference_update(path)

    for dst in dsts:
        if dst == src:
            edge_path = []
        elif dst in selected and nx.has_path(selected, src, dst):
            route = nx.shortest_path(selected, src, dst, weight="cost")
            edge_path = [[u, v, G[u][v]] for u, v in zip(route, route[1:])]
        else:
            edge_path = []

        for partition in range(bc_topology.num_partitions):
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