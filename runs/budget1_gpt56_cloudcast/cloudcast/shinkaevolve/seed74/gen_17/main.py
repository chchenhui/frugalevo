# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Build a minimum-cost shared directed broadcast subnetwork.

    For normal broadcast fan-outs, a directed Steiner dynamic program chooses
    where destination subtrees should merge before paying for their common
    source-side links.  Thus a trunk is charged once even when it serves many
    clouds.  Very large terminal sets use a shared-tree fallback to bound the
    exponential subset state space.
    """
    import heapq

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    terminals = list(dict.fromkeys(dst for dst in dsts if dst != src))

    if not terminals:
        for dst in dsts:
            for partition in range(num_partitions):
                bc_topology.set_dst_partition_paths(dst, partition, [])
        return bc_topology

    h = nx.DiGraph()
    h.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        try:
            cost = float(data.get("cost"))
        except (TypeError, ValueError):
            continue
        if u != v and v != src and cost >= 0 and not pd.isna(cost):
            edge_data = dict(data)
            edge_data["cost"] = cost
            h.add_edge(u, v, **edge_data)

    if src not in h or any(dst not in h for dst in terminals):
        return bc_topology

    # For larger fan-outs, discard links that cannot participate in a
    # reasonably priced source-to-terminal corridor.  This keeps relay nodes
    # that can form a shared trunk while making the subset DP feasible for a
    # few more destinations than the unrestricted graph permits.
    if len(terminals) > 14:
        forward = nx.single_source_dijkstra_path_length(h, src, weight="cost")
        reverse = h.reverse(copy=False)
        terminal_distances = {
            dst: nx.single_source_dijkstra_path_length(reverse, dst, weight="cost")
            for dst in terminals
            if dst in forward
        }

        reduced = nx.DiGraph()
        reduced.add_nodes_from(h.nodes(data=True))
        for u, v, data in h.edges(data=True):
            if u not in forward:
                continue
            edge_cost = data["cost"]
            for dst, distances in terminal_distances.items():
                shortest = forward.get(dst, float("inf"))
                tail_distance = distances.get(v, float("inf"))
                # A modest slack admits alternate relay branches that can
                # become cheaper once their source-side links are shared.
                if forward[u] + edge_cost + tail_distance <= shortest * 1.35 + 1e-12:
                    reduced.add_edge(u, v, **dict(data))
                    break

        # Never let pruning remove the baseline independently cheapest route.
        for dst in terminals:
            if dst not in forward:
                continue
            try:
                path = nx.shortest_path(h, src, dst, weight="cost")
            except nx.NetworkXNoPath:
                continue
            reduced.add_edges_from(
                (u, v, dict(h[u][v])) for u, v in zip(path, path[1:])
            )
        h = reduced

    tree = nx.DiGraph()
    tree.add_node(src)

    if len(terminals) <= 16:
        nodes = list(h.nodes)
        node_index = {node: index for index, node in enumerate(nodes)}
        state_count = 1 << len(terminals)
        infinity = float("inf")
        costs = [[infinity] * len(nodes) for _ in range(state_count)]
        choices = [[None] * len(nodes) for _ in range(state_count)]

        for mask in range(1, state_count):
            if mask & (mask - 1) == 0:
                bit = mask.bit_length() - 1
                index = node_index[terminals[bit]]
                costs[mask][index] = 0.0
                choices[mask][index] = ("terminal",)
            else:
                submask = (mask - 1) & mask
                while submask:
                    other = mask ^ submask
                    if submask < other:
                        for index in range(len(nodes)):
                            candidate = costs[submask][index] + costs[other][index]
                            if candidate < costs[mask][index]:
                                costs[mask][index] = candidate
                                choices[mask][index] = ("merge", submask, other)
                    submask = (submask - 1) & mask

            queue = [(cost, index) for index, cost in enumerate(costs[mask])
                     if cost < infinity]
            heapq.heapify(queue)
            while queue:
                current_cost, current_index = heapq.heappop(queue)
                if current_cost != costs[mask][current_index]:
                    continue
                current = nodes[current_index]
                for predecessor in h.predecessors(current):
                    predecessor_index = node_index[predecessor]
                    candidate = current_cost + h[predecessor][current]["cost"]
                    if candidate < costs[mask][predecessor_index]:
                        costs[mask][predecessor_index] = candidate
                        choices[mask][predecessor_index] = ("edge", current_index)
                        heapq.heappush(queue, (candidate, predecessor_index))

        full_mask = state_count - 1
        source_index = node_index[src]
        if costs[full_mask][source_index] < infinity:
            visited = set()

            def reconstruct(mask, index):
                state = (mask, index)
                if state in visited:
                    return
                visited.add(state)
                choice = choices[mask][index]
                if choice is None or choice[0] == "terminal":
                    return
                if choice[0] == "merge":
                    reconstruct(choice[1], index)
                    reconstruct(choice[2], index)
                else:
                    next_index = choice[1]
                    u, v = nodes[index], nodes[next_index]
                    tree.add_edge(u, v, **dict(h[u][v]))
                    reconstruct(mask, next_index)

            reconstruct(full_mask, source_index)

    if len(terminals) > 16 or any(dst not in tree for dst in terminals):
        tree = nx.DiGraph()
        tree.add_node(src)
        remaining = set(terminals)
        while remaining:
            best = None
            for anchor in tree.nodes:
                lengths, paths = nx.single_source_dijkstra(h, anchor, weight="cost")
                for dst in remaining:
                    if dst in lengths:
                        candidate = (lengths[dst], str(dst), str(anchor), dst, paths[dst])
                        if best is None or candidate[:3] < best[:3]:
                            best = candidate
            if best is None:
                break
            _, _, _, dst, path = best
            tree.add_edges_from(
                (u, v, dict(h[u][v])) for u, v in zip(path, path[1:])
            )
            remaining.remove(dst)

    for dst in dsts:
        if dst == src:
            edges = []
        elif dst not in tree or not nx.has_path(tree, src, dst):
            continue
        else:
            path = nx.shortest_path(tree, src, dst)
            edges = [[u, v, dict(G[u][v])] for u, v in zip(path, path[1:])]
        for partition in range(num_partitions):
            bc_topology.set_dst_partition_paths(dst, partition, list(edges))

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