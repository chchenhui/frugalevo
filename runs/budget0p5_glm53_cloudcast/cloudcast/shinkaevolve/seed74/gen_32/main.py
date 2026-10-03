# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions, discount=0.0, num_refine=3):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    # drop edges with no cost info to avoid infinite weights
    for u, v, d in list(h.edges(data=True)):
        if d.get("cost") is None:
            h.remove_edge(u, v)

    import heapq

    def edge_cost(used, s, t):
        # already-paid (shared trunk) edges are free for later destinations
        return discount if (s, t) in used else h[s][t]["cost"]

    def shortest_with_discount(used, dst):
        """Dijkstra on the graph where already-used edges cost `discount`."""
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
                nd = d + edge_cost(used, u, v)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return None, float("inf")
        path = [dst]
        while path[-1] != src:
            if path[-1] not in prev:
                return None, float("inf")
            path.append(prev[path[-1]])
        path.reverse()
        return path, dist[dst]

    # original (undiscounted) shortest path costs, for reachability + ordering
    orig_cost = {}
    for d in dsts:
        try:
            orig_cost[d] = nx.shortest_path_length(h, src, d, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            orig_cost[d] = float("inf")

    unreachable = [d for d in dsts if orig_cost[d] == float("inf")]
    reachable = [d for d in dsts if orig_cost[d] != float("inf")]

    # Warm start: seed used_edges with the union of edges on each dst's plain
    # shortest path, then refine so trunks shared by multiple dsts dominate.
    used_edges = set()
    for d in reachable:
        try:
            p = nx.dijkstra_path(h, src, d, weight="cost")
            for i in range(len(p) - 1):
                used_edges.add((p[i], p[i + 1]))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    chosen = {d: None for d in reachable}
    for _ in range(num_refine):
        # Each pass re-routes every destination given the current shared trunk;
        # edges used by any destination in the last pass stay free next pass.
        pass_used = set()
        candidate_paths = {}
        for d in reachable:
            path, dcost = shortest_with_discount(used_edges, d)
            candidate_paths[d] = path
            if path is not None:
                for i in range(len(path) - 1):
                    pass_used.add((path[i], path[i + 1]))
        used_edges = pass_used
        for d, path in candidate_paths.items():
            chosen[d] = path

    # Order the final assignment: try several orderings and commit the one
    # whose sequential discounted routing yields the lowest total cost
    # (cost of each unique edge is paid once, shared by all partitions/dsts).

    def savings(d):
        path = chosen[d]
        if path is None:
            return -float("inf")
        dcost = sum(edge_cost(used_edges, path[i], path[i + 1])
                    for i in range(len(path) - 1))
        return orig_cost[d] - dcost

    def simulate(order):
        """Route destinations sequentially with discount; return (routes, total)."""
        sim_used = set()
        routes = {}
        total = 0.0
        for d in order:
            path, _ = shortest_with_discount(sim_used, d)
            if path is None:
                path = chosen.get(d)
            if path is None:
                routes[d] = None
                continue
            routes[d] = path
            for i in range(0, len(path) - 1):
                s, t = path[i], path[i + 1]
                if (s, t) not in sim_used:
                    sim_used.add((s, t))
                    total += h[s][t]["cost"]
        return routes, total

    candidate_orders = [
        sorted(reachable, key=lambda d: orig_cost[d]),                      # ascending cost
        sorted(reachable, key=lambda d: orig_cost[d], reverse=True),        # descending cost
        sorted(reachable, key=savings, reverse=True),                       # largest savings first
    ]

    best_routes, best_total, best_order = None, float("inf"), None
    for order in candidate_orders:
        routes, total = simulate(order)
        if total < best_total:
            best_total, best_routes, best_order = total, routes, order

    bc_topology = BroadCastTopology(src, dsts, num_partitions)
    final_used = set()
    for d in best_order:
        path = best_routes.get(d)
        if path is None:
            unreachable.append(d)
            continue
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            final_used.add((s, t))
            for j in range(num_partitions):
                bc_topology.append_dst_partition_path(d, j, [s, t, G[s][t]])

    # unreachable destinations: empty partitions to avoid crashing
    for d in set(unreachable):
        for j in range(num_partitions):
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
        else:
            self.paths = {dst: {str(i): None for i in range(num_partitions)} for dst in dsts}

    def _add_full_path(self, dst: str, path: List[str], G, num_partitions: int):
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(num_partitions):
                self.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    def _replace_partition_path(self, dst: str, partition: int, path: List[str], G):
        partition = str(partition)
        edges = []
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            edges.append([s, t, G[s][t]])
        self.paths[dst][partition] = edges

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