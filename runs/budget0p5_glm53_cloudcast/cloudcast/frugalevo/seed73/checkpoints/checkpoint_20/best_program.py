# EVOLVE-BLOCK-START
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """Shared-multicast Steiner-tree broadcast.

    Builds an undirected symmetrized graph (edge weight = min of the two
    directed costs), computes a Steiner tree over terminals {src} + dsts,
    orients it from src via BFS, and assigns each dst its unique tree path
    so shared prefixes carry data once instead of per-dst duplicates.
    Falls back to plain Dijkstra for dsts not covered by the tree.
    """
    from networkx.algorithms.approximation.steinertree import steiner_tree

    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    # Symmetrize into an undirected graph with cheap edge weights.
    U = nx.Graph()
    for s, t, data in G.edges(data=True):
        if s == t or data.get("cost") is None:
            continue
        w = data["cost"]
        if U.has_edge(s, t):
            U[s][t]["cost"] = min(U[s][t]["cost"], w)
        else:
            U.add_edge(s, t, cost=w)

    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))

    def edge_data(s, t):
        if G.has_edge(s, t):
            return G[s][t]
        return G[t][s]

    def record(dst, path_nodes):
        for i in range(len(path_nodes) - 1):
            s, t = path_nodes[i], path_nodes[i + 1]
            for j in range(num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, edge_data(s, t)])

    terminals = {src} | {d for d in dsts if d != src and d in U}

    def dcost(s, t):
        d = edge_data(s, t)
        c = d.get("cost")
        return 0.0 if c is None else c

    def sol_cost(pmap):
        """Exact scored objective: sum of directed costs over the union of
        unique directed edges used by all per-dst paths (shared edges
        charged once, matching the multicast evaluator)."""
        used = set()
        for p in pmap.values():
            for i in range(len(p) - 1):
                used.add((p[i], p[i + 1]))
        return sum(dcost(s, t) for s, t in used)

    def build_solution(weight_mode):
        """Build a Steiner-tree broadcast plan under one symmetrization
        weight mode ('min' or 'sum' of the two directed edge costs), orient
        the tree from src, and polish it with bounded local search
        (re-attach, relay-prune, junction re-root) evaluated on the exact
        directed union-cost objective. Returns (paths_map, cost)."""
        Uw = nx.Graph()
        for s, t, data in G.edges(data=True):
            if s == t or data.get("cost") is None:
                continue
            w = data["cost"]
            if Uw.has_edge(s, t):
                Uw[s][t]["cost"] = (
                    min(Uw[s][t]["cost"], w) if weight_mode == "min"
                    else Uw[s][t]["cost"] + w
                )
            else:
                Uw.add_edge(s, t, cost=w)
        pmap = {}
        if len(terminals) > 1 and nx.is_connected(Uw.subgraph(Uw.nodes)):
            try:
                T = steiner_tree(Uw, list(terminals), weight="cost")
                if src in T:
                    parent = {src: None}
                    order = [src]
                    for node in order:
                        for nbr in T[node]:
                            if nbr not in parent:
                                parent[nbr] = node
                                order.append(nbr)
                    for dst in dsts:
                        if dst != src and dst in parent:
                            pn = []
                            node = dst
                            while node is not None:
                                pn.append(node)
                                node = parent[node]
                            pn.reverse()
                            pmap[dst] = pn
            except Exception:
                pmap = {}

        # Bounded local-search polish on the exact directed-cost objective.
        # Moves per sweep: (a) re-attach via directed shortest path;
        # (b) relay-prune: bypass each relay on the dst's path;
        # (c) junction re-root at <=5 cheapest one-hop neighbors of dst.
        # Accept only strictly improving moves; <= 3 sweeps.
        base = sol_cost(pmap) if pmap else float("inf")
        for _ in range(3):
            improved = False
            for dst in list(pmap):
                cur = pmap[dst]
                cands = []
                try:
                    cands.append(nx.dijkstra_path(h, src, dst, weight="cost"))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                for v in list(cur[1:-1]):
                    hh = h.copy()
                    hh.remove_node(v)
                    try:
                        cands.append(nx.dijkstra_path(hh, src, dst, weight="cost"))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                if dst in h:
                    nbr_costs = sorted(
                        ((h[dst][nbr].get("cost") or 0.0, nbr) for nbr in h[dst]),
                        key=lambda x: x[0],
                    )[:5]
                    for _, nbr in nbr_costs:
                        try:
                            pre = nx.dijkstra_path(h, src, nbr, weight="cost")
                            if pre[-1] == nbr and nbr != dst:
                                cands.append(pre + [dst])
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
                for p in cands:
                    if not p or len(p) < 2:
                        continue
                    trial = dict(pmap)
                    trial[dst] = p
                    c = sol_cost(trial)
                    if c < base - 1e-12:
                        pmap, base, improved = trial, c, True
            if not improved:
                break
        return pmap, base

    # Run the build+polish pipeline under both symmetrization metrics and
    # keep the cheaper plan (evaluated on the exact scored objective).
    best_map, best_cost = {}, float("inf")
    for mode in ("min", "sum"):
        try:
            pm, c = build_solution(mode)
            if c < best_cost:
                best_map, best_cost = pm, c
        except Exception:
            pass
    paths_map = best_map

    for dst, p in paths_map.items():
        record(dst, p)

    # Fallback: Dijkstra for any dst not covered by the polished tree.
    for dst in dsts:
        if dst == src or dst in paths_map:
            continue
        try:
            record(dst, nx.dijkstra_path(h, src, dst, weight="cost"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

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
