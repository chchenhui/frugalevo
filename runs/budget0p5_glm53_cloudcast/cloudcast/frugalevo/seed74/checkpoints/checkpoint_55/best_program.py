# EVOLVE-BLOCK-START
import heapq
import networkx as nx
import json
import os
import pandas as pd
from typing import Dict, List


def search_algorithm(src, dsts, G, num_partitions):
    """
    Approach: global Steiner-tree multicast. Build a collapsed undirected view S
    of G where each pair's weight is the min cost over parallel directed edges
    (None-cost edges treated as free, matching make_nx_graph's unpriced pairs).
    Run steiner_tree(S, [src]+dsts) so expensive trunk edges near the source are
    shared across destinations instead of paid once per dst (per partition).
    Expand each destination's unique tree path from src into directed hops using
    the cheapest real directed edge of G for each consecutive pair, appending
    [s, t, edge_data] per partition exactly like the incumbent. Per-dst Dijkstra
    fallback preserves validity if the tree route is incomplete.
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    def dijkstra_route(dst):
        """Incumbent per-dst Dijkstra fallback; appends and returns True on success."""
        try:
            path = nx.dijkstra_path(h, src, dst, weight="cost")
        except (nx.NetworkXNoPath, nx.NetworkXException):
            return False
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])
        return True

    # Directed view with None costs treated as free (used for all routing
    # decisions and cost simulation).
    W = nx.DiGraph()
    for u, v, data in h.edges(data=True):
        c = data.get("cost")
        W.add_edge(u, v, cost=0.0 if c is None else c)

    inf = float("inf")

    def sp_cost(paths):
        """Simulated cost: each distinct directed edge in the union of paths
        is charged exactly once (preserves the sharing benefit)."""
        seen = set()
        total = 0.0
        for p in paths:
            for i in range(len(p) - 1):
                e = (p[i], p[i + 1])
                if e not in seen:
                    seen.add(e)
                    total += W[p[i]][p[i + 1]]["cost"]
        return total

    def splice(a, b):
        """Concatenate two node paths sharing their junction node."""
        return a + b[1:]

    reachable = [d for d in dsts if d in W and nx.has_path(W, src, d)]
    if not reachable:
        for dst in dsts:
            dijkstra_route(dst)
        return bc_topology

    # ---- Free-SCC contraction of the zero-cost subgraph ----
    # make_nx_graph leaves many pairs unpriced (cost=None). The parent priced
    # those edges at 0 but still routed hop-by-hop and let the seed search
    # waste effort inside free regions, sometimes attaching terminals through
    # priced detours when a free path existed. Here each strongly-connected
    # component of the zero-cost subgraph F is contracted into one supernode:
    # intra-SCC traversal is free AND unconditional (mutual reachability in F
    # guarantees a free directed path between any two nodes of an SCC), so all
    # search effort in the contracted digraph C concentrates on priced boundary
    # edges — the only edges that matter to cost. Terminals in the source's
    # free SCC become zero-cost by construction. Seeds (SPH multistart, hub,
    # SPT) are computed on C with the same start budget as the parent, then
    # expanded back to real node paths (cheapest priced real edge per
    # contracted hop + free BFS inside SCCs) and fed to the unchanged
    # sp_cost/local_search machinery.
    F = nx.DiGraph()
    for u, v, data in h.edges(data=True):
        if data.get("cost") is None:
            F.add_edge(u, v)
    scc_id = {}
    _next = 0
    for comp in nx.strongly_connected_components(F):
        for n in comp:
            scc_id[n] = _next
        _next += 1
    for n in h.nodes:
        if n not in scc_id:
            scc_id[n] = _next
            _next += 1

    # Contracted digraph C: one min-cost priced edge per cross-SCC pair, with
    # a map back to the cheapest real directed edge for expansion. Priced edges
    # whose endpoints share an SCC are dropped (a free path exists instead).
    C = nx.DiGraph()
    cheap_edge = {}
    for u, v, data in h.edges(data=True):
        c = data.get("cost")
        if c is None:
            continue
        a, b = scc_id[u], scc_id[v]
        if a == b:
            continue
        if not C.has_edge(a, b) or c < C[a][b]["cost"]:
            C.add_edge(a, b, cost=c)
            cheap_edge[(a, b)] = (u, v)

    src_scc = scc_id[src]
    term_sccs = sorted({scc_id[d] for d in reachable})

    def free_seg(a, b):
        """Free directed path (edge list implicit) from a to b inside the
        zero-cost subgraph F; [] if a == b, None if unreachable."""
        if a == b:
            return []
        try:
            return nx.shortest_path(F, a, b)[1:]
        except nx.NetworkXException:
            return None

    def expand_cpath(cnodes, dst):
        """Expand a contracted-node path (starting at src_scc) into a real
        node path ending at dst: cheapest priced real edge per contracted hop,
        free BFS paths inside SCCs between hops and for the final approach to
        dst. Returns the path or None if any hop cannot be realized."""
        real = [src]
        for i in range(1, len(cnodes)):
            a, b = cnodes[i - 1], cnodes[i]
            if a == b:
                continue
            if (a, b) not in cheap_edge:
                return None
            u, v = cheap_edge[(a, b)]
            if real[-1] != u:
                seg = free_seg(real[-1], u)
                if seg is None:
                    return None
                real.extend(seg)
            real.append(v)
        if real[-1] != dst:
            seg = free_seg(real[-1], dst)
            if seg is None:
                return None
            real.extend(seg)
        if not all(h.has_edge(real[i], real[i + 1]) for i in range(len(real) - 1)):
            return None
        return real

    def sph(graph, root, terminal_order):
        """Directed SPH on `graph`: grow an arborescence rooted at `root` by
        repeatedly attaching the cheapest unplaced terminal via a multi-source
        Dijkstra from all current tree nodes; return the directed edge set or
        None if a terminal is unreachable. Bounded by |T| Dijkstra sweeps."""
        T_nodes = {root}
        T_edges = set()
        order = [t for t in terminal_order if t != root]
        while order:
            dist = {n: 0.0 for n in T_nodes}
            prev = {}
            pq = [(0.0, n) for n in T_nodes]
            heapq.heapify(pq)
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, inf):
                    continue
                for v, ed in graph[u].items():
                    nd = d + ed["cost"]
                    if nd < dist.get(v, inf) - 1e-12:
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(pq, (nd, v))
            best_t, best_d = None, inf
            for t in order:
                d = dist.get(t, inf)
                if d < best_d:
                    best_t, best_d = t, d
            if best_t is None:
                return None
            v = best_t
            while v not in T_nodes:
                u = prev[v]
                T_edges.add((u, v))
                v = u
            T_nodes.add(best_t)
            order.remove(best_t)
        return T_edges

    def contracted_routes(edges):
        """Build per-dst real paths from a contracted edge set: BFS tree over
        contracted nodes from src_scc, then expand each dst's contracted path.
        Returns routes dict or None if any dst is missing/unrealizable."""
        succ = {}
        for u, v in edges:
            succ.setdefault(u, []).append(v)
        cpaths = {src_scc: [src_scc]}
        queue = [src_scc]
        while queue:
            a = queue.pop(0)
            for b in succ.get(a, []):
                if b not in cpaths:
                    cpaths[b] = cpaths[a] + [b]
                    queue.append(b)
        routes = {}
        for d in reachable:
            t = scc_id[d]
            if t not in cpaths:
                return None
            p = expand_cpath(cpaths[t], d)
            if p is None:
                return None
            routes[d] = p
        return routes

    # ---- Seed A: SPH Steiner multistart on the contracted graph ----
    # Same R=10 deterministic orderings as the parent (ascending, descending
    # src distance, 8 hash-shuffled), each strictly bounded by |T| sweeps.
    tree_routes = None
    tree_cost = inf
    if len(reachable) > 1:
        try:
            src_dist = {
                d: nx.dijkstra_path_length(C, src_scc, scc_id[d]) for d in reachable
            }
        except nx.NetworkXException:
            src_dist = {d: inf for d in reachable}
        orders = [
            sorted(reachable, key=str),
            sorted(reachable, key=lambda d: (src_dist.get(d, inf), str(d))),
        ]
        for r in range(8):
            salt = r + 1

            def hkey(d, _s=salt):
                hv = 2166136261
                for ch in str(d) + "|" + str(_s):
                    hv = ((hv ^ ord(ch)) * 16777619) & 0xFFFFFFFF
                return (hv % 1000003, str(d))

            orders.append(sorted(reachable, key=hkey))
        for order in orders:
            edges = sph(C, src_scc, [scc_id[d] for d in order])
            if edges is None:
                continue
            r_routes = contracted_routes(edges)
            if r_routes is None:
                continue
            c = sp_cost(list(r_routes.values()))
            if c < tree_cost:
                tree_cost = c
                tree_routes = r_routes

    # ---- Seed B: hub-replication trunk on the contracted graph ----
    # Supernodes are far fewer than real nodes, so exhaustive single-hub and
    # bounded two-hub chains remain cheap while capturing free trunk travel.
    c_nodes = [n for n in C.nodes if n != src_scc]
    cp_path = {}
    targets = set(c_nodes) | set(term_sccs)
    for a in [src_scc] + c_nodes:
        for b in targets:
            if a == b:
                continue
            try:
                if nx.has_path(C, a, b):
                    cp_path[(a, b)] = nx.dijkstra_path(C, a, b)
            except nx.NetworkXException:
                pass

    def centrality(a):
        tot = 0.0
        for t in term_sccs:
            if t == a:
                continue
            try:
                tot += nx.dijkstra_path_length(C, a, t)
            except nx.NetworkXException:
                return inf
        return tot

    hub_routes = None
    hub_cost = inf

    def eval_hubs(chain):
        """Evaluate trunk src_scc->chain->terminals on C using precomputed
        contracted pair paths; expand to real paths. Returns (routes, cost)
        or (None, inf)."""
        in_path = [src_scc]
        for c in chain:
            if (in_path[-1], c) not in cp_path:
                return None, inf
            in_path = in_path + cp_path[(in_path[-1], c)][1:]
        cpaths = {}
        for d in reachable:
            t = scc_id[d]
            if (in_path[-1], t) not in cp_path:
                return None, inf
            cpaths[d] = in_path + cp_path[(in_path[-1], t)][1:]
        routes = {}
        for d in reachable:
            p = expand_cpath(cpaths[d], d)
            if p is None:
                return None, inf
            routes[d] = p
        return routes, sp_cost(list(routes.values()))

    two_stage = c_nodes
    if len(c_nodes) > 25:
        two_stage = sorted(c_nodes, key=centrality)[:25]
    for u in c_nodes:
        out_paths, c = eval_hubs([u])
        if out_paths is not None and c < hub_cost:
            hub_cost = c
            hub_routes = out_paths
    for u in two_stage:
        for v in c_nodes:
            if u == v:
                continue
            out_paths, c = eval_hubs([u, v])
            if out_paths is not None and c < hub_cost:
                hub_cost = c
                hub_routes = out_paths

    # ---- Real-node pair paths for local_search splice moves ----
    # local_search's cross-path splice needs shortest real-node paths between
    # arbitrary junction/destination pairs; precompute once on W as before.
    nodes = [n for n in W.nodes if n != src]
    pair_path = {}
    for a in [src] + nodes:
        for b in nodes:
            if a == b:
                continue
            try:
                if nx.has_path(W, a, b):
                    pair_path[(a, b)] = nx.dijkstra_path(W, a, b)
            except nx.NetworkXException:
                pass

    # ---- Seed C: SPT seed (per-dst cheapest contracted paths, expanded) ----
    # Each dst takes its individually cheapest route through C (maximal
    # fan-out); hill-climbing from it can add sharing incrementally.
    spt_routes = None
    spt_cost = inf
    if all((src_scc, scc_id[d]) in cp_path for d in reachable):
        spt_routes = {}
        for d in reachable:
            p = expand_cpath(cp_path[(src_scc, scc_id[d])], d)
            if p is None:
                spt_routes = None
                break
            spt_routes[d] = p
        if spt_routes is not None:
            spt_cost = sp_cost(list(spt_routes.values()))

    # ---- Best-of-seeds local search ----
    # The hill-climb runs independently from each surviving seed (tree, hub,
    # SPT); the argmin of sp_cost is committed. Monotone: the tree seed starts
    # from the incumbent's committed structure and cost never increases, so
    # the result can only match or beat it. Two move types per sweep:
    #   (1) marginal-cost Dijkstra substitution (E-edges priced 0),
    #   (2) cross-path splicing: reuse a prefix of another dst's path plus a
    #       precomputed shortest suffix from the junction — this can create
    #       NEW shared edges that the marginal Dijkstra view cannot propose.
    # Bounded plateau moves (equal total cost) are accepted up to a small
    # budget per sweep to traverse plateaus; cost is never increased.
    def local_search(seed_routes):
        """Hill-climb with marginal-cost Dijkstra + cross-path splice moves;
        accepts strictly-improving and (budgeted) equal-cost moves only."""
        paths = {d: list(seed_routes[d]) for d in seed_routes}
        cur = sp_cost(list(paths.values()))
        for _sweep in range(3):
            E = set()
            for p in paths.values():
                for i in range(len(p) - 1):
                    E.add((p[i], p[i + 1]))
            M = nx.DiGraph()
            for u, v, data in W.edges(data=True):
                M.add_edge(u, v, cost=0.0 if (u, v) in E else data["cost"])
            improved = False
            plateau = 0
            for dst in list(paths):
                # Move 1: marginal-cost Dijkstra substitution.
                try:
                    cand = nx.dijkstra_path(M, src, dst, weight="cost")
                except nx.NetworkXException:
                    cand = None
                if cand is not None and cand != paths[dst]:
                    trial = dict(paths)
                    trial[dst] = cand
                    tc = sp_cost(list(trial.values()))
                    if tc < cur - 1e-12 or (tc <= cur + 1e-12 and plateau < 4):
                        if tc <= cur + 1e-12 and tc > cur - 1e-12:
                            plateau += 1
                        paths = trial
                        cur = tc
                        improved = True
                # Move 2: cross-path splice (share a prefix of another path).
                for o, op in list(paths.items()):
                    if o == dst:
                        continue
                    for k in range(1, len(op)):
                        j = op[k]
                        if (j, dst) not in pair_path:
                            continue
                        cand2 = op[: k + 1] + pair_path[(j, dst)][1:]
                        if cand2 == paths[dst]:
                            continue
                        if not all(
                            h.has_edge(cand2[i], cand2[i + 1])
                            for i in range(len(cand2) - 1)
                        ):
                            continue
                        trial = dict(paths)
                        trial[dst] = cand2
                        tc = sp_cost(list(trial.values()))
                        if tc < cur - 1e-12 or (tc <= cur + 1e-12 and plateau < 4):
                            if tc <= cur + 1e-12 and tc > cur - 1e-12:
                                plateau += 1
                            paths = trial
                            cur = tc
                            improved = True
            if not improved:
                break
        return paths, cur

    seeds = []
    if tree_routes is not None:
        seeds.append(tree_routes)
    if hub_routes is not None:
        seeds.append(hub_routes)
    if spt_routes is not None:
        seeds.append(spt_routes)

    if not seeds:
        for dst in dsts:
            dijkstra_route(dst)
        return bc_topology

    best_paths = None
    best_cost = inf
    for seed in seeds:
        p, c = local_search(seed)
        if c < best_cost:
            best_cost = c
            best_paths = p
    routes = best_paths

    for dst in dsts:
        if dst in routes:
            tpath = routes[dst]
            for i in range(len(tpath) - 1):
                s, t = tpath[i], tpath[i + 1]
                for j in range(bc_topology.num_partitions):
                    bc_topology.append_dst_partition_path(dst, j, [s, t, h[s][t]])
        else:
            dijkstra_route(dst)

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
