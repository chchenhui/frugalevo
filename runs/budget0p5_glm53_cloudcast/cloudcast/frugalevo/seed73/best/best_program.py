# EVOLVE-BLOCK-START
import networkx as nx
import json
import math
import os
import random
import pandas as pd
from itertools import islice
from typing import Dict, List


def exact_steiner_tree(Uw, terms):
    """Dreyfus-Wagner exact Steiner tree on undirected graph Uw (edge
    attr 'cost') over terminal list terms (terms[0] must be src).

    Subset DP: C[v][mask] = min cost of a tree connecting terminals in
    `mask` plus node v. Two-phase recurrence per mask: (1) merge submasks
    (3^k total work), (2) one min-plus edge relaxation pass using the
    *final* C[u][mask] of neighbors (single pass suffices since the edge
    term is min_u C[u][mask] + w(u,v)). Argmin decomposition choices are
    recorded in merge_split / epar and backtracked to recover the optimal
    tree edges. Returns an nx.Graph of the optimal tree, or None if some
    terminal is unreachable. O(3^k * n + 2^k * m).
    """
    nodes = list(Uw.nodes)
    k = len(terms)
    full = (1 << k) - 1
    INF = float("inf")
    C = {v: [INF] * (full + 1) for v in nodes}
    merge_split = {}
    epar = {}
    for i, t in enumerate(terms):
        if t in C:
            C[t][1 << i] = 0.0
    adj = {v: [(u, d["cost"]) for u, d in Uw[v].items()] for v in nodes}
    for mask in range(1, full + 1):
        for v in nodes:
            cv = C[v]
            sub = (mask - 1) & mask
            while sub:
                if cv[sub] + cv[mask ^ sub] < cv[mask]:
                    cv[mask] = cv[sub] + cv[mask ^ sub]
                    merge_split[(v, mask)] = sub
                    epar.pop((v, mask), None)
                sub = (sub - 1) & mask
            for u, w in adj[v]:
                if C[u][mask] + w < cv[mask]:
                    cv[mask] = C[u][mask] + w
                    epar[(v, mask)] = u
                    merge_split.pop((v, mask), None)
    best_v, best_c = None, INF
    for v in nodes:
        if C[v][full] < best_c:
            best_c, best_v = C[v][full], v
    if best_v is None or best_c == INF:
        return None
    tree_edges = set()

    def bt(v, mask):
        if (v, mask) in merge_split:
            s = merge_split[(v, mask)]
            bt(v, s)
            bt(v, mask ^ s)
        elif (v, mask) in epar:
            u = epar[(v, mask)]
            tree_edges.add((u, v))
            bt(u, mask)

    bt(best_v, full)
    T = nx.Graph()
    for u, v in tree_edges:
        T.add_edge(u, v, cost=Uw[u][v]["cost"])
    return T


def directed_steiner_tree(Gc, src, terms):
    """Exact directed Dreyfus-Wagner Steiner arborescence on digraph Gc
    (edge attr 'cost'), rooted at src, spanning terminal list `terms`.

    Subset DP over masks of `terms` (src is the ROOT, not a terminal):
    C[v][mask] = min cost of a directed tree rooted at v reaching all
    terminals in `mask`. Recurrence per mask: (1) merge submasks at v
    (3^k total work), (2) edge relaxation C[v][mask] = min over out-edges
    (v,u) of C[u][mask] + w(v,u) — every relaxation follows the edge in
    its true forward direction, so asymmetric per-direction prices are
    honored exactly. Argmin choices recorded in merge_split / epar and
    backtracked to recover the optimal directed edge set. Returns an
    nx.DiGraph of the optimal arborescence, or None if unreachable.
    O(3^k * n + 2^k * m); used for k <= 9 terminals.
    """
    nodes = list(Gc.nodes)
    k = len(terms)
    full = (1 << k) - 1
    INF = float("inf")
    C = {v: [INF] * (full + 1) for v in nodes}
    merge_split = {}
    epar = {}
    for i, t in enumerate(terms):
        if t in C:
            C[t][1 << i] = 0.0
    adj = {v: [(u, d["cost"]) for u, d in Gc[v].items()] for v in nodes}
    for mask in range(1, full + 1):
        for v in nodes:
            cv = C[v]
            sub = (mask - 1) & mask
            while sub:
                if cv[sub] + cv[mask ^ sub] < cv[mask]:
                    cv[mask] = cv[sub] + cv[mask ^ sub]
                    merge_split[(v, mask)] = sub
                    epar.pop((v, mask), None)
                sub = (sub - 1) & mask
            for u, w in adj[v]:
                if C[u][mask] + w < cv[mask]:
                    cv[mask] = C[u][mask] + w
                    epar[(v, mask)] = u
                    merge_split.pop((v, mask), None)
    if src not in C or C[src][full] == INF:
        return None
    tree_edges = set()

    def bt(v, mask):
        if (v, mask) in merge_split:
            s = merge_split[(v, mask)]
            bt(v, s)
            bt(v, mask ^ s)
        elif (v, mask) in epar:
            u = epar[(v, mask)]
            tree_edges.add((v, u))
            bt(u, mask)

    bt(src, full)
    T = nx.DiGraph()
    T.add_node(src)
    for a, b in tree_edges:
        T.add_edge(a, b, cost=Gc[a][b]["cost"])
    return T


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
        # Multi-seed construction + per-seed annealing on the exact directed
        # union-cost objective. Three structurally different seeds:
        #   A: exact/2-approx Steiner tree on the symmetrized undirected
        #      graph, oriented from src (maximal shared-prefix savings).
        #   B: directed Prim-Steiner growth on the original digraph — every
        #      tree edge is traversed in its true forward direction, so
        #      asymmetric per-direction prices are honored exactly (one
        #      cached reversed-graph Dijkstra per uncovered terminal; each
        #      round attaches the cheapest tree-node -> terminal path).
        #   C: union of independent per-dst Dijkstra paths (baseline).
        # Each seed is annealed SEPARATELY (distinct basins of attraction:
        # the reroute/splice moves cannot flip segment direction, so only a
        # directed seed can reach direction-aware optima) and the best
        # annealed result is returned, so output is never worse than the
        # incumbent single-seed pipeline. Budget: 3 seeds x 5000 iterations
        # of cheap sol_cost evaluations per weight mode, deterministic RNG.
        seeds = []
        if len(terminals) > 1:
            # --- Seed A: terminal-partitioned exact-Steiner forest ---
            # Partition dst terminals into small clusters (<=7). Each cluster
            # subtree is grown DIRECTED on the original digraph (Prim-Steiner
            # growth restricted to the cluster's terminals, every edge
            # traversed in its true forward direction), so per-cluster
            # subtrees honor asymmetric per-direction prices exactly — a
            # solution class neither the whole-terminal undirected-exact tree
            # (Seed A of the incumbent, exact only on the symmetrized graph
            # and oriented afterwards) nor the global directed greedy tree
            # (Seed B) can express. One additional clustering variant uses
            # the exact undirected DW per cluster for diversity. Clusters
            # come from (1) greedy farthest-point on all-pairs shortest-path
            # costs and (2) provider-prefix grouping. All seeds feed the
            # existing annealing polish and best-of selection, so the
            # pipeline remains a monotone refinement. Budget: clusterings x
            # per-cluster directed growth with one cached reversed-graph
            # Dijkstra per terminal — well under a second.
            try:
                if nx.is_connected(Uw.subgraph(Uw.nodes)):
                    others = [t for t in terminals if t != src]

                    # All-pairs shortest-path costs among terminals on Uw.
                    dist = {}
                    for a in [src] + others:
                        try:
                            da = nx.single_source_dijkstra_path_length(
                                Uw, a, weight="cost")
                        except nx.NetworkXNoPath:
                            da = {}
                        for b in [src] + others:
                            dist[(a, b)] = 0.0 if a == b else da.get(b, float("inf"))

                    # Directed view of G with finite costs only.
                    Gc = nx.DiGraph(
                        (s, t, {"cost": d["cost"]}) for s, t, d in G.edges(data=True)
                        if s != t and d.get("cost") is not None
                    )
                    Grev = Gc.reverse(copy=False)

                    def grow_directed_cluster(cl, parent):
                        """Directed Prim-Steiner growth for one cluster:
                        attach each cluster terminal to the growing tree via
                        its cheapest directed path (cached reversed-graph
                        Dijkstra per terminal). Parents recorded into the
                        shared per-seed parent map."""
                        tree_nodes = {src}
                        unc = [t for t in cl if t != src and t in Gc]
                        term_dist, term_paths = {}, {}
                        while unc:
                            best = None
                            for t in unc:
                                if t not in term_dist:
                                    try:
                                        term_dist[t], term_paths[t] = nx.single_source_dijkstra(
                                            Grev, t, weight="cost")
                                    except nx.NetworkXNoPath:
                                        term_dist[t] = {}
                                for v in tree_nodes:
                                    dv = term_dist[t].get(v)
                                    if dv is not None and (best is None or dv < best[0]):
                                        best = (dv, v, t)
                            if best is None:
                                break
                            _, v, t = best
                            seg = list(reversed(term_paths[t][v]))  # v -> t in G
                            for i in range(len(seg) - 1):
                                parent[seg[i + 1]] = seg[i]
                            tree_nodes.update(seg)
                            unc.remove(t)

                    def forest_seed(clusters, directed=True):
                        """Union of per-cluster subtrees oriented from src.
                        directed=True: per-cluster directed Prim-Steiner
                        growth; directed=False: exact undirected DW per
                        cluster. Returns a {dst: path} map or None."""
                        parent = {src: None}
                        for cl in clusters:
                            tl = [src] + [t for t in cl if t != src]
                            if len(tl) < 2:
                                continue
                            if directed:
                                grow_directed_cluster(cl, parent)
                            else:
                                T = exact_steiner_tree(Uw, tl)
                                if T is None or src not in T:
                                    return None
                                order = [src]
                                seen = {src}
                                while order:
                                    node = order.pop(0)
                                    for nbr in T[node]:
                                        if nbr not in seen:
                                            seen.add(nbr)
                                            if nbr not in parent:
                                                parent[nbr] = node
                                            order.append(nbr)
                        pmap_f = {}
                        for dst in dsts:
                            if dst != src and dst in parent:
                                pn = []
                                node = dst
                                while node is not None:
                                    pn.append(node)
                                    node = parent[node]
                                pn.reverse()
                                pmap_f[dst] = pn
                        return pmap_f or None

                    # Clustering 1: greedy farthest-point, clusters <= 7.
                    def farthest_clusters(cap=7):
                        pts = others[:]
                        clusters = []
                        while pts:
                            cl = [pts[0]]
                            pts = pts[1:]
                            while len(cl) < cap and pts:
                                best_p, best_d = None, -1.0
                                for p in pts:
                                    dmin = min(dist[(p, c)] for c in cl)
                                    if dmin > best_d:
                                        best_d, best_p = dmin, p
                                cl.append(best_p)
                                pts.remove(best_p)
                            clusters.append(cl)
                        return clusters

                    # Clustering 2: provider-prefix grouping (split on '-').
                    def prefix_clusters(cap=7):
                        groups = {}
                        for t in others:
                            key = str(t).split("-")[0]
                            groups.setdefault(key, []).append(t)
                        clusters = []
                        for key in sorted(groups):
                            g = groups[key]
                            for i in range(0, len(g), cap):
                                clusters.append(g[i:i + cap])
                        return clusters

                    for clusters in (farthest_clusters(), prefix_clusters()):
                        fs = forest_seed(clusters, directed=True)
                        if fs:
                            seeds.append(fs)
                    # Exact-undirected forest variant for diversity.
                    fs_u = forest_seed(farthest_clusters(), directed=False)
                    if fs_u:
                        seeds.append(fs_u)
            except Exception:
                pass

            # --- Seed B: directed Prim-Steiner growth on G ---
            try:
                Gc = nx.DiGraph(
                    (s, t, {"cost": d["cost"]}) for s, t, d in G.edges(data=True)
                    if s != t and d.get("cost") is not None
                )
                unc = [t for t in terminals if t != src and t in Gc]
                if unc:
                    Grev = Gc.reverse(copy=False)
                    tree_nodes = {src}
                    parent = {}
                    term_dist = {}
                    term_paths = {}
                    while unc:
                        best = None
                        for t in unc:
                            if t not in term_dist:
                                try:
                                    term_dist[t], term_paths[t] = nx.single_source_dijkstra(
                                        Grev, t, weight="cost")
                                except nx.NetworkXNoPath:
                                    term_dist[t] = {}
                            for v in tree_nodes:
                                dv = term_dist[t].get(v)
                                if dv is not None and (best is None or dv < best[0]):
                                    best = (dv, v, t)
                        if best is None:
                            break
                        _, v, t = best
                        seg = list(reversed(term_paths[t][v]))  # v -> t in G
                        for i in range(len(seg) - 1):
                            parent[seg[i + 1]] = seg[i]
                        tree_nodes.update(seg)
                        unc.remove(t)
                    bmap = {}
                    for dst in dsts:
                        if dst != src and dst in parent:
                            pn = []
                            node = dst
                            while node is not None:
                                pn.append(node)
                                node = parent.get(node)
                            if pn[-1] == src:
                                pn.reverse()
                                bmap[dst] = pn
                    if bmap:
                        seeds.append(bmap)
                    # --- Seed B2: exact directed Dreyfus-Wagner ---
                    # DP run directly on the digraph Gc rooted at src:
                    # C[v][mask] = min cost of a directed tree rooted at v
                    # reaching terminals in mask. Every DP edge relaxation
                    # follows the edge in its true forward direction, so
                    # asymmetric per-direction prices are honored exactly —
                    # the exact optimum of the scored union-of-directed-
                    # edges objective, a class neither the symmetrized
                    # undirected-exact tree (oriented afterwards) nor the
                    # greedy directed growth can express. Active for
                    # k <= 9 dsts (2^k * m states); feeds the existing
                    # annealing polish and best-of-seed selection.
                    dterms = [d for d in dsts if d != src and d in Gc]
                    if 1 <= len(dterms) <= 9:
                        Tdir = directed_steiner_tree(Gc, src, dterms)
                        if Tdir is not None:
                            dmap = {}
                            for d in dterms:
                                try:
                                    p = nx.shortest_path(Tdir, src, d)
                                    if p[-1] == d:
                                        dmap[d] = p
                                except (nx.NetworkXNoPath, nx.NodeNotFound):
                                    pass
                            if dmap:
                                seeds.append(dmap)
            except Exception:
                pass

            # --- Seed C: independent per-dst Dijkstra union ---
            try:
                cmap = {}
                for dst in dsts:
                    if dst == src or dst not in U:
                        continue
                    try:
                        cmap[dst] = nx.dijkstra_path(h, src, dst, weight="cost")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                if cmap:
                    seeds.append(cmap)
            except Exception:
                pass

            # --- Seed D: subset-marginal DP with per-state zero-priced
            # Dijkstra + incumbent ordering router as retained seed ---
            # Incumbent: route dsts one at a time; edges used by earlier
            # dsts cost 0. Kept as a seed (monotone baseline).
            # New mechanism (subset-marginal DP): S[mask] = min union cost
            # of serving exactly the dsts in `mask`,
            #   S[mask] = min over d in mask of
            #     S[mask\{d}] + marginal(d | solution(mask\{d})),
            # where marginal(d) is computed with a REAL zero-priced Dijkstra
            # over the used-edge set of the optimal mask\{d} solution — so
            # each DP state may pick a physically different path, a solution
            # class the fixed-ordering router (fixed cached paths) cannot
            # express. Budget: k <= 8 -> <= 2048 small Dijkstras; above that
            # only the incumbent router runs. A final cleanup sweep re-routes
            # each dst under zero-priced reuse, accepting strict gains only.
            try:
                routable = [d for d in dsts if d != src and d in U]
                if routable:
                    def seq_route(order):
                        used = set()
                        pmap_o = {}
                        for d in order:
                            def zw(u, v, ed):
                                return 0.0 if (u, v) in used else dcost(u, v)
                            try:
                                p = nx.dijkstra_path(h, src, d, weight=zw)
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                continue
                            pmap_o[d] = p
                            for i in range(len(p) - 1):
                                used.add((p[i], p[i + 1]))
                        return pmap_o

                    def cleanup(pm_in, c_in):
                        """Single re-route sweep under zero-priced reuse;
                        accepts strict improvements only."""
                        order = sorted(pm_in)
                        used = set()
                        for d in order:
                            for i in range(len(pm_in[d]) - 1):
                                used.add((pm_in[d][i], pm_in[d][i + 1]))
                        for d in order:
                            old = pm_in[d]
                            def zw2(u, v, ed):
                                return 0.0 if (u, v) in used else dcost(u, v)
                            try:
                                np_ = nx.dijkstra_path(h, src, d, weight=zw2)
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                continue
                            if list(np_) == list(old):
                                continue
                            trial = dict(pm_in)
                            trial[d] = np_
                            tc = sol_cost(trial)
                            if tc < c_in:
                                pm_in, c_in = trial, tc
                                used = set()
                                for dd in order:
                                    if dd in pm_in:
                                        for i in range(len(pm_in[dd]) - 1):
                                            used.add((pm_in[dd][i], pm_in[dd][i + 1]))
                        return pm_in, c_in

                    base_order = sorted(routable)
                    orderings = [
                        base_order,
                        list(reversed(base_order)),
                        sorted(routable, key=lambda d: -dcost(src, d) if G.has_edge(src, d) else 0.0),
                    ]
                    rng_o = random.Random(777)
                    shuf = base_order[:]
                    rng_o.shuffle(shuf)
                    orderings.append(shuf)

                    best_seq, best_seq_cost = None, float("inf")
                    for order in orderings:
                        pm_o = seq_route(order)
                        c_o = sol_cost(pm_o) if pm_o else float("inf")
                        pm_o, c_o = cleanup(pm_o, c_o)
                        if c_o < best_seq_cost:
                            best_seq, best_seq_cost = pm_o, c_o
                    if best_seq:
                        seeds.append(best_seq)

                    # --- Subset DP with per-state zero-priced Dijkstra ---
                    k = len(routable)
                    if 0 < k <= 8:
                        full_m = (1 << k) - 1
                        dp_cost = {0: 0.0}
                        dp_pmap = {0: {}}
                        dp_used = {0: set()}
                        for mask in range(1, full_m + 1):
                            best = (float("inf"), None)
                            for i in range(k):
                                if not (mask >> i) & 1:
                                    continue
                                prev = mask ^ (1 << i)
                                if prev not in dp_cost:
                                    continue
                                d = routable[i]
                                used = dp_used[prev]
                                if not used:
                                    mc = 0.0
                                    p = None
                                    try:
                                        p = nx.dijkstra_path(h, src, d, weight="cost")
                                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                                        p = None
                                    if p is None:
                                        continue
                                else:
                                    def zwm(u, v, ed):
                                        return 0.0 if (u, v) in used else dcost(u, v)
                                    try:
                                        p = nx.dijkstra_path(h, src, d, weight=zwm)
                                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                                        continue
                                    mc = sum(dcost(p[j], p[j + 1])
                                             for j in range(len(p) - 1)
                                             if (p[j], p[j + 1]) not in used)
                                c = dp_cost[prev] + mc
                                if c < best[0]:
                                    best = (c, (d, prev, p))
                            if best[1] is None:
                                continue
                            c, (d, prev, p) = best
                            dp_cost[mask] = c
                            pm_n = dict(dp_pmap[prev])
                            pm_n[d] = list(p)
                            dp_pmap[mask] = pm_n
                            us_n = set(dp_used[prev])
                            for j in range(len(p) - 1):
                                us_n.add((p[j], p[j + 1]))
                            dp_used[mask] = us_n
                        if full_m in dp_cost:
                            dp_map = dict(dp_pmap[full_m])
                            dp_map, c_dp = cleanup(dp_map, sol_cost(dp_map))
                            if dp_map and c_dp < best_seq_cost:
                                seeds.append(dp_map)
                            elif dp_map:
                                seeds.append(dp_map)
            except Exception:
                pass

        # Per-seed simulated-annealing polish on the exact directed
        # union-cost objective (uphill moves allowed under a geometric
        # temperature schedule). Moves: (a) reroute from the <=5 cheapest
        # simple paths; (b) splice at a shared node with another dst's
        # current path. Best-ever map per seed is kept; the overall best
        # across seeds is returned.
        pmap, base = {}, float("inf")
        for seed in seeds:
            rng = random.Random(12345)
            cand_paths = {}
            for d in seed:
                try:
                    cand_paths[d] = list(
                        islice(nx.shortest_simple_paths(h, src, d, weight="cost"), 5)
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    cand_paths[d] = [seed[d]]
            cur, cur_cost = dict(seed), sol_cost(seed)
            best_map, best_cost = dict(seed), cur_cost
            dst_list = list(cur)
            T0 = 0.5 * cur_cost / max(1, len(dst_list))
            T_end = 0.01
            iters = 5000
            for it in range(iters):
                T = T0 * (T_end / T0) ** (it / max(1, iters - 1)) if T0 > 0 else T_end
                dst = rng.choice(dst_list)
                cp = cur[dst]
                cands = [rng.choice(cand_paths[dst])]
                other = rng.choice(dst_list)
                op = cur[other]
                cp_set = set(cp)
                shared = [v for v in op if v in cp_set]
                if shared:
                    v = rng.choice(shared)
                    spliced = op[: op.index(v) + 1] + cp[cp.index(v) + 1:]
                    cands.append(spliced)
                p = rng.choice(cands)
                if not p or len(p) < 2 or p[0] != src or p[-1] != dst:
                    continue
                trial = dict(cur)
                trial[dst] = p
                c = sol_cost(trial)
                delta = c - cur_cost
                if delta <= 0 or (cur_cost > 0 and rng.random() < math.exp(-delta / max(T, 1e-12))):
                    cur, cur_cost = trial, c
                    if c < best_cost:
                        best_map, best_cost = dict(trial), c
            if best_cost < base:
                pmap, base = best_map, best_cost
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
