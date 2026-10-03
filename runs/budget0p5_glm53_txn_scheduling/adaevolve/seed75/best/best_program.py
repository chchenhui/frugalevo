import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Independent-component decomposition + exact DP on small components.

    1. Build the key->transaction incidence map from the workload.
    2. Transactions sharing no key with any other transaction ("free"
       txns) are removed from the combinatorial core; they add no
       conflict cost wherever placed.
    3. Find connected components of the conflict graph (networkx if
       available, union-find fallback). Components have zero
       cross-conflicts, so each is scheduled independently.
    4. Small components (<= 12 txns) are solved exactly via Held-Karp
       style DP over bitmask orderings using the true cost function.
    5. Large components get multi-restart exact-cost greedy + hybrid
       local search (2-opt, or-opt, segment reversal) restricted to
       the component.
    6. Concatenate component schedules, greedily reinsert free txns,
       then polish the full sequence with local search under the
       remaining time budget.
    """
    import time

    time_budget = 110.0
    start_time = time.time()
    n = workload.num_txns

    # ---------- key incidence ----------
    def txn_keys(t):
        for attr in ("txns", "transactions", "txn_list"):
            data = getattr(workload, attr, None)
            if data is None:
                continue
            item = data[t] if isinstance(data, (list, tuple)) else data.get("txn%d" % t)
            if item is None and isinstance(data, dict):
                for k in (t, str(t), "txn%d" % t):
                    if k in data:
                        item = data[k]
                        break
            if item is None:
                continue
            keys = set()
            if isinstance(item, str):
                toks = item.split()
            elif isinstance(item, (list, tuple)):
                toks = [str(x) for x in item]
            else:
                toks = []
            for tok in toks:
                parts = tok.split("-")
                if len(parts) == 2:
                    keys.add(parts[1])
            if keys or toks:
                return keys
        return set()

    keys_of = {}
    key_map = {}
    for t in range(n):
        ks = txn_keys(t)
        keys_of[t] = ks
        for k in ks:
            key_map.setdefault(k, set()).add(t)

    # free txns: no key shared with any other txn
    free_txns = [t for t in range(n)
                 if all(len(key_map[k]) == 1 for k in keys_of[t])]
    core = [t for t in range(n) if t not in set(free_txns)]

    # ---------- connected components of conflict graph ----------
    components = []
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(core)
        for k, ts in key_map.items():
            ts = sorted(x for x in ts if x in set(core))
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    G.add_edge(ts[i], ts[j])
        components = [sorted(c) for c in nx.connected_components(G)]
    except Exception:
        # union-find fallback
        parent = {t: t for t in core}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for k, ts in key_map.items():
            ts = [x for x in ts if x in set(core)]
            for i in range(1, len(ts)):
                a, b = find(ts[0]), find(ts[i])
                if a != b:
                    parent[a] = b
        groups = {}
        for t in core:
            groups.setdefault(find(t), []).append(t)
        components = [sorted(g) for g in groups.values()]

    components.sort(key=len)  # solve small (exact) first

    def exact_dp(comp, deadline):
        """Held-Karp style DP over bitmask orderings of a small
        component, using the true cost function on prefixes."""
        m = len(comp)
        if m > 12:
            return None
        full = (1 << m) - 1
        # dp[mask] = (best_prefix_cost, best_prefix_order_indices)
        dp = {0: (0, [])}
        masks = sorted(dp.keys())
        processed = {0}
        order_masks = sorted(range(1, full + 1), key=lambda x: bin(x).count("1"))
        for mask in order_masks:
            if time.time() - start_time > deadline:
                return None
            best = None
            for i in range(m):
                bit = 1 << i
                if not (mask & bit):
                    continue
                prev = mask ^ bit
                if prev not in dp:
                    continue
                pc, porder = dp[prev]
                seq = porder + [comp[i]]
                c = workload.get_opt_seq_cost(seq)
                if best is None or c < best[0]:
                    best = (c, seq)
            if best is not None:
                dp[mask] = best
        if full in dp:
            return dp[full]
        return None

    def greedy_comp(comp, start_txn, noise=0.0):
        seq = [start_txn]
        remaining = [x for x in comp if x != start_txn]
        while remaining:
            scored = []
            for t in remaining:
                c = workload.get_opt_seq_cost(seq + [t])
                scored.append((c, random.random() if noise else t, t))
            scored.sort()
            pick = scored[0][2]
            if noise and len(scored) > 1 and random.random() < noise:
                pick = scored[1][2]
            seq.append(pick)
            remaining.remove(pick)
        return workload.get_opt_seq_cost(seq), seq

    def local_search(seq, cost, txns=None):
        m = len(seq)
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            for i in range(m - 1):
                for j in range(i + 1, m):
                    if time.time() - start_time >= time_budget:
                        return cost, seq
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
            for i in range(m):
                if time.time() - start_time >= time_budget:
                    return cost, seq
                t = seq.pop(i)
                for j in range(m):
                    if j == i:
                        continue
                    seq.insert(j, t)
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                        break
                    seq.pop(j)
                else:
                    seq.insert(i, t)
            for i in range(m - 1):
                for j in range(i + 2, m):
                    if time.time() - start_time >= time_budget:
                        return cost, seq
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i:j + 1] = seq[i:j + 1][::-1]
        return cost, seq

    # ---------- solve each component ----------
    comp_schedules = []
    for comp in components:
        m = len(comp)
        if m <= 12:
            res = exact_dp(comp, start_time + time_budget * 0.55)
            if res is not None:
                comp_schedules.append(res[1])
                continue
        # heuristic: multi-start greedy + local search on component
        best_c, best_s = None, None
        starts = comp if m <= 8 else [
            comp[i] for i in range(0, m, max(1, m // 8))][:8]
        for s in starts:
            if time.time() - start_time >= time_budget * 0.6:
                break
            c, seq = greedy_comp(comp, s)
            if best_c is None or c < best_c:
                best_c, best_s = c, seq
        while time.time() - start_time < time_budget * 0.55:
            s = random.choice(comp)
            c, seq = greedy_comp(comp, s, noise=0.15)
            if c < best_c:
                best_c, best_s = c, seq
        best_c, best_s = local_search(best_s, best_c)
        comp_schedules.append(best_s)

    # ---------- assemble + reinsert free txns ----------
    seq = []
    for cs in comp_schedules:
        seq.extend(cs)
    for t in free_txns:
        best_c, best_j = None, 0
        for j in range(len(seq) + 1):
            seq.insert(j, t)
            c = workload.get_opt_seq_cost(seq)
            if best_c is None or c < best_c:
                best_c, best_j = c, j
            seq.pop(j)
        seq.insert(best_j, t)

    best_cost = workload.get_opt_seq_cost(seq)
    best_seq = seq

    # ---------- full-sequence polish + ILS with remaining time ----------
    best_cost, best_seq = local_search(best_seq, best_cost)

    pool = [(best_cost, best_seq)]
    while time.time() - start_time < time_budget:
        if random.random() < 0.5:
            base = best_seq
        else:
            base = random.choice(pool)[1]
        cand = base[:]
        if random.random() < 0.5:
            for _ in range(2):
                i, j = random.sample(range(n), 2)
                cand[i], cand[j] = cand[j], cand[i]
        else:
            i, j = sorted(random.sample(range(n), 2))
            cand[i:j + 1] = cand[i:j + 1][::-1]
        c, cand = local_search(cand, workload.get_opt_seq_cost(cand))
        if c < best_cost:
            best_cost, best_seq = c, cand
        if len(pool) < 5 or c < pool[-1][0]:
            pool.append((c, cand))
            pool.sort(key=lambda x: x[0])
            pool = pool[:5]

    assert len(set(best_seq)) == workload.num_txns
    return best_cost, best_seq

# EVOLVE-BLOCK-END

def get_random_costs():
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3]


if __name__ == "__main__":
    makespan, schedule = get_random_costs()
    print(f"Makespan: {makespan}")
