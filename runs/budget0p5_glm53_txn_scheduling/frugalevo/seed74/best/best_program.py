import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

# Shared deadline across the three-workload evaluation (hard limit 360s,
# leave margin for startup / caller-side cost recomputation).
_DEADLINE = time.time() + 300.0


_cost_cache = {}


def _cost(workload, seq):
    """Memoized wrapper around the simulator's true cost; greedy prefixes,
    pair costs, and local-search candidates repeat heavily, so caching
    buys extra search iterations within the same deadline. The cache key
    includes workload identity to avoid cross-workload collisions."""
    key = (id(workload), tuple(seq))
    c = _cost_cache.get(key)
    if c is None:
        c = workload.get_opt_seq_cost(list(seq))
        _cost_cache[key] = c
    return c


def _op_keys(workload, t):
    """Extract (read-key set, write-key set) for transaction t.

    Robust against op representations: ops may be strings like 'w-17' /
    'r-5' or tuples/lists whose first element is the type string or tag.
    Purely structural access to workload.txns; the simulator cost remains
    the sole judge of schedule quality.
    """
    rkeys, wkeys = set(), set()
    for op in workload.txns[t]:
        typ = key = None
        if isinstance(op, str):
            if len(op) >= 2 and op[0] in ("r", "w") and op[1] in ("-", " "):
                typ, key = op[0], op[2:]
        else:
            try:
                e0 = op[0]
                if isinstance(e0, str) and len(e0) >= 2 and e0[0] in ("r", "w") and e0[1] in ("-", " "):
                    typ, key = e0[0], e0[2:]
                elif e0 in ("r", "w", "read", "write"):
                    typ = "r" if e0 in ("r", "read") else "w"
                    key = str(op[1])
            except Exception:
                typ = key = None
        if typ == "w":
            wkeys.add(str(key))
        elif typ == "r":
            rkeys.add(str(key))
    return rkeys, wkeys


def _build_conflict_graph(workload, deadline):
    """Structural directed conflict graph from key overlap (no simulator
    evaluations, O(n^2) set intersections — essentially free).

    Each shared written key contributes weight 1.0 (write-write conflicts
    fully serialize); each read/write crossing contributes 0.5. A node's
    'pressure' is the sum of incident weights (a hot-key proxy used to
    rank seeds). Edge direction is a heuristic precedence: the side whose
    writes are consumed more by the other goes first.
    """
    n = workload.num_txns
    weight = [[0.0] * n for _ in range(n)]
    pressure = [0.0] * n
    rk, wk = {}, {}
    for t in range(n):
        rk[t], wk[t] = _op_keys(workload, t)
    for a in range(n):
        for b in range(a + 1, n):
            ww = len(wk[a] & wk[b])
            aw = len(rk[a] & wk[b])
            bw = len(wk[a] & rk[b])
            if ww == 0 and aw == 0 and bw == 0:
                continue
            w = ww + 0.5 * (aw + bw)
            pressure[a] += w
            pressure[b] += w
            if aw >= bw:
                weight[a][b] = w
            else:
                weight[b][a] = w
    return weight, pressure


def _graph_seed(n, weight, pressure, rng):
    """Kahn-style topological seed: repeatedly pick, among nodes with no
    unplaced predecessors, the one with the largest noisy conflict
    pressure. Multiplicative lognormal noise makes the seeds genuinely
    distinct orderings rather than near-copies. Leftovers from cycles are
    appended by descending pressure."""
    indeg = [0] * n
    succs = [[] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if weight[a][b] > 0:
                succs[a].append(b)
                indeg[b] += 1
    placed = [False] * n
    seq = []
    avail = [v for v in range(n) if indeg[v] == 0]
    noise = [rng.lognormvariate(0.0, 0.6) for _ in range(n)]
    while avail:
        best = max(avail, key=lambda v: pressure[v] * noise[v])
        avail.remove(best)
        placed[best] = True
        seq.append(best)
        for s in succs[best]:
            indeg[s] -= 1
            if indeg[s] == 0 and not placed[s]:
                avail.append(s)
    rest = [v for v in range(n) if not placed[v]]
    rest.sort(key=lambda v: -pressure[v])
    seq.extend(rest)
    return seq


def _local_search(workload, seq, cost, deadline):
    """Best-improvement local search over shift + 2-opt swaps.

    Neighborhood: (a) shift moves — remove one transaction, reinsert at
    the best of all positions; (b) full 2-opt — swap any two positions
    (strictly contains the adjacent-swap move, so no separate sweep).
    Every candidate is evaluated through the memoized true-cost wrapper,
    which deduplicates permutations across passes, restarts, and stages.
    All moves preserve permutation validity; the simulator remains the
    sole judge. Deadline-checked so the stronger neighborhood cannot
    blow the shared budget.
    """
    n = len(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shift pass.
        for i in range(n):
            if time.time() > deadline:
                return seq, cost
            item = seq[i]
            rest = seq[:i] + seq[i + 1:]
            best_j, best_c = i, cost
            for j in range(n):
                if j == i:
                    continue
                cand = rest[:j] + [item] + rest[j:]
                c = _cost(workload, cand)
                if c < best_c:
                    best_c, best_j = c, j
            if best_j != i:
                seq = rest[:best_j] + [item] + rest[best_j:]
                cost = best_c
                improved = True
        # Or-opt pass: relocate contiguous segments of length 2-3.
        for L in (2, 3):
            for i in range(n - L + 1):
                if time.time() > deadline:
                    return seq, cost
                seg = seq[i:i + L]
                rest = seq[:i] + seq[i + L:]
                best_j, best_c = i, cost
                for j in range(len(rest) + 1):
                    cand = rest[:j] + seg + rest[j:]
                    if cand == seq:
                        continue
                    c = _cost(workload, cand)
                    if c < best_c:
                        best_c, best_j = c, j
                if best_j != i:
                    seq = rest[:best_j] + seg + rest[best_j:]
                    cost = best_c
                    improved = True
        # 2-opt pass (arbitrary pair swaps).
        for i in range(n - 1):
            if time.time() > deadline:
                return seq, cost
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return seq, cost
                cand = seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = _cost(workload, cand)
                if c < cost:
                    seq, cost = cand, c
                    improved = True
    return seq, cost


def _anneal(workload, seq, cost, deadline, rng):
    """Annealing over the true cost with stagnation-triggered warm restarts
    and a three-move proposal set.

    Corrections over the two prior failed variants: the walk starts from
    the ENTRY INCUMBENT (not a structural seed — that wasted the whole
    slice wandering), and instead of a high-temperature reheat (which
    destroyed the incumbent and regressed the score), a STAGNATION
    TRIGGER restarts the walk from the best-so-far state at a LOW
    temperature (1% of best cost), so each restart is a fresh cool
    perturbation around the incumbent rather than destructive wandering.

    Proposals: random position swap, random shift-reinsertion, and
    SEGMENT INVERSION (reverse a contiguous run of length 3-8) — a
    geometrically distinct escape direction that flips the relative order
    of a whole conflicting run in one move, which swap/shift chains
    cannot reach without individually rejected intermediates. All
    proposals preserve permutation validity; Metropolis acceptance with a
    monotone geometric cooling; best-so-far tracking guarantees the
    return is never worse than the entry incumbent; the deadline bounds
    compute.
    """
    n = len(seq)
    cur_seq, cur_cost = seq[:], cost
    best_seq, best_cost = seq[:], cost
    T_end_scale = 0.0005
    restart_T0_scale = 0.01

    def _cool(T0, remain):
        T_end = max(1e-12, T_end_scale * best_cost)
        horizon = max(300.0, remain * 3000.0)
        return T_end, (T_end / T0) ** (1.0 / horizon)

    T0 = max(1e-9, 0.03 * cur_cost)
    T_end, cool = _cool(T0, max(1.0, deadline - time.time()))
    T = T0
    stall = 0
    stall_limit = 25 * n + 500
    while time.time() < deadline:
        T = max(T_end, T * cool)
        cand = cur_seq[:]
        u = rng.random()
        if u < 0.35:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            cand[i], cand[j] = cand[j], cand[i]
        elif u < 0.7:
            i = rng.randrange(n)
            item = cand.pop(i)
            j = rng.randrange(n)
            cand.insert(j, item)
        else:
            L = rng.randint(3, 8)
            i = rng.randrange(0, max(1, n - L + 1))
            j = min(n, i + L)
            cand[i:j] = cand[i:j][::-1]
        c = _cost(workload, cand)
        d = c - cur_cost
        if d <= 0 or rng.random() < pow(2.718281828459045, -d / T):
            cur_seq, cur_cost = cand, c
            if c < best_cost:
                best_seq, best_cost = cand[:], c
                stall = 0
            else:
                stall += 1
        else:
            stall += 1
        # Warm restart: cool perturbation from the best state when stuck.
        if stall >= stall_limit and time.time() < deadline:
            cur_seq, cur_cost = best_seq[:], best_cost
            T0 = max(1e-9, restart_T0_scale * best_cost)
            T_end, cool = _cool(T0, max(1.0, deadline - time.time()))
            T = T0
            stall = 0
    return best_seq, best_cost


def get_best_schedule(workload, num_seqs):
    """
    Multi-start full greedy + swap local search under a shared time budget.

    Runs full greedy (evaluating every candidate with the simulator's true
    cost) from several random starting transactions, then applies pairwise
    swap hill climbing. Keeps and returns the best valid permutation found
    before the deadline. Time is split conservatively across workloads.
    """
    n = workload.num_txns
    # Reserve time for up to 3 workloads; give each a slice of the budget.
    remaining = _DEADLINE - time.time()
    deadline = time.time() + max(30.0, remaining / 3.0)

    rng = random.Random(12345)
    best_seq = None
    best_cost = None

    # Stage 1 (conflict-graph seeds): shrunk to 10% of the slice. The
    # structural graph build is nearly free and the seeds mainly supply a
    # starting point; the marginal evaluation is worth far more in the
    # later annealing/polish phase, which demonstrably produced the
    # incumbent's best cost.
    graph_deadline = time.time() + (deadline - time.time()) * 0.10
    seed_stage_end = graph_deadline
    try:
        weight, pressure = _build_conflict_graph(workload, graph_deadline)
    except Exception:
        weight = pressure = None
    if weight is not None:
        for k in range(5):
            now = time.time()
            if now > seed_stage_end:
                break
            seq = _graph_seed(n, weight, pressure, rng)
            cost = _cost(workload, seq)
            if time.time() < seed_stage_end:
                seq, cost = _local_search(workload, seq, cost, seed_stage_end)
            if best_cost is None or cost < best_cost:
                best_seq, best_cost = seq, cost

    # Stage 2: regret-insertion construction, capped at ~30% of the slice
    # (was ~70%). Construction only needs to produce a good-enough seed;
    # the remaining ~60% is consumed by the Stage-3 annealing + polish
    # below, roughly doubling SA iterations around the incumbent at near-
    # zero marginal cost thanks to the memoized true-cost cache.
    stage2_deadline = time.time() + (deadline - time.time()) * 0.30
    # Unlike the previous append-only beam (which could only extend prefixes
    # at the end), each step inserts a not-yet-placed transaction at ANY
    # position of the partial schedule. At step k we evaluate the top-6
    # candidates (by conflict pressure) at every insertion position via the
    # memoized true-cost wrapper, then pick the candidate with the largest
    # regret (best minus second-best insertion cost) so that the most
    # constrained transactions are placed first, each at its cheapest slot.
    # The complete permutation is then polished by the existing local search.
    try:
        _, pressure2 = _build_conflict_graph(workload, deadline)
    except Exception:
        pressure2 = None
    if pressure2 is None:
        pressure2 = [0.0] * n
    order_cands = sorted(range(n), key=lambda v: -pressure2[v])
    cand_pool = order_cands[:6]
    pool = order_cands[:]
    rng.shuffle(pool)
    # Seed partial schedule with the highest-pressure transaction.
    partial = [order_cands[0]]
    placed = {order_cands[0]}
    cur_cost = _cost(workload, partial)
    while len(partial) < n and time.time() < stage2_deadline:
        cands = []
        ci = 0
        for t in order_cands:
            if t not in placed:
                cands.append(t)
                ci += 1
                if ci >= 6:
                    break
        if not cands:
            break
        best_t, best_pos, best_c, second_c = None, None, None, None
        for t in cands:
            costs = []
            for pos in range(len(partial) + 1):
                if time.time() > deadline:
                    break
                cand = partial[:pos] + [t] + partial[pos:]
                costs.append((_cost(workload, cand), pos))
            if not costs:
                costs = [(_cost(workload, partial + [t]), len(partial))]
            costs.sort(key=lambda e: e[0])
            cmin, pmin = costs[0]
            csecond = costs[1][0] if len(costs) > 1 else cmin + 1.0
            regret = csecond - cmin
            if best_c is None or regret > (second_c - best_c) + 1e-12 or (
                abs(regret - (second_c - best_c)) <= 1e-12 and cmin < best_c):
                best_t, best_pos, best_c, second_c = t, pmin, cmin, csecond
        partial = partial[:best_pos] + [best_t] + partial[best_pos:]
        placed.add(best_t)
        cur_cost = best_c
    # Fill anything missed due to deadline expiry in a single append sweep.
    for t in range(n):
        if t not in placed:
            partial.append(t)
    cost = _cost(workload, partial)
    seq, cost = _local_search(workload, partial, cost, deadline)
    if best_cost is None or cost < best_cost:
        best_seq, best_cost = seq, cost

    # Guarantee a feasible complete schedule even if the budget expired early.
    if best_seq is None:
        best_seq = list(range(n))
        best_cost = workload.get_opt_seq_cost(best_seq)

    # Stage 3: simulated annealing from the incumbent, consuming any
    # leftover slice. Non-monotone acceptance (Metropolis criterion with
    # geometric cooling) lets the walk escape the incumbent's basin;
    # best-so-far tracking keeps the return never worse than the entry
    # incumbent. A brief local-search polish runs on the annealed best.
    best_seq, best_cost = _anneal(workload, best_seq, best_cost, deadline, rng)
    if time.time() < deadline:
        best_seq, best_cost = _local_search(workload, best_seq, best_cost, deadline)

    assert len(set(best_seq)) == n
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
