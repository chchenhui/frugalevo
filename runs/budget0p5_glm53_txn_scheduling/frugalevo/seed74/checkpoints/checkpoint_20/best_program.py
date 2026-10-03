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
    """Simulated annealing over permutation proposals with a fixed
    monotone cooling schedule.

    Mechanism: non-monotone Metropolis acceptance — c <= current always
    accepted; worsening moves accepted with probability exp(-delta/T) —
    letting the walk cross cost ridges that 2-opt descent and
    accept-only-if-better kicks cannot cross. The two prior tuned variants
    regressed because their effective temperature trajectories were
    time-coupled (horizon shrank as the slice elapsed, re-heat wandered):
    here the cooling horizon is computed ONCE at entry from a fixed
    iterations-per-second estimate, so T decays monotonically and
    predictably from T0 = 3% of the entry cost to 0.05% — cooler than
    before to limit wandering, warm enough early to cross ridges.
    Proposals are the empirically strongest pair: random position swap and
    random shift-reinsertion (both permutation-preserving, so validity is
    structural). Every candidate is judged by the memoized true-cost
    wrapper; best-so-far tracking guarantees the return is never worse
    than the entry incumbent; the deadline slice bounds compute.
    """
    cur_seq, cur_cost = seq[:], cost
    best_seq, best_cost = seq[:], cost
    T0 = max(1e-9, 0.03 * cost)
    T_end = max(1e-12, 0.0005 * cost)
    n = len(seq)
    # Fixed horizon (in iterations) estimated once from the slice length;
    # the loop itself remains strictly deadline-bounded.
    horizon = max(300.0, (deadline - time.time()) * 3000.0)
    cool = (T_end / T0) ** (1.0 / horizon)
    T = T0
    while time.time() < deadline:
        T = max(T_end, T * cool)
        cand = cur_seq[:]
        if rng.random() < 0.5:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            cand[i], cand[j] = cand[j], cand[i]
        else:
            i = rng.randrange(n)
            item = cand.pop(i)
            j = rng.randrange(n)
            cand.insert(j, item)
        c = _cost(workload, cand)
        d = c - cur_cost
        if d <= 0 or rng.random() < pow(2.718281828459045, -d / T):
            cur_seq, cur_cost = cand, c
            if c < best_cost:
                best_seq, best_cost = cand[:], c
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

    # Stage 1 (conflict-graph seeds; the structural graph build is nearly
    # free, so seeds get most of the stage budget for local search).
    graph_deadline = time.time() + (deadline - time.time()) * 0.25
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

    # Stage 2: beam-front construction on the remaining ~70% budget.
    # Keeps the w best partial prefixes alive at every depth; each extension
    # is costed by the simulator's true cost (memoized), so unlike the
    # irrevocable single-path greedy, multiple construction frontiers are
    # explored in parallel. The best complete sequence then goes through
    # the existing shift/2-opt local search.
    width = 4
    beam = []
    for s in range(n):
        beam.append(([s], _cost(workload, [s])))
        if len(beam) >= width:
            break
    for depth in range(1, n):
        if time.time() > deadline:
            break
        exts = []
        for pseq, pcost in beam:
            used = set(pseq)
            for t in range(n):
                if t in used:
                    continue
                cand = pseq + [t]
                exts.append((pcost + _cost(workload, cand), cand))
        exts.sort(key=lambda e: e[0])
        seen = set()
        beam = []
        for c, cand in exts:
            key = tuple(cand)
            if key in seen:
                continue
            seen.add(key)
            beam.append((cand, c))
            if len(beam) >= width:
                break
    if beam:
        beam.sort(key=lambda e: e[1])
        seq, cost = beam[0]
        seq, cost = _local_search(workload, seq, cost, deadline)
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
