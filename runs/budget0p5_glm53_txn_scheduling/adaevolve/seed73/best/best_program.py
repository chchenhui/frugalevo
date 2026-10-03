import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Conflict-graph-driven scheduling:
    1) Extract per-transaction read/write key sets and build a symmetric
       pairwise conflict-weight matrix (shared keys written by >= 1 side).
    2) Construct diverse initial schedules with a GRASP-style greedy over
       the conflict graph (O(n) per step, no simulator calls), enabling
       many cheap multi-start restarts.
    3) Refine with exact-cost Iterated Local Search (insertion / swap /
       2-opt descent using workload.get_opt_seq_cost) under a systematic
       simulated-annealing acceptance with geometric cooling.
    4) Spend the remaining budget on a beam search over partial schedules
       pruned by the incumbent upper bound.
    All optimization and the reported makespan use the exact cost
    function; the conflict graph is used only for fast construction.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import time
    import re

    n = workload.num_txns
    deadline = time.time() + 100.0
    exhaustive = (n * n <= 400)

    # ---------- conflict-graph extraction ----------
    writes = [set() for _ in range(n)]
    reads = [set() for _ in range(n)]
    proxy_ok = False
    try:
        src = None
        for attr in ('txn_ops', 'txns', 'operations', 'ops',
                     'txn_operations', 'workload', 'items'):
            if hasattr(workload, attr):
                src = getattr(workload, attr)
                break
        tx_list = None
        if isinstance(src, dict):
            def keynum(k):
                m = re.search(r'(\d+)', str(k))
                return int(m.group(1)) if m else 0
            tx_list = [src[k] for k in sorted(src.keys(), key=keynum)]
        elif src is not None:
            try:
                tx_list = list(src)
            except Exception:
                tx_list = None
        if tx_list is not None and len(tx_list) == n:
            parsed = True
            for idx in range(n):
                ops = tx_list[idx]
                if isinstance(ops, str):
                    ops = ops.split()
                if not isinstance(ops, (list, tuple)):
                    parsed = False
                    break
                for op in ops:
                    if isinstance(op, str):
                        s = op.strip()
                        if len(s) >= 3 and s[1] in '-_:':
                            if s[0].lower() == 'w':
                                writes[idx].add(s[2:])
                            else:
                                reads[idx].add(s[2:])
                            continue
                        parsed = False
                        break
                    t = getattr(op, 'op_type', None)
                    if t is None:
                        t = getattr(op, 'type', None)
                    k = getattr(op, 'key', None)
                    if k is None:
                        k = getattr(op, 'item', None)
                    if k is None:
                        parsed = False
                        break
                    if str(t).lower().startswith('w'):
                        writes[idx].add(str(k))
                    else:
                        reads[idx].add(str(k))
                if not parsed:
                    break
            proxy_ok = parsed
    except Exception:
        proxy_ok = False

    C = None
    if proxy_ok:
        # symmetric conflict weight: shared keys written by >= 1 side
        C = [[0] * n for _ in range(n)]
        for i in range(n):
            all_i = reads[i] | writes[i]
            for j in range(i + 1, n):
                shared = all_i & (reads[j] | writes[j])
                c = len(shared & (writes[i] | writes[j]))
                C[i][j] = c
                C[j][i] = c

    def grasp_construct(rcl=3):
        """GRASP over the conflict graph: repeatedly place the txn with
        minimum accumulated conflict weight to placed txns, choosing
        randomly among the best rcl candidates for diversification.
        O(n) per step, no simulator calls."""
        start = random.randint(0, n - 1)
        seq = [start]
        placed = [False] * n
        placed[start] = True
        score = [C[start][j] for j in range(n)]
        for _ in range(n - 1):
            rem = [j for j in range(n) if not placed[j]]
            if not rem:
                break
            rem.sort(key=lambda j: score[j])
            t = random.choice(rem[:min(rcl, len(rem))])
            seq.append(t)
            placed[t] = True
            row = C[t]
            for j in rem:
                if not placed[j]:
                    score[j] += row[j]
        return seq

    def greedy_construct(num_samples):
        if proxy_ok:
            return grasp_construct(3)
        start_txn = random.randint(0, n - 1)
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            if exhaustive:
                candidates = list(remaining)
            else:
                k = min(num_samples, len(remaining))
                candidates = random.sample(remaining, k)
            min_cost = float('inf')
            min_txn = candidates[0]
            for t in candidates:
                cost = workload.get_opt_seq_cost(txn_seq + [t])
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
            txn_seq.append(min_txn)
            remaining.remove(min_txn)
        return txn_seq

    def local_search(seq, cost):
        # Insertion + swap + 2-opt segment-reversal neighborhoods,
        # first-improvement descent until local optimum or deadline.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # insertion (relocate) moves
            for i in range(n):
                if time.time() >= deadline:
                    return cost
                t = seq.pop(i)
                for j in range(n):
                    if j == i:
                        continue
                    seq.insert(j, t)
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                        break
                    seq.pop(j)
                if improved:
                    break
                seq.insert(i, t)
            if improved:
                continue
            # swap moves
            for i in range(n - 1):
                if time.time() >= deadline:
                    return cost
                for j in range(i + 1, n):
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                        break
                    seq[i], seq[j] = seq[j], seq[i]
                if improved:
                    break
            if improved:
                continue
            # 2-opt: reverse segment [i..j], often fixes many
            # conflict orderings at once
            for i in range(n - 1):
                if time.time() >= deadline:
                    return cost
                for j in range(i + 2, n):
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                        break
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                if improved:
                    break
        return cost

    def perturb(seq, strength):
        s = seq[:]
        for _ in range(strength):
            if random.random() < 0.5:
                # relocate move
                i = random.randint(0, n - 1)
                t = s.pop(i)
                s.insert(random.randint(0, n - 1), t)
            else:
                # swap move (preserves relative structure more)
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                s[i], s[j] = s[j], s[i]
        return s

    def beam_search(best_cost, best_seq, budget):
        """Beam search over partial schedules. Prefix cost is an
        admissible lower bound (remaining txns only add cost), so
        entries with prefix_cost >= best_cost are pruned. Prefix
        costs are memoized to avoid recomputation."""
        cache = {}

        def pcost(prefix):
            key = tuple(prefix)
            c = cache.get(key)
            if c is None:
                c = workload.get_opt_seq_cost(list(prefix))
                cache[key] = c
            return c

        beam = [([t], pcost([t])) for t in range(n)]
        beam = [e for e in beam if e[1] < best_cost]
        beam.sort(key=lambda e: e[1])
        beam = beam[:max(20, min(150, 400 // max(1, n)))]
        end = time.time() + budget
        for depth in range(n - 1):
            if time.time() >= end or not beam:
                break
            nxt = []
            for prefix, c in beam:
                if time.time() >= end:
                    break
                used = set(prefix)
                for t in range(n):
                    if t in used:
                        continue
                    np_ = prefix + [t]
                    nc = pcost(np_)
                    if nc < best_cost:
                        nxt.append((np_, nc))
            if not nxt:
                break
            nxt.sort(key=lambda e: e[1])
            beam = nxt[:len(beam)]
            # a full-length entry is a complete candidate
            if len(beam[0][0]) == n:
                if beam[0][1] < best_cost:
                    best_cost = beam[0][1]
                    best_seq = beam[0][0][:]
                beam = beam[1:]
        return best_cost, best_seq

    best_seq = None
    best_cost = float('inf')
    cur_seq = None
    cur_cost = float('inf')
    stagnation = 0
    strength = max(2, n // 10)
    ils_deadline = time.time() + 40.0
    # simulated annealing: geometric cooling from ~5% of initial makespan
    T0 = None
    alpha = 0.95
    temp = None
    t_start = time.time()
    budget = 40.0
    while time.time() < ils_deadline:
        if best_seq is None:
            # fresh start from conflict-graph GRASP construction
            seq = greedy_construct(8)
            stagnation = 0
            c0 = workload.get_opt_seq_cost(seq)
            if T0 is None:
                T0 = max(2.0, c0 * 0.05)
                temp = T0
        elif cur_seq is None or stagnation >= 30:
            if random.random() < 0.4:
                # fresh GRASP restart: cheap, conflict-aware, diverse
                seq = greedy_construct(8)
            else:
                # heavy perturbation from best to diversify
                seq = perturb(best_seq, max(4, n // 5))
            stagnation = 0
        else:
            # ILS: perturb current solution
            seq = perturb(cur_seq, strength)
        cost = workload.get_opt_seq_cost(seq)
        cost = local_search(seq, cost)
        # geometric cooling schedule
        frac = (time.time() - t_start) / budget
        temp = max(0.3, T0 * (alpha ** int(frac * 120)))
        if cost < best_cost:
            best_cost = cost
            best_seq = seq[:]
            cur_seq = seq[:]
            cur_cost = cost
            stagnation = 0
            strength = max(2, n // 10)  # reset kick size on improvement
        elif cost < cur_cost:
            # accept improving move relative to current
            cur_seq = seq[:]
            cur_cost = cost
            stagnation += 1
        elif cost == cur_cost:
            # plateau drift: accept equal-cost solution to wander
            cur_seq = seq[:]
            cur_cost = cost
            stagnation += 1
            strength = min(strength + 1, n // 2)
        elif random.random() < pow(2.718281828, -(cost - cur_cost) / max(temp, 1e-9)):
            # simulated annealing: accept worse solutions with a
            # cooling probability, systematically escaping local optima
            cur_seq = seq[:]
            cur_cost = cost
            stagnation += 1
        else:
            stagnation += 1
            strength = min(strength + 1, n // 2)

    if best_seq is None:
        best_seq = list(range(n))
        best_cost = workload.get_opt_seq_cost(best_seq)
    # Beam search with remaining budget, pruned by ILS upper bound.
    remaining = deadline - time.time()
    if remaining > 2.0:
        best_cost, best_seq = beam_search(best_cost, best_seq, remaining - 1.0)
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
