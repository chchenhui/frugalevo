import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan schedule using greedy construction with incremental
    cost evaluation (when available) followed by long simulated annealing
    with mixed move neighborhoods.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import math
    import time

    n = workload.num_txns

    # ------------------------------------------------------------------
    # Build a fast evaluator. Prefer workload.get_incremental_seq_cost,
    # verifying it matches workload.get_opt_seq_cost exactly. Fall back
    # to the full evaluator otherwise.
    # ------------------------------------------------------------------
    use_incremental = False
    try:
        def inc_eval(seq):
            key_map = {}
            cost = 0
            for t in seq:
                key_map, cost = workload.get_incremental_seq_cost(t, key_map, cost)
            return cost

        # verify on a few random permutations
        for _ in range(3):
            perm = list(range(n))
            random.shuffle(perm)
            if inc_eval(perm) != workload.get_opt_seq_cost(perm):
                break
        else:
            use_incremental = True
    except Exception:
        use_incremental = False

    if use_incremental:
        def eval_cost(seq):
            return inc_eval(seq)
    else:
        def eval_cost(seq):
            return workload.get_opt_seq_cost(seq)

    # ------------------------------------------------------------------
    # Greedy construction using incremental state: evaluating each
    # remaining candidate is one incremental call, so we can consider
    # ALL remaining transactions each step (true greedy), optionally
    # with randomized candidate subsampling for diversification.
    # ------------------------------------------------------------------
    def greedy(sample_rate=1.0, candidate_cap=None):
        start = random.randint(0, n - 1)
        seq = [start]
        remaining = [x for x in range(n) if x != start]
        key_map = {}
        cost = 0
        if use_incremental:
            key_map, cost = workload.get_incremental_seq_cost(start, key_map, cost)
        else:
            cost = eval_cost(seq)

        while remaining:
            if random.random() > sample_rate:
                # random acceptance for diversification
                idx = random.randint(0, len(remaining) - 1)
                t = remaining.pop(idx)
            else:
                cands = remaining
                if candidate_cap is not None and len(cands) > candidate_cap:
                    cands = random.sample(cands, candidate_cap)
                best_t = None
                best_c = float('inf')
                best_km = None
                for t in cands:
                    if use_incremental:
                        km, c = workload.get_incremental_seq_cost(t, key_map, cost)
                    else:
                        c = eval_cost(seq + [t])
                        km = None
                    if c < best_c:
                        best_c, best_t, best_km = c, t, km
                t = best_t
                if use_incremental:
                    key_map = best_km
                cost = best_c
                remaining.remove(t)
            if not use_incremental:
                cost = eval_cost(seq + [t])
            seq.append(t)

        return eval_cost(seq), seq

    # ------------------------------------------------------------------
    # Local search: first-improvement adjacent swaps
    # ------------------------------------------------------------------
    def local_search(seq, cost, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(seq) - 1):
                if time.time() > deadline:
                    return seq, cost
                cand = seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < cost:
                    seq, cost = cand, c
                    improved = True
        return seq, cost

    def perturb(seq, k):
        s = seq[:]
        for _ in range(k):
            i = random.randint(0, len(s) - 1)
            j = random.randint(0, len(s) - 1)
            if i != j:
                t = s.pop(i)
                s.insert(j, t)
        return s

    start_time = time.time()
    time_limit = 30.0
    end_time = start_time + time_limit + 10.0

    # --- Phase 1: many greedy restarts (cheap now with incremental eval) ---
    best_cost = float('inf')
    best_seq = None
    greedy_deadline = min(start_time + time_limit * 0.35, end_time)
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        if trial % 3 == 0:
            cost, seq = greedy(sample_rate=1.0)
        elif trial % 3 == 1:
            cost, seq = greedy(sample_rate=random.uniform(0.6, 0.95))
        else:
            cost, seq = greedy(sample_rate=1.0, candidate_cap=15)
        if cost < best_cost:
            best_cost, best_seq = cost, seq

    if best_seq is None:
        best_cost, best_seq = greedy(1.0)

    # --- Phase 2: quick polish of best greedy ---
    best_seq, best_cost = local_search(best_seq, best_cost,
                                        min(start_time + time_limit * 0.5, end_time))

    # --- Phase 3: long SA with mixed neighborhoods ---
    cur_seq = best_seq[:]
    cur_cost = best_cost
    T0 = max(1.0, best_cost * 0.02)
    T = T0
    alpha = 0.99995
    T_min = 0.05
    since_best = 0
    kick_thresh = 4000
    while time.time() < end_time:
        T *= alpha
        if T < T_min:
            T = T_min
        r = random.random()
        cand = cur_seq[:]
        if r < 0.60 or n < 3:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            t = cand.pop(i)
            cand.insert(j, t)
        elif r < 0.85:
            i = random.randint(0, n - 2)
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
        else:
            span = min(n - 1, random.randint(2, 4))
            i = random.randint(0, n - span - 1)
            j = random.randint(0, n - span - 1)
            if i == j:
                continue
            seg = cand[i:i + span]
            rest = cand[:i] + cand[i + span:]
            cand = rest[:j] + seg + rest[j:]
        c = eval_cost(cand)
        delta = c - cur_cost
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-9)):
            cur_seq, cur_cost = cand, c
            since_best += 1
            if c < best_cost:
                best_cost, best_seq = c, cand[:]
                since_best = 0
        if since_best >= kick_thresh:
            cur_seq = perturb(best_seq, 3)
            cur_cost = eval_cost(cur_seq)
            T = max(T, T0 * 0.3)
            since_best = 0

    # --- Final polish ---
    best_seq, best_cost = local_search(best_seq, best_cost, end_time + 2.0)

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