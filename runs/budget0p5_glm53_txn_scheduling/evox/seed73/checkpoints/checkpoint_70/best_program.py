import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Iterated local search (ILS) with insertion descent and ruin-and-recreate:
      1) Seed with randomized true-cost greedy constructions.
      2) Refine with full insertion descent (relocate each transaction to
         its best position, judged by actual makespan) until local optimum.
      3) Perturb by removing a contiguous block or scattered set of
         transactions and greedily reinserting each at its best position.
      4) Accept if cost <= incumbent (plateau drift) and repeat until the
         time budget is exhausted.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import time as _time
    deadline = _time.time() + 110  # per-workload budget; 3 workloads < 360s
    n = workload.num_txns

    def cost_of(seq):
        return workload.get_opt_seq_cost(seq)

    def greedy_seed():
        # true-cost greedy: append the txn minimizing actual makespan
        remaining = list(range(n))
        random.shuffle(remaining)
        seq = [remaining.pop()]
        while remaining:
            best_t, best_c = None, float('inf')
            for t in remaining:
                c = cost_of(seq + [t])
                if c < best_c:
                    best_c, best_t = c, t
            seq.append(best_t)
            remaining.remove(best_t)
        return seq

    def insertion_descent(seq, cost):
        # full-neighborhood insertion descent until local optimum
        improved = True
        while improved:
            improved = False
            for i in range(n):
                t = seq.pop(i)
                best_j, best_c = i, cost
                for j in range(n):
                    seq.insert(j, t)
                    c = cost_of(seq)
                    if c < best_c:
                        best_c, best_j = c, j
                    seq.pop(j)
                seq.insert(best_j, t)
                if best_c < cost:
                    cost, improved = best_c, True
        return cost, seq

    # --- Phase 1: several greedy seeds, refine the best ---
    best_c, best_seq = float('inf'), list(range(n))
    for _ in range(3):
        seq = greedy_seed()
        c, seq = insertion_descent(seq, cost_of(seq))
        if c < best_c:
            best_c, best_seq = c, seq[:]
        if _time.time() > deadline:
            return best_c, best_seq

    # --- Phase 2: iterated local search with ruin-and-recreate ---
    while _time.time() < deadline:
        cur = best_seq[:]
        k = random.randint(max(2, n // 10), max(3, n // 5))
        if random.random() < 0.5 and k >= 2:
            # contiguous block ruin
            a = random.randint(0, n - k)
            removed = cur[a:a + k]
            del cur[a:a + k]
        else:
            # scattered ruin
            idxs = sorted(random.sample(range(n), k), reverse=True)
            removed = [cur[i] for i in idxs]
            for i in idxs:
                del cur[i]
        # greedy recreate: insert each removed txn at its best position
        random.shuffle(removed)
        for t in removed:
            best_j, best_c2 = 0, float('inf')
            for j in range(len(cur) + 1):
                cur.insert(j, t)
                c = cost_of(cur)
                if c < best_c2:
                    best_c2, best_j = c, j
                cur.pop(j)
            cur.insert(best_j, t)
        # short descent on the recreated solution
        c, cur = insertion_descent(cur, cost_of(cur))
        if c <= best_c:
            best_c, best_seq = c, cur[:]

    return best_c, best_seq

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
