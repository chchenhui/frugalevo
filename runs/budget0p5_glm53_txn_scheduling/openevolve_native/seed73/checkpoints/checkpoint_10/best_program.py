import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def _local_search(workload, seq, cost, deadline):
    """First-improvement local search: pairwise swaps + single-txn reinsertion.

    Swaps fix mis-ordered pairs; Or-opt reinsertion moves a single
    transaction to its best position, which swaps cannot achieve alone.
    Both use true cost evaluation.
    """
    import time
    n = len(seq)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return seq, cost
                seq[i], seq[j] = seq[j], seq[i]
                c = workload.get_opt_seq_cost(seq)
                if c < cost:
                    cost = c
                    improved = True
                else:
                    seq[i], seq[j] = seq[j], seq[i]
        for i in range(n):
            if time.time() > deadline:
                return seq, cost
            t = seq.pop(i)
            best_j, best_c = i, cost
            for j in range(n):
                if j == i:
                    continue
                seq.insert(j, t)
                c = workload.get_opt_seq_cost(seq)
                del seq[j]
                if c < best_c:
                    best_c, best_j = c, j
            seq.insert(best_j, t)
            if best_j != i:
                cost = best_c
                improved = True
    return seq, cost


def _perturb(seq):
    """Double-bridge + random swaps + Or-opt segment move: escape local
    optima while keeping most of the ordering structure intact."""
    n = len(seq)
    if n >= 8:
        a, b, c, d = sorted(random.sample(range(n), 4))
        seq[:] = seq[:a] + seq[c:d] + seq[b:c] + seq[a:b] + seq[d:]
    for _ in range(3):
        i, j = random.randrange(n), random.randrange(n)
        seq[i], seq[j] = seq[j], seq[i]
    if n >= 6:
        seg_len = random.choice([2, 3])
        i = random.randint(0, n - seg_len)
        block = seq[i:i + seg_len]
        rest = seq[:i] + seq[i + seg_len:]
        j = random.randint(0, len(rest))
        seq[:] = rest[:j] + block + rest[j:]


def get_best_schedule(workload, num_seqs):
    """Iterated local search (ILS) for transaction scheduling.

    1. Construct: greedy — repeatedly append the txn that increases
       true makespan the least.
    2. Improve: 2-opt swaps + single-txn reinsertion to local optimum.
    3. Iterate: perturb (double-bridge + swaps + segment move),
       re-optimize, accept with simulated-annealing criterion so the
       search can traverse worse regions; track global best.
    Runs within a ~100s wall-clock budget.
    """
    import time
    import math

    deadline = time.time() + 100.0
    n = workload.num_txns

    def _greedy_construct():
        start_txn = random.randint(0, n - 1)
        seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining and time.time() < deadline:
            min_cost = None
            min_txn = None
            for t in remaining:
                c = workload.get_opt_seq_cost(seq + [t])
                if min_cost is None or c < min_cost:
                    min_cost, min_txn = c, t
            seq.append(min_txn)
            remaining.remove(min_txn)
        if remaining:
            seq.extend(remaining)
        return seq

    seq = _greedy_construct()

    cost = workload.get_opt_seq_cost(seq)
    seq, cost = _local_search(workload, seq, cost, deadline)
    best_seq, best_cost = seq[:], cost
    cur_seq, cur_cost = seq[:], cost

    # SA-style acceptance with geometric cooling; reheat from best
    # occasionally to keep exploring around the incumbent.
    temp = max(1.0, cur_cost * 0.02)
    since_best = 0
    while time.time() < deadline:
        cand = cur_seq[:]
        _perturb(cand)
        c = workload.get_opt_seq_cost(cand)
        cand, c = _local_search(workload, cand, c, deadline)
        if c < best_cost:
            best_cost, best_seq = c, cand[:]
            since_best = 0
        else:
            since_best += 1
        if c <= cur_cost or random.random() < math.exp(-(c - cur_cost) / temp):
            cur_seq, cur_cost = cand[:], c
        temp *= 0.995
        if temp < 0.5:
            temp = max(1.0, best_cost * 0.01)
        # Stagnation: restart from a fresh greedy construction.
        if since_best > 150:
            cur_seq, cur_cost = best_seq[:], best_cost
            temp = max(1.0, best_cost * 0.02)
            since_best = 0

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
