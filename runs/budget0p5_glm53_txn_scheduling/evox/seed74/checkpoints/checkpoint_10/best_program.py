import random
import time

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Time-budgeted multi-restart greedy with cost caching, plus hybrid
    local search (adjacent swaps + random insertion moves).

    Greedy phase: repeatedly append the candidate transaction (sampled
    from remaining) that yields the lowest actual makespan via
    get_opt_seq_cost, cached by exact sequence.

    Local search: alternate full adjacent-swap passes with random
    insertion moves (remove txn at i, reinsert at j) to escape
    swap-local optima. Restarts continue while the time budget remains.
    """
    deadline = time.time() + 100.0
    cost_cache = {}

    def cached_cost(seq):
        key = tuple(seq)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(list(seq))
        return cost_cache[key]

    def greedy_from(start_txn, num_samples):
        txn_seq = [start_txn]
        remaining = [x for x in range(workload.num_txns) if x != start_txn]
        while remaining:
            k = min(num_samples, len(remaining))
            candidates = random.sample(remaining, k)
            min_cost, min_txn = float('inf'), -1
            for t in candidates:
                cost = cached_cost(txn_seq + [t])
                if cost < min_cost:
                    min_cost, min_txn = cost, t
            txn_seq.append(min_txn)
            remaining.remove(min_txn)
        return txn_seq

    def local_search(seq):
        seq = list(seq)
        best = cached_cost(seq)
        n = len(seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # adjacent swap pass
            for i in range(n - 1):
                cand = seq[:i] + [seq[i + 1], seq[i]] + seq[i + 2:]
                c = cached_cost(cand)
                if c < best:
                    best, seq, improved = c, cand, True
            # random insertion moves
            for _ in range(n):
                i = random.randrange(n)
                j = random.randrange(n)
                if i == j:
                    continue
                cand = seq[:i] + seq[i + 1:]
                cand.insert(j, seq[i])
                c = cached_cost(cand)
                if c < best:
                    best, seq, improved = c, cand, True
        return best, seq

    best_cost, best_seq = None, None
    starts = list(range(workload.num_txns))
    random.shuffle(starts)
    for start_txn in starts:
        if time.time() >= deadline:
            break
        seq = greedy_from(start_txn, 10)
        cost, seq = local_search(seq)
        if best_cost is None or cost < best_cost:
            best_cost, best_seq = cost, seq
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
