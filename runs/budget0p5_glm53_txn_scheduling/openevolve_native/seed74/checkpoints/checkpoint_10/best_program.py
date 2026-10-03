import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Full-candidate greedy construction with multiple restarts,
    followed by adjacent-swap local search on true cost.

    1. For each start transaction (all starts for small workloads,
       sampled starts for large ones), greedily build a sequence:
       at each step append the remaining transaction that yields
       the lowest true cost via workload.get_opt_seq_cost.
    2. Polish each greedy solution with a first-improvement
       adjacent-swap local search on true cost.
    3. Return the best (cost, sequence) found.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns

    def greedy_from(start_txn):
        txn_seq = [start_txn]
        remaining = [t for t in range(n) if t != start_txn]
        while remaining:
            best_cost = None
            best_txns = []
            for t in remaining:
                cand = txn_seq + [t]
                cost = workload.get_opt_seq_cost(cand)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_txns = [t]
                elif cost == best_cost:
                    best_txns.append(t)
            pick = random.choice(best_txns)
            txn_seq.append(pick)
            remaining.remove(pick)
        return txn_seq

    def local_search(seq):
        """Adjacent-swap + random-pair-swap first-improvement search on true cost."""
        seq = list(seq)
        cost = workload.get_opt_seq_cost(seq)
        improved = True
        while improved:
            improved = False
            for i in range(len(seq) - 1):
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                new_cost = workload.get_opt_seq_cost(seq)
                if new_cost < cost:
                    cost = new_cost
                    improved = True
                else:
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
            for _ in range(min(n, 20)):
                i, j = sorted(random.sample(range(n), 2))
                seq[i], seq[j] = seq[j], seq[i]
                new_cost = workload.get_opt_seq_cost(seq)
                if new_cost < cost:
                    cost = new_cost
                    improved = True
                else:
                    seq[i], seq[j] = seq[j], seq[i]
        return cost, seq

    if n <= 15:
        starts = list(range(n))
    else:
        starts = random.sample(range(n), min(12, n))

    best_cost = None
    best_seq = None
    for s in starts:
        seq = greedy_from(s)
        cost = workload.get_opt_seq_cost(seq)
        if best_cost is None or cost < best_cost:
            best_cost, best_seq = cost, seq

    # Polish only the best greedy solution with local search.
    best_cost, best_seq = local_search(best_seq)

    # Perturbation restarts: reverse a random segment, re-polish, keep improvements.
    for _ in range(min(4 * n, 200)):
        cand = best_seq[:]
        i, j = sorted(random.sample(range(n), 2))
        cand[i:j + 1] = reversed(cand[i:j + 1])
        cost, cand = local_search(cand)
        if cost < best_cost:
            best_cost, best_seq = cost, cand

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
