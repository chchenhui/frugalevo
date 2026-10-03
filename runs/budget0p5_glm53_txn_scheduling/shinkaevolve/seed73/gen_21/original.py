import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan schedule via beam search with randomized restarts.
    Maintains a beam of partial orderings, expanding each with all remaining
    transactions and pruning to the best k by actual makespan cost.
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    best_cost = float('inf')
    best_seq = None

    # random tie-breaking seeds to escape local optima over restarts
    num_restarts = max(1, min(8, 200 // max(1, n)))
    beam_width = max(2, min(6, 400 // max(1, n)))

    for restart in range(num_restarts):
        # Beam of (cost, seq, remaining_frozenset)
        # start with a few random seeds plus all-zero-cost options
        seeds = set()
        seeds.add(0)
        seeds.add(random.randrange(n))
        if n > 1:
            seeds.add(random.randrange(n))
        beam = []
        for s in seeds:
            seq = [s]
            rem = [x for x in range(n) if x != s]
            c = workload.get_opt_seq_cost(seq)
            beam.append((c, seq, rem))
        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        while beam and len(beam[0][1]) < n:
            candidates = []
            for c, seq, rem in beam:
                if not rem:
                    continue
                # expand with all remaining transactions (full lookahead)
                for t in rem:
                    new_seq = seq + [t]
                    new_rem = [x for x in rem if x != t]
                    cost = workload.get_opt_seq_cost(new_seq)
                    candidates.append((cost, new_seq, new_rem))
            if not candidates:
                break
            # prune to beam width, with random tie-break jitter to diversify
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_width]

        if beam:
            c, seq, rem = beam[0]
            final_cost = workload.get_opt_seq_cost(seq)
            if final_cost < best_cost:
                best_cost = final_cost
                best_seq = seq

    assert best_seq is not None and len(set(best_seq)) == n
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
