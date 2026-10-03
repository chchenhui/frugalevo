import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan serial transaction order.

    A beam preserves several competing conflict orderings, then insertion local
    search optimizes complete orders using the simulator's exact objective.
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    cost_cache = {}

    def cost(sequence):
        sequence = tuple(sequence)
        if sequence not in cost_cache:
            cost_cache[sequence] = workload.get_opt_seq_cost(list(sequence))
        return cost_cache[sequence]

    # Unlike random sampled greedy selection, retain alternatives whose early
    # conflict cost is close to the current best.  The cap bounds evaluations
    # for large workloads while num_seqs still increases search effort.
    beam_width = min(32, max(4, num_seqs * 3))
    all_txns = tuple(range(n))
    beam = [((), all_txns)]

    for _ in range(n):
        candidates = []
        for prefix, remaining in beam:
            for txn in remaining:
                next_prefix = prefix + (txn,)
                next_remaining = tuple(x for x in remaining if x != txn)
                candidates.append((cost(next_prefix), next_prefix, next_remaining))

        candidates.sort(key=lambda entry: (entry[0], entry[1]))
        beam = []
        seen = set()
        for _, prefix, remaining in candidates:
            if prefix not in seen:
                beam.append((prefix, remaining))
                seen.add(prefix)
            if len(beam) == beam_width:
                break

    def improve_by_relocation(sequence):
        """Best-improvement insertion moves, scored by final makespan."""
        sequence = list(sequence)
        current_cost = cost(sequence)
        improved = True
        while improved:
            improved = False
            best_cost = current_cost
            best_sequence = sequence
            for source in range(n):
                reduced = sequence[:source] + sequence[source + 1:]
                for destination in range(n):
                    candidate = reduced[:destination] + [sequence[source]] + reduced[destination:]
                    candidate_cost = cost(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_sequence = candidate
            if best_cost < current_cost:
                sequence = best_sequence
                current_cost = best_cost
                improved = True
        return current_cost, sequence

    best_cost = float("inf")
    best_schedule = None
    for sequence, _ in beam:
        candidate_cost, candidate_schedule = improve_by_relocation(sequence)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_schedule = candidate_schedule

    return best_cost, best_schedule

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