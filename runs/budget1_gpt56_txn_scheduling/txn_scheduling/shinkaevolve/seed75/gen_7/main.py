import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using objective-aware beam search.

    Partial schedules are scored by the simulator itself, so choices account
    directly for read/write conflict delays rather than transaction metadata.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    cost_cache = {}

    def schedule_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    # Keep several alternative low-cost prefixes.  Unlike random sampling,
    # every possible next transaction is considered for every beam member.
    beam_width = max(1, min(num_seqs, num_txns))
    beam = [([], tuple(range(num_txns)), 0)]

    for _ in range(num_txns):
        expanded = []
        for prefix, remaining, _ in beam:
            for txn in remaining:
                candidate = prefix + [txn]
                next_remaining = tuple(x for x in remaining if x != txn)
                expanded.append(
                    (candidate, next_remaining, schedule_cost(candidate))
                )

        expanded.sort(key=lambda state: (state[2], tuple(state[0])))
        beam = expanded[:beam_width]

    best_sequence, _, best_cost = min(
        beam, key=lambda state: (state[2], tuple(state[0]))
    )

    # Relocation search repairs early beam decisions.  Testing every insertion
    # position is substantially stronger than swapping only adjacent entries.
    # Two improving passes capture interacting conflict chains without making
    # the search unbounded.
    for _ in range(2):
        improved_sequence = best_sequence
        improved_cost = best_cost

        for source_index in range(num_txns):
            moved_txn = best_sequence[source_index]
            reduced = (
                best_sequence[:source_index] + best_sequence[source_index + 1:]
            )
            for destination_index in range(num_txns):
                candidate = (
                    reduced[:destination_index]
                    + [moved_txn]
                    + reduced[destination_index:]
                )
                candidate_cost = schedule_cost(candidate)
                if candidate_cost < improved_cost:
                    improved_sequence = candidate
                    improved_cost = candidate_cost

        if improved_cost >= best_cost:
            break
        best_sequence, best_cost = improved_sequence, improved_cost

    return best_cost, best_sequence

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