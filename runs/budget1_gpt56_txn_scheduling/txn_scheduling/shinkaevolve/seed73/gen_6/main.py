import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using exact scoring, restarts, and
    local search over transaction orderings.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    # get_opt_seq_cost is comparatively expensive and the same prefixes occur
    # in multiple greedy starts and local-search moves.
    cost_cache = {}

    def sequence_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(list(sequence))
        return cost_cache[key]

    # Small instances are cheap enough to solve exactly.  This is especially
    # useful for workloads where every conflict orientation matters.
    if num_txns <= 8:
        import itertools

        best_sequence = None
        best_cost = float("inf")
        for sequence in itertools.permutations(range(num_txns)):
            cost = sequence_cost(sequence)
            if cost < best_cost:
                best_cost = cost
                best_sequence = list(sequence)
        return best_cost, best_sequence

    num_starts = max(1, int(num_seqs))
    constructed = []

    for start in range(num_starts):
        sequence = []
        remaining = list(range(num_txns))

        while remaining:
            # Score actual incremental schedules, rather than sampling or using
            # transaction length proxies.  The first start is pure greedy; the
            # others diversify among equally promising conflict orientations.
            choices = [
                (sequence_cost(sequence + [txn]), txn)
                for txn in remaining
            ]
            choices.sort(key=lambda choice: (choice[0], choice[1]))

            rcl_size = 1 if start == 0 else min(3, len(choices))
            _, chosen = choices[0] if rcl_size == 1 else choices[
                random.randrange(rcl_size)
            ]
            sequence.append(chosen)
            remaining.remove(chosen)

        constructed.append((sequence_cost(sequence), sequence))

    constructed.sort(key=lambda candidate: candidate[0])
    best_cost, best_sequence = constructed[0]

    # For large workloads, refine only the most promising starts to keep the
    # number of full simulator evaluations bounded.
    local_limit = num_starts if num_txns <= 20 else min(3, num_starts)

    for _, initial_sequence in constructed[:local_limit]:
        sequence = initial_sequence[:]
        current_cost = sequence_cost(sequence)

        # A relocation changes the ordering of all conflicts involving one
        # transaction.  Swaps additionally escape common relocation plateaus.
        for _ in range(min(num_txns, 12)):
            move_cost = current_cost
            move_sequence = None

            for old_index in range(num_txns):
                txn = sequence[old_index]
                without_txn = sequence[:old_index] + sequence[old_index + 1:]
                for new_index in range(num_txns):
                    if new_index == old_index:
                        continue
                    candidate = (
                        without_txn[:new_index]
                        + [txn]
                        + without_txn[new_index:]
                    )
                    candidate_cost = sequence_cost(candidate)
                    if candidate_cost < move_cost:
                        move_cost = candidate_cost
                        move_sequence = candidate

            for left in range(num_txns):
                for right in range(left + 1, num_txns):
                    candidate = sequence[:]
                    candidate[left], candidate[right] = (
                        candidate[right],
                        candidate[left],
                    )
                    candidate_cost = sequence_cost(candidate)
                    if candidate_cost < move_cost:
                        move_cost = candidate_cost
                        move_sequence = candidate

            if move_sequence is None:
                break
            sequence = move_sequence
            current_cost = move_cost

        if current_cost < best_cost:
            best_cost = current_cost
            best_sequence = sequence

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