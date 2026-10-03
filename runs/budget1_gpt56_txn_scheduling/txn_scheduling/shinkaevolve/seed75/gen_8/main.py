import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan serial transaction ordering.

    Greedy choices and local-search moves are scored by the simulator's real
    conflict-aware makespan, rather than by transaction size proxies.
    """
    if workload.num_txns == 0:
        return 0, []

    cost_cache = {}

    def schedule_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    def greedy_schedule():
        sequence = []
        remaining = list(range(workload.num_txns))

        while remaining:
            # Shuffle first so equally good conflict choices do not make every
            # restart follow exactly the same path.
            candidates = remaining[:]
            random.shuffle(candidates)

            best_txn = candidates[0]
            best_cost = None
            for txn in candidates:
                candidate_cost = schedule_cost(sequence + [txn])
                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_txn = txn

            sequence.append(best_txn)
            remaining.remove(best_txn)

        return sequence

    def improve_by_insertion(sequence):
        """Use best improving one-transaction moves to repair greedy choices."""
        sequence = sequence[:]
        current_cost = schedule_cost(sequence)

        # A small number of full best-improvement passes captures long conflict
        # chains without allowing local search to dominate runtime.
        for _ in range(3):
            best_cost = current_cost
            best_sequence = None
            size = len(sequence)

            for source in range(size):
                for destination in range(size):
                    if source == destination:
                        continue
                    candidate = sequence[:]
                    txn = candidate.pop(source)
                    candidate.insert(destination, txn)
                    candidate_cost = schedule_cost(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_sequence = candidate

            if best_sequence is None:
                break
            sequence = best_sequence
            current_cost = best_cost

        return current_cost, sequence

    best_cost = None
    best_schedule = None
    for _ in range(max(1, num_seqs)):
        candidate_cost, candidate_schedule = improve_by_insertion(greedy_schedule())
        if best_cost is None or candidate_cost < best_cost:
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