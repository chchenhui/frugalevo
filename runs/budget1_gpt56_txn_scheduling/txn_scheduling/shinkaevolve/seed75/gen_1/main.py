import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Search transaction permutations using conflict-aware beam construction and
    insertion local search.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    transactions = list(range(workload.num_txns))
    if not transactions:
        return 0, []

    # Prefix costs are repeatedly encountered while expanding the beam and
    # during local improvement, so avoid re-running the simulator for them.
    cost_cache = {}

    def sequence_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(list(sequence))
        return cost_cache[key]

    # Keep several low-cost alternatives instead of committing to one sampled
    # greedy choice.  Prefixes are scored by the actual conflict scheduler.
    beam_width = max(1, min(num_seqs, workload.num_txns))
    beam = [()]
    for _ in range(workload.num_txns):
        expanded = []
        seen = set()
        for prefix in beam:
            used = set(prefix)
            for txn in transactions:
                if txn not in used:
                    candidate = prefix + (txn,)
                    if candidate not in seen:
                        seen.add(candidate)
                        expanded.append((sequence_cost(candidate), candidate))
        expanded.sort(key=lambda entry: (entry[0], entry[1]))
        beam = [candidate for _, candidate in expanded[:beam_width]]

    def improve_by_insertion(schedule):
        """Apply best improving transaction moves until locally optimal."""
        current = list(schedule)
        current_cost = sequence_cost(current)
        while True:
            best_cost = current_cost
            best_schedule = None
            for source in range(len(current)):
                reduced = current[:source] + current[source + 1:]
                for destination in range(len(current)):
                    candidate = reduced[:destination] + [current[source]] + reduced[destination:]
                    candidate_cost = sequence_cost(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_schedule = candidate

            if best_schedule is None:
                return current_cost, current
            current, current_cost = best_schedule, best_cost

    best_cost = float("inf")
    best_schedule = None
    for candidate in beam:
        candidate_cost, candidate_schedule = improve_by_insertion(candidate)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_schedule = candidate_schedule

    if workload.debug:
        print("best schedule:", best_schedule, "cost:", best_cost)
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