import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using conflict-aware beam search
    and insertion local search.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    # Every decision is scored by the simulator's actual dependency-aware
    # makespan.  Cache scores since beam expansion and local search can reach
    # the same ordering more than once.
    cost_cache = {}

    def sequence_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    beam_width = max(1, min(num_seqs, num_txns))
    all_txns = tuple(range(num_txns))

    # Keep several good prefixes instead of committing to one random greedy
    # path.  A prefix is represented as (cost, tuple_of_transactions).
    beam = [(sequence_cost([txn]), (txn,)) for txn in all_txns]
    beam.sort(key=lambda entry: (entry[0], entry[1]))
    beam = beam[:beam_width]

    for _ in range(1, num_txns):
        expanded = []
        for _, prefix in beam:
            used = set(prefix)
            for txn in all_txns:
                if txn not in used:
                    candidate = prefix + (txn,)
                    expanded.append((sequence_cost(candidate), candidate))

        # Distinct prefixes are normally generated once, but retaining this
        # guard makes the beam robust if its construction changes later.
        expanded.sort(key=lambda entry: (entry[0], entry[1]))
        beam = []
        seen = set()
        for entry in expanded:
            if entry[1] not in seen:
                beam.append(entry)
                seen.add(entry[1])
                if len(beam) == beam_width:
                    break

    def improve_by_insertions(sequence, current_cost):
        """Best-improvement relocation search using full makespan evaluations."""
        sequence = list(sequence)
        improved = True
        while improved:
            improved = False
            best_cost = current_cost
            best_sequence = sequence

            for source in range(num_txns):
                txn = sequence[source]
                without_txn = sequence[:source] + sequence[source + 1:]
                for destination in range(num_txns):
                    candidate = without_txn[:destination] + [txn] + without_txn[destination:]
                    if candidate == sequence:
                        continue
                    candidate_cost = sequence_cost(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_sequence = candidate

            if best_cost < current_cost:
                sequence = best_sequence
                current_cost = best_cost
                improved = True

        return current_cost, sequence

    # Local optimization is applied to the best complete beam schedules.  The
    # beam already supplies different conflict patterns, while relocation can
    # correct a poor earlier placement without rebuilding the entire schedule.
    best_cost = None
    best_schedule = None
    for cost, schedule in beam:
        cost, schedule = improve_by_insertions(schedule, cost)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_schedule = schedule

    if workload.debug:
        print("best schedule:", best_schedule, "makespan:", best_cost)

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