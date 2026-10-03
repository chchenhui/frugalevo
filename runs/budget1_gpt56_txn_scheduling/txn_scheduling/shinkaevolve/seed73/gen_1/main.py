import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using cost-guided beam search and
    insertion local search.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    # The simulator is the objective function, so cache every evaluated prefix
    # and complete order.  Local search otherwise evaluates many duplicates.
    cost_cache = {}

    def cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(list(key))
        return cost_cache[key]

    width = max(4, int(num_seqs))
    width = min(width, max(n, 1))

    # Retain several alternatives at every depth instead of committing to one
    # randomly selected greedy prefix.
    beam = [(txn,) for txn in range(n)]
    beam.sort(key=lambda seq: (cost(seq), seq))
    beam = beam[:width]

    for depth in range(1, n):
        expanded = []
        for prefix in beam:
            used = set(prefix)
            for txn in range(n):
                if txn not in used:
                    candidate = prefix + (txn,)
                    expanded.append((cost(candidate), candidate))
        expanded.sort(key=lambda entry: (entry[0], entry[1]))
        beam = [sequence for _, sequence in expanded[:width]]

    candidates = [list(sequence) for sequence in beam]

    # Add full greedy schedules from distinct starts.  They are cheap useful
    # alternatives when an early beam ranking is locally misleading.
    start_count = min(n, max(width, 8))
    for start in range(start_count):
        sequence = [start]
        remaining = set(range(n))
        remaining.remove(start)
        while remaining:
            next_txn = min(
                remaining,
                key=lambda txn: (cost(sequence + [txn]), txn)
            )
            sequence.append(next_txn)
            remaining.remove(next_txn)
        candidates.append(sequence)

    def improve_by_insertion(sequence):
        """Repeated best transaction relocation under the true makespan."""
        sequence = list(sequence)
        current_cost = cost(sequence)

        # A few passes capture interacting relocations without unbounded work.
        for _ in range(3):
            best_cost = current_cost
            best_sequence = None
            for source in range(n):
                txn = sequence[source]
                reduced = sequence[:source] + sequence[source + 1:]
                for destination in range(n):
                    trial = reduced[:destination] + [txn] + reduced[destination:]
                    trial_cost = cost(trial)
                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        best_sequence = trial
            if best_sequence is None:
                break
            sequence = best_sequence
            current_cost = best_cost
        return current_cost, sequence

    best_cost = float("inf")
    best_schedule = None
    for candidate in candidates:
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