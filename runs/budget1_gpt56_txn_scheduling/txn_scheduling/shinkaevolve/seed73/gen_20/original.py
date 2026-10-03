import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Build several conflict-aware greedy schedules, then improve the best one
    with moves evaluated against the complete makespan.
    """
    if workload.num_txns == 0:
        return 0, []

    cost_cache = {}

    def sequence_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    def greedy_from(start_txn):
        sequence = [start_txn]
        remaining = set(range(workload.num_txns))
        remaining.remove(start_txn)

        # At every position, score every feasible next transaction using the
        # simulator's real prefix cost rather than a length-based proxy or a
        # random subset of candidates.
        while remaining:
            candidate_costs = []
            best_prefix_cost = None
            for txn in remaining:
                cost = sequence_cost(sequence + [txn])
                if best_prefix_cost is None or cost < best_prefix_cost:
                    best_prefix_cost = cost
                    candidate_costs = [txn]
                elif cost == best_prefix_cost:
                    candidate_costs.append(txn)

            # Random tie breaking gives repeated starts useful diversification
            # without accepting a candidate with a worse measured cost.
            selected = random.choice(candidate_costs)
            sequence.append(selected)
            remaining.remove(selected)

        return sequence_cost(sequence), sequence

    # Starting with different transactions is especially important for
    # read/write hotspots: the first writer or reader can determine the
    # critical dependency chain for the entire schedule.
    starts = list(range(workload.num_txns))
    random.shuffle(starts)
    attempts = max(1, num_seqs)
    best_cost = None
    best_schedule = None

    for attempt in range(attempts):
        if attempt < len(starts):
            start = starts[attempt]
        else:
            start = random.randrange(workload.num_txns)
        cost, schedule = greedy_from(start)
        if best_cost is None or cost < best_cost:
            best_cost, best_schedule = cost, schedule

    # Greedy prefixes cannot always see a conflict that becomes critical only
    # after later transactions are added.  Relocation and swap moves evaluate
    # complete schedules and repair those decisions directly.
    for _ in range(4):
        improved_cost = best_cost
        improved_schedule = best_schedule
        n = len(best_schedule)

        for source in range(n):
            shortened = best_schedule[:source] + best_schedule[source + 1:]
            moved_txn = best_schedule[source]
            for destination in range(n):
                candidate = shortened[:destination] + [moved_txn] + shortened[destination:]
                if candidate == best_schedule:
                    continue
                cost = sequence_cost(candidate)
                if cost < improved_cost:
                    improved_cost, improved_schedule = cost, candidate

        for left in range(n):
            for right in range(left + 1, n):
                candidate = best_schedule[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                cost = sequence_cost(candidate)
                if cost < improved_cost:
                    improved_cost, improved_schedule = cost, candidate

        if improved_cost >= best_cost:
            break
        best_cost, best_schedule = improved_cost, improved_schedule

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