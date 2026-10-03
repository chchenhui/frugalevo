import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using repeated objective-driven
    construction followed by local permutation improvement.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    # The simulator cost is the authoritative conflict-delay objective.  The
    # same prefixes and neighbours occur often during search, so cache them.
    cost_cache = {}

    def sequence_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    # Exhaustive search is inexpensive for tiny workloads and makes this path
    # genuinely optimal rather than heuristic.
    if num_txns <= 8:
        from itertools import permutations

        best_schedule = None
        best_cost = float("inf")
        for schedule in permutations(range(num_txns)):
            cost = sequence_cost(schedule)
            if cost < best_cost:
                best_cost = cost
                best_schedule = list(schedule)
        return best_cost, best_schedule

    def greedy_construct(start_txn):
        schedule = [start_txn]
        remaining = list(range(num_txns))
        remaining.remove(start_txn)

        while remaining:
            # Evaluating every choice becomes costly on large workloads.  A
            # broad random pool still evaluates the real incremental makespan,
            # unlike the former ten-draw sampling which could repeat choices.
            if len(remaining) <= 24:
                candidates = remaining[:]
            else:
                candidates = random.sample(remaining, 24)

            best_txn = None
            best_cost = float("inf")
            for txn in candidates:
                candidate_cost = sequence_cost(schedule + [txn])
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_txn = txn

            schedule.append(best_txn)
            remaining.remove(best_txn)

        return schedule

    # Different starts expose different conflict chains.  num_seqs now controls
    # the number of independent constructions rather than being ignored.
    starts = list(range(num_txns))
    random.shuffle(starts)
    attempts = max(1, num_seqs)
    best_schedule = None
    best_cost = float("inf")
    for attempt in range(attempts):
        schedule = greedy_construct(starts[attempt % num_txns])
        cost = sequence_cost(schedule)
        if cost < best_cost:
            best_cost = cost
            best_schedule = schedule

    # Improve the best construction with moves that directly reorder conflict
    # chains.  Keep the best move from each pass, which avoids accepting noisy
    # random regressions.
    for _ in range(3):
        improved_schedule = best_schedule
        improved_cost = best_cost

        # Adjacent exchanges are cheap and often remove a local conflict.
        for index in range(num_txns - 1):
            candidate = best_schedule[:]
            candidate[index], candidate[index + 1] = (
                candidate[index + 1],
                candidate[index],
            )
            cost = sequence_cost(candidate)
            if cost < improved_cost:
                improved_cost = cost
                improved_schedule = candidate

        # Long swaps and insertion moves repair conflict chains that cannot be
        # corrected by one adjacent exchange.
        for _ in range(6 * num_txns):
            left, right = random.sample(range(num_txns), 2)

            swapped = best_schedule[:]
            swapped[left], swapped[right] = swapped[right], swapped[left]
            cost = sequence_cost(swapped)
            if cost < improved_cost:
                improved_cost = cost
                improved_schedule = swapped

            inserted = best_schedule[:]
            txn = inserted.pop(left)
            inserted.insert(right, txn)
            cost = sequence_cost(inserted)
            if cost < improved_cost:
                improved_cost = cost
                improved_schedule = inserted

        if improved_cost >= best_cost:
            break
        best_schedule = improved_schedule
        best_cost = improved_cost

    if workload.debug:
        print("best:", best_cost, best_schedule)
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