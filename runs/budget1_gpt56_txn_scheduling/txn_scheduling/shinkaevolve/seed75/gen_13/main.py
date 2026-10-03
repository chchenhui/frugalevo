import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Construct several conflict-aware greedy schedules, then improve the best
    ones with exact-cost relocation, swap, and block-move neighborhoods.
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    cost_cache = {}

    def sequence_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    def greedy_from(start_txn):
        sequence = [start_txn]
        remaining = set(range(n))
        remaining.remove(start_txn)

        while remaining:
            # Exhaustive selection is substantially more reliable for the
            # usual benchmark sizes.  For very large workloads, retain a
            # bounded random candidate set to keep search time practical.
            if len(remaining) <= 32:
                candidates = sorted(remaining)
            else:
                candidates = random.sample(
                    list(remaining), min(16, len(remaining))
                )

            best_txn = None
            best_cost = None
            for txn in candidates:
                candidate_cost = sequence_cost(sequence + [txn])
                if (best_cost is None or candidate_cost < best_cost or
                        (candidate_cost == best_cost and txn < best_txn)):
                    best_cost = candidate_cost
                    best_txn = txn

            sequence.append(best_txn)
            remaining.remove(best_txn)

        return sequence_cost(sequence), sequence

    def improve(sequence, initial_cost):
        current = list(sequence)
        current_cost = initial_cost
        # A best-improvement VND pass makes each accepted move meaningful and
        # avoids being trapped in the first improving move encountered.
        max_passes = 8 if n <= 30 else 3

        for _ in range(max_passes):
            best_sequence = current
            best_cost = current_cost

            # Relocation: move one transaction across a conflicting region.
            if n <= 30:
                relocation_moves = (
                    (source, destination)
                    for source in range(n)
                    for destination in range(n)
                    if source != destination
                )
            else:
                relocation_moves = (
                    (random.randrange(n), random.randrange(n))
                    for _ in range(400)
                )

            for source, destination in relocation_moves:
                if source == destination:
                    continue
                candidate = current[:]
                txn = candidate.pop(source)
                candidate.insert(destination, txn)
                candidate_cost = sequence_cost(candidate)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_sequence = candidate

            # Swaps efficiently correct inversions that relocation may miss.
            if n <= 30:
                swap_moves = (
                    (left, right)
                    for left in range(n - 1)
                    for right in range(left + 1, n)
                )
            else:
                swap_moves = (
                    tuple(sorted(random.sample(range(n), 2)))
                    for _ in range(250)
                )

            for left, right in swap_moves:
                candidate = current[:]
                candidate[left], candidate[right] = (
                    candidate[right], candidate[left]
                )
                candidate_cost = sequence_cost(candidate)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_sequence = candidate

            # Moving a pair together captures interacting writer/reader
            # groups for which neither individual relocation is beneficial.
            if n >= 3:
                if n <= 30:
                    block_moves = (
                        (start, destination)
                        for start in range(n - 1)
                        for destination in range(n - 1)
                    )
                else:
                    block_moves = (
                        (random.randrange(n - 1), random.randrange(n - 1))
                        for _ in range(250)
                    )

                for start, destination in block_moves:
                    block = current[start:start + 2]
                    remainder = current[:start] + current[start + 2:]
                    candidate = (
                        remainder[:destination] + block + remainder[destination:]
                    )
                    candidate_cost = sequence_cost(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_sequence = candidate

            if best_cost >= current_cost:
                break
            current, current_cost = best_sequence, best_cost

        return current_cost, current

    starts_to_use = max(1, min(n, num_seqs))
    if starts_to_use == n:
        starts = list(range(n))
    else:
        # Include transaction zero deterministically and diversify the rest.
        starts = [0] + random.sample(range(1, n), starts_to_use - 1)

    constructed = [greedy_from(start) for start in starts]
    constructed.sort(key=lambda item: item[0])

    # Refining several independently constructed schedules is more robust
    # than spending all local-search effort on a single greedy trajectory.
    refine_count = min(len(constructed), 4 if n <= 30 else 2)
    best_cost, best_schedule = constructed[0]
    for seed_cost, seed_schedule in constructed[:refine_count]:
        improved_cost, improved_schedule = improve(seed_schedule, seed_cost)
        if improved_cost < best_cost:
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