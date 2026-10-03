import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using exact search for very small
    workloads, diversified beam-greedy construction, and exact-cost VND
    refinement for larger workloads.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    cost_cache = {}

    def schedule_cost(sequence):
        key = tuple(sequence)
        value = cost_cache.get(key)
        if value is None:
            value = workload.get_opt_seq_cost(sequence)
            cost_cache[key] = value
        return value

    # Small workloads are cheap enough to solve exactly.  This also avoids
    # heuristic variance on workloads where every possible ordering matters.
    if num_txns <= 8:
        import itertools

        best_cost = float("inf")
        best_order = None
        for order in itertools.permutations(range(num_txns)):
            candidate_cost = schedule_cost(order)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_order = list(order)
        return best_cost, best_order

    def candidate_pool(remaining, width):
        """Use all choices when affordable, otherwise unbiased sampling."""
        if len(remaining) <= width:
            return remaining[:]
        return random.sample(remaining, width)

    def greedy_order(start_txn):
        order = [start_txn]
        remaining = list(range(num_txns))
        remaining.remove(start_txn)

        while remaining:
            width = len(remaining) if len(remaining) <= 30 else min(
                len(remaining), max(16, 3 * max(1, num_seqs))
            )
            candidates = candidate_pool(remaining, width)

            best_prefix_cost = float("inf")
            tied = []
            for txn in candidates:
                candidate_cost = schedule_cost(order + [txn])
                if candidate_cost < best_prefix_cost:
                    best_prefix_cost = candidate_cost
                    tied = [txn]
                elif candidate_cost == best_prefix_cost:
                    tied.append(txn)

            chosen = random.choice(tied)
            order.append(chosen)
            remaining.remove(chosen)

        return schedule_cost(order), order

    # A beam preserves several good prefixes.  The old sampled greedy method
    # supplies independent paths; the beam prevents one early choice from
    # irrevocably determining every resulting schedule.
    beam_width = min(max(4, num_seqs), 8)
    initial_txns = list(range(num_txns))
    random.shuffle(initial_txns)
    beam = [(schedule_cost([txn]), [txn], set(range(num_txns)) - {txn})
            for txn in initial_txns[:beam_width]]

    for _ in range(1, num_txns):
        expanded = []
        seen = set()

        for _, order, remaining_set in beam:
            remaining = list(remaining_set)
            width = len(remaining) if len(remaining) <= 26 else min(
                len(remaining), max(14, 2 * beam_width + 8)
            )

            for txn in candidate_pool(remaining, width):
                new_order = order + [txn]
                order_key = tuple(new_order)
                if order_key in seen:
                    continue
                seen.add(order_key)
                expanded.append((
                    schedule_cost(new_order),
                    new_order,
                    remaining_set - {txn},
                ))

        expanded.sort(key=lambda item: item[0])

        # Keep a few tied alternatives in randomized order, avoiding a fixed
        # tie-breaking bias when prefix costs are identical.
        next_beam = []
        index = 0
        while index < len(expanded) and len(next_beam) < beam_width:
            end = index + 1
            while end < len(expanded) and expanded[end][0] == expanded[index][0]:
                end += 1
            tied_group = expanded[index:end]
            random.shuffle(tied_group)
            for state in tied_group:
                if len(next_beam) >= beam_width:
                    break
                next_beam.append(state)
            index = end
        beam = next_beam

    seed_orders = [(cost, order) for cost, order, _ in beam]

    starts = list(range(num_txns))
    random.shuffle(starts)
    greedy_attempts = max(2, num_seqs)
    for index in range(greedy_attempts):
        seed_orders.append(greedy_order(starts[index % num_txns]))

    def refine(order, current_cost):
        """
        Best-improving variable-neighborhood descent.  Insertions repair
        displaced transactions, exchanges repair distant conflict inversions,
        and reversals repair incorrectly oriented conflict-heavy regions.
        """
        rounds = 5 if num_txns <= 32 else 4

        for _ in range(rounds):
            best_cost = current_cost
            best_order = None

            if num_txns <= 30:
                insertion_moves = [
                    (source, destination)
                    for source in range(num_txns)
                    for destination in range(num_txns)
                    if source != destination
                ]
                exchange_moves = [
                    (left, right)
                    for left in range(num_txns - 1)
                    for right in range(left + 1, num_txns)
                ]
                reverse_moves = exchange_moves
            else:
                samples = 7 * num_txns
                insertion_moves = [
                    (random.randrange(num_txns), random.randrange(num_txns))
                    for _ in range(samples)
                ]
                exchange_moves = [
                    tuple(sorted(random.sample(range(num_txns), 2)))
                    for _ in range(samples)
                ]
                reverse_moves = [
                    tuple(sorted(random.sample(range(num_txns), 2)))
                    for _ in range(samples)
                ]

            for source, destination in insertion_moves:
                if source == destination:
                    continue
                candidate = order[:]
                txn = candidate.pop(source)
                candidate.insert(destination, txn)
                candidate_cost = schedule_cost(candidate)
                if candidate_cost < best_cost:
                    best_cost, best_order = candidate_cost, candidate

            for left, right in exchange_moves:
                candidate = order[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_cost = schedule_cost(candidate)
                if candidate_cost < best_cost:
                    best_cost, best_order = candidate_cost, candidate

            for left, right in reverse_moves:
                if right - left < 2:
                    continue
                candidate = order[:]
                candidate[left:right + 1] = reversed(candidate[left:right + 1])
                candidate_cost = schedule_cost(candidate)
                if candidate_cost < best_cost:
                    best_cost, best_order = candidate_cost, candidate

            if best_order is None:
                break
            order, current_cost = best_order, best_cost

        return current_cost, order

    # Deduplicate seeds before refinement.  Keep a broader elite than the
    # previous implementation because beam and independent greedy paths often
    # expose different conflict structures.
    unique_seeds = {}
    for cost, order in seed_orders:
        key = tuple(order)
        if key not in unique_seeds or cost < unique_seeds[key][0]:
            unique_seeds[key] = (cost, order)

    ranked_seeds = sorted(unique_seeds.values(), key=lambda item: item[0])
    elite_count = min(max(3, min(num_seqs, 6)), len(ranked_seeds))

    best_cost = float("inf")
    best_order = None
    for seed_cost, seed_order in ranked_seeds[:elite_count]:
        refined_cost, refined_order = refine(seed_order[:], seed_cost)
        if refined_cost < best_cost:
            best_cost, best_order = refined_cost, refined_order

    return best_cost, best_order

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