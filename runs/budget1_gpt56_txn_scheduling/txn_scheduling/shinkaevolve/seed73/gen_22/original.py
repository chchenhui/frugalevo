import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan serial transaction order using simulated-cost
    construction, local ordering search, and ruin-and-recreate improvement.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    cost_cache = {}

    def schedule_cost(order):
        key = tuple(order)
        cached = cost_cache.get(key)
        if cached is None:
            cached = workload.get_opt_seq_cost(order)
            cost_cache[key] = cached
        return cached

    # Exhaustive search is cheap enough on genuinely small instances and
    # guarantees that these cases receive an optimal schedule.
    if num_txns <= 8:
        import itertools

        best_cost = float("inf")
        best_order = None
        for order in itertools.permutations(range(num_txns)):
            cost = schedule_cost(order)
            if cost < best_cost:
                best_cost = cost
                best_order = list(order)
        return best_cost, best_order

    def candidate_subset(values, limit):
        if len(values) <= limit:
            return values[:]
        return random.sample(values, limit)

    singleton_scores = [
        (schedule_cost([txn]), txn) for txn in range(num_txns)
    ]
    singleton_scores.sort()
    singleton_rank = [txn for _, txn in singleton_scores]

    def construct(start, randomize=False, prefix=None):
        """
        Greedily append the transaction producing the lowest measured prefix
        cost.  Randomized runs select only from a small near-best candidate
        list, preserving quality while exploring alternate conflict orders.
        """
        if prefix is None:
            order = [start]
        else:
            order = prefix[:]

        used = set(order)
        remaining = [txn for txn in range(num_txns) if txn not in used]

        while remaining:
            if len(remaining) <= 38:
                candidates = remaining[:]
            else:
                candidates = candidate_subset(
                    remaining,
                    min(len(remaining), max(28, 3 * max(1, num_seqs))),
                )

            scored = [(schedule_cost(order + [txn]), txn) for txn in candidates]
            scored.sort()

            if randomize and len(scored) > 1:
                best = scored[0][0]
                # Relative slack makes this robust for workloads of different
                # makespan scales, while the absolute term handles small costs.
                slack = max(1, int(best * 0.018))
                restricted = [
                    txn for cost, txn in scored
                    if cost <= best + slack
                ]
                txn = random.choice(restricted[:min(6, len(restricted))])
            else:
                txn = scored[0][1]

            order.append(txn)
            remaining.remove(txn)

        return schedule_cost(order), order

    # Prefix beam search supplies schedules with different early dependency
    # orientations.  Starts include good singleton starts and deliberately
    # weaker/random starts, since singleton cost alone is not globally enough.
    beam_width = min(max(8, num_seqs + 2), 14)
    chosen_starts = singleton_rank[:max(1, beam_width // 2)]
    leftovers = [txn for txn in range(num_txns) if txn not in chosen_starts]
    random.shuffle(leftovers)
    chosen_starts.extend(leftovers[:beam_width - len(chosen_starts)])

    beam = [
        (schedule_cost([txn]), [txn], set(range(num_txns)) - {txn})
        for txn in chosen_starts
    ]

    for depth in range(1, num_txns):
        expanded = []
        seen = set()

        for _, order, remaining_set in beam:
            remaining = list(remaining_set)
            if len(remaining) <= 32:
                candidates = remaining
            else:
                candidates = candidate_subset(
                    remaining, min(len(remaining), 26 + beam_width)
                )

            for txn in candidates:
                new_order = order + [txn]
                key = tuple(new_order)
                if key in seen:
                    continue
                seen.add(key)
                expanded.append((
                    schedule_cost(new_order),
                    new_order,
                    remaining_set - {txn},
                ))

        expanded.sort(key=lambda entry: entry[0])

        # Keep the leading states and one occasional state outside the strict
        # prefix elite region to avoid collapsing onto equivalent schedules.
        beam = expanded[:beam_width]
        if len(expanded) > beam_width * 2 and beam_width >= 6:
            replacement = random.choice(expanded[beam_width:beam_width * 3])
            beam[-1] = replacement

    seeds = [(cost, order) for cost, order, _ in beam]

    # Independent constructions complement the beam.  Deterministic starts
    # cover favorable singleton states; randomized restricted-choice runs
    # expose different transaction conflict orientations.
    attempts = max(10, 2 * num_seqs + 4)
    starts = singleton_rank[:]
    random.shuffle(starts)
    for attempt in range(attempts):
        if attempt < len(singleton_rank) // 3:
            start = singleton_rank[attempt]
            diverse = False
        else:
            start = starts[attempt % num_txns]
            diverse = True
        seeds.append(construct(start, randomize=diverse))

    def local_descent(order, current_cost, rounds=4):
        """
        Best-improvement descent over multiple neighborhoods.  Insertion is
        particularly useful for transaction scheduling because it can move one
        conflict-heavy transaction across an entire conflicting region.
        """
        for _ in range(rounds):
            best_cost = current_cost
            best_order = None

            def consider(candidate):
                nonlocal best_cost, best_order
                value = schedule_cost(candidate)
                if value < best_cost:
                    best_cost = value
                    best_order = candidate

            # Adjacent exchanges cheaply correct directly inverted conflicts.
            for left in range(num_txns - 1):
                candidate = order[:]
                candidate[left], candidate[left + 1] = (
                    candidate[left + 1],
                    candidate[left],
                )
                consider(candidate)

            if num_txns <= 34:
                insertion_moves = [
                    (source, destination)
                    for source in range(num_txns)
                    for destination in range(num_txns)
                    if source != destination
                ]
                pair_moves = [
                    (left, right)
                    for left in range(num_txns - 1)
                    for right in range(left + 1, num_txns)
                ]
            else:
                count = 12 * num_txns
                insertion_moves = [
                    (random.randrange(num_txns), random.randrange(num_txns))
                    for _ in range(count)
                ]
                pair_moves = [
                    tuple(sorted(random.sample(range(num_txns), 2)))
                    for _ in range(count)
                ]

            for source, destination in insertion_moves:
                if source == destination:
                    continue
                candidate = order[:]
                moved = candidate.pop(source)
                candidate.insert(destination, moved)
                consider(candidate)

            for left, right in pair_moves:
                candidate = order[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                consider(candidate)

            # Reversing a region changes many pairwise conflict orientations
            # together, which simple single-transaction moves can miss.
            reversal_moves = pair_moves if num_txns <= 34 else pair_moves[:len(pair_moves) // 2]
            for left, right in reversal_moves:
                if right - left < 2:
                    continue
                candidate = order[:]
                candidate[left:right + 1] = reversed(candidate[left:right + 1])
                consider(candidate)

            if best_order is None:
                break

            order = best_order
            current_cost = best_cost

        return current_cost, order

    def ruin_and_recreate(order, current_cost, diversity=False):
        """
        Remove several transactions and reinsert them at measured best
        positions.  This is a large-neighborhood move: the retained schedule
        preserves useful dependency orientation while a bad conflict cluster
        can be rebuilt from scratch.
        """
        if num_txns < 7:
            return current_cost, order

        remove_count = max(2, min(
            num_txns // 3,
            3 + num_txns // 10,
        ))

        # Mix contiguous damage (often a bad conflict region) with scattered
        # damage (independent ordering decisions).
        if random.random() < 0.55:
            start = random.randrange(num_txns - remove_count + 1)
            removed = order[start:start + remove_count]
        else:
            positions = sorted(random.sample(range(num_txns), remove_count))
            removed = [order[pos] for pos in positions]

        removed_set = set(removed)
        partial = [txn for txn in order if txn not in removed_set]
        random.shuffle(removed)

        for txn in removed:
            scored = []
            if len(partial) <= 42:
                positions = range(len(partial) + 1)
            else:
                positions = set([0, len(partial)])
                positions.update(random.sample(
                    range(1, len(partial)),
                    min(32, len(partial) - 1),
                ))

            for position in positions:
                candidate = partial[:]
                candidate.insert(position, txn)
                scored.append((schedule_cost(candidate), position))

            scored.sort()
            if diversity and len(scored) > 1:
                threshold = scored[0][0] + max(1, int(scored[0][0] * 0.012))
                options = [
                    position for value, position in scored
                    if value <= threshold
                ]
                position = random.choice(options[:min(4, len(options))])
            else:
                position = scored[0][1]
            partial.insert(position, txn)

        rebuilt_cost = schedule_cost(partial)
        return rebuilt_cost, partial

    unique = {}
    for cost, order in seeds:
        key = tuple(order)
        if key not in unique or cost < unique[key][0]:
            unique[key] = (cost, order)

    ranked = sorted(unique.values(), key=lambda item: item[0])
    elite_count = min(len(ranked), max(5, min(9, num_seqs)))

    best_cost = float("inf")
    best_order = None
    refined = []

    for seed_cost, seed_order in ranked[:elite_count]:
        cost, order = local_descent(seed_order[:], seed_cost)
        refined.append((cost, order))
        if cost < best_cost:
            best_cost, best_order = cost, order

    # Repeated large-neighborhood reconstruction is applied only to the best
    # local optima, making the search deeper without spending calls on weak
    # schedules.
    for base_cost, base_order in sorted(refined, key=lambda item: item[0])[:4]:
        current_cost, current_order = base_cost, base_order[:]
        for iteration in range(3 if num_txns <= 40 else 2):
            rebuilt_cost, rebuilt_order = ruin_and_recreate(
                current_order,
                current_cost,
                diversity=(iteration > 0),
            )
            rebuilt_cost, rebuilt_order = local_descent(
                rebuilt_order,
                rebuilt_cost,
                rounds=2,
            )

            # Keep the best state as the next reconstruction base, but allow
            # one non-improving reconstructed state to be explored once.
            if rebuilt_cost <= current_cost or iteration == 0:
                current_cost, current_order = rebuilt_cost, rebuilt_order

            if rebuilt_cost < best_cost:
                best_cost, best_order = rebuilt_cost, rebuilt_order

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