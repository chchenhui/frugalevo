import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a conflict-aware transaction order with low makespan.

    Prefix beam search supplies several high-quality starting schedules.
    Variable-neighborhood descent then improves them using the real simulator
    objective, including coordinated block moves that can cross conflict-heavy
    regions where a one-transaction move is insufficient.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    cost_cache = {}

    def evaluate(sequence):
        key = tuple(sequence)
        cached = cost_cache.get(key)
        if cached is None:
            cached = workload.get_opt_seq_cost(list(key))
            cost_cache[key] = cached
        return cached

    # Exact optimization is inexpensive for small workloads.
    if num_txns <= 7:
        import itertools

        best_cost = float("inf")
        best_schedule = []
        for sequence in itertools.permutations(range(num_txns)):
            cost = evaluate(sequence)
            if cost < best_cost:
                best_cost = cost
                best_schedule = list(sequence)
        return best_cost, best_schedule

    all_txns = tuple(range(num_txns))

    # Wider beams preserve alternatives whose early prefix cost is slightly
    # worse but whose final conflict interactions are much better.
    beam_width = max(12, min(72, max(1, num_seqs) * 7))
    beam = [(evaluate([txn]), (txn,)) for txn in all_txns]
    beam.sort(key=lambda entry: (entry[0], entry[1]))
    beam = beam[:beam_width]

    for _depth in range(1, num_txns):
        expanded = []
        for _prefix_cost, prefix in beam:
            used = set(prefix)
            for txn in all_txns:
                if txn not in used:
                    candidate = prefix + (txn,)
                    expanded.append((evaluate(candidate), candidate))

        expanded.sort(key=lambda entry: (entry[0], entry[1]))
        beam = expanded[:beam_width]

    def search_positions(current, current_cost):
        """Return conflict-informed source and destination positions."""
        if num_txns <= 30:
            positions = list(range(num_txns))
            return positions, positions

        stride = max(1, num_txns // 14)
        anchors = list(range(0, num_txns, stride))
        if anchors[-1] != num_txns - 1:
            anchors.append(num_txns - 1)

        impacts = []
        for position in range(num_txns):
            reduced = current[:position] + current[position + 1:]
            impacts.append((current_cost - evaluate(reduced), position))

        impacts.sort(key=lambda entry: (-entry[0], entry[1]))
        impact_count = min(num_txns, max(8, num_txns // 12))
        important = [position for _impact, position in impacts[:impact_count]]

        sources = set(anchors)
        destinations = set(anchors)
        for position in important:
            sources.add(position)
            destinations.add(position)
            if position > 0:
                destinations.add(position - 1)
            if position + 1 < num_txns:
                destinations.add(position + 1)

        return sorted(sources), sorted(destinations)

    def improve(start_schedule, start_cost, max_rounds):
        """
        Best-improvement variable-neighborhood descent.

        Each round evaluates several move families.  The best strict
        improvement is chosen, rather than the first improvement, so a move
        cannot hide a stronger coordinated conflict reduction later in the
        same neighborhood.
        """
        current = list(start_schedule)
        current_cost = start_cost

        for _round in range(max_rounds):
            best_cost = current_cost
            best_schedule = current
            source_positions, destination_positions = search_positions(
                current, current_cost
            )

            # Single transaction relocation.
            for source in source_positions:
                moved = current[source]
                stripped = current[:source] + current[source + 1:]

                for destination in destination_positions:
                    insert_at = min(destination, len(stripped))
                    if insert_at == source:
                        continue
                    candidate = (
                        stripped[:insert_at]
                        + [moved]
                        + stripped[insert_at:]
                    )
                    candidate_cost = evaluate(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_schedule = candidate

            # Relocate short contiguous groups while retaining their internal
            # transaction order.  Such groups often represent a compatible
            # chain of operations that should cross a bottleneck together.
            for block_length in range(2, min(4, num_txns) + 1):
                if num_txns <= 30:
                    starts = range(num_txns - block_length + 1)
                    destinations = range(num_txns - block_length + 1)
                else:
                    starts = [
                        pos for pos in source_positions
                        if pos + block_length <= num_txns
                    ]
                    destinations = [
                        pos for pos in destination_positions
                        if pos <= num_txns - block_length
                    ]

                for start in starts:
                    block = current[start:start + block_length]
                    stripped = current[:start] + current[start + block_length:]

                    for destination in destinations:
                        if destination == start:
                            continue
                        insert_at = min(destination, len(stripped))
                        candidate = (
                            stripped[:insert_at]
                            + block
                            + stripped[insert_at:]
                        )
                        candidate_cost = evaluate(candidate)
                        if candidate_cost < best_cost:
                            best_cost = candidate_cost
                            best_schedule = candidate

            # Arbitrary swaps capture inversions which relocation alone may
            # not expose as an improving move.
            for left_index, left in enumerate(source_positions):
                for right in source_positions[left_index + 1:]:
                    candidate = current.copy()
                    candidate[left], candidate[right] = (
                        candidate[right],
                        candidate[left],
                    )
                    candidate_cost = evaluate(candidate)
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_schedule = candidate

            # Exchange two disjoint short blocks.  Unlike relocating one
            # block, this simultaneously moves both sides of a conflict
            # frontier and can escape insertion-local optima.
            if num_txns <= 30:
                exchange_starts = list(range(num_txns))
            else:
                exchange_starts = source_positions

            for first_length in (1, 2):
                for second_length in (1, 2):
                    for first_start in exchange_starts:
                        first_end = first_start + first_length
                        if first_end > num_txns:
                            continue

                        for second_start in exchange_starts:
                            if second_start <= first_start:
                                continue
                            second_end = second_start + second_length
                            if second_end > num_txns or second_start < first_end:
                                continue

                            # A 1-by-1 block exchange is already covered by
                            # swaps, so avoid duplicate simulator calls.
                            if first_length == 1 and second_length == 1:
                                continue

                            first_block = current[first_start:first_end]
                            second_block = current[second_start:second_end]
                            candidate = (
                                current[:first_start]
                                + second_block
                                + current[first_end:second_start]
                                + first_block
                                + current[second_end:]
                            )
                            candidate_cost = evaluate(candidate)
                            if candidate_cost < best_cost:
                                best_cost = candidate_cost
                                best_schedule = candidate

            if best_cost >= current_cost:
                break

            current = best_schedule
            current_cost = best_cost

        return current_cost, current

    best_cost, best_tuple = beam[0]
    best_schedule = list(best_tuple)

    # Improve multiple distinct final beam paths.
    seed_count = min(len(beam), max(4, min(12, num_seqs + 2)))
    improved_seeds = []
    for seed_cost, seed_tuple in beam[:seed_count]:
        cost, schedule = improve(
            list(seed_tuple),
            seed_cost,
            max(3, min(num_txns, 10)),
        )
        improved_seeds.append((cost, schedule))
        if cost < best_cost:
            best_cost = cost
            best_schedule = schedule

    # Deterministic iterated local search.  A perturbation can enter a
    # different basin while preserving most of the good conflict ordering.
    perturb_bases = sorted(improved_seeds, key=lambda entry: entry[0])[:3]
    for base_cost, base_schedule in perturb_bases:
        source_positions, _destination_positions = search_positions(
            base_schedule, base_cost
        )

        candidate_positions = list(source_positions)
        if num_txns <= 30:
            candidate_positions = sorted(
                set(candidate_positions).union(
                    {0, num_txns - 1, num_txns // 3, (2 * num_txns) // 3}
                )
            )

        perturbations = []
        for index in range(min(4, len(candidate_positions))):
            left = candidate_positions[index]
            right = candidate_positions[-1 - index]
            if left == right:
                continue

            swapped = base_schedule.copy()
            swapped[left], swapped[right] = swapped[right], swapped[left]
            perturbations.append(swapped)

            # Also create a long insertion perturbation, which retains the
            # relative order of the transactions crossed by the moved item.
            moved = base_schedule[left]
            stripped = base_schedule[:left] + base_schedule[left + 1:]
            insert_at = min(right, len(stripped))
            perturbations.append(
                stripped[:insert_at] + [moved] + stripped[insert_at:]
            )

        for perturbed in perturbations[:8]:
            perturbed_cost = evaluate(perturbed)
            cost, schedule = improve(
                perturbed,
                perturbed_cost,
                max(3, min(num_txns, 8)),
            )
            if cost < best_cost:
                best_cost = cost
                best_schedule = schedule

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