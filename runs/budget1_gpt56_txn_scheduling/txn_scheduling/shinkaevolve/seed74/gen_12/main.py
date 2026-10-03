import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order.

    The scheduler uses exact permutation search for very small workloads and a
    beam-search plus insertion-local-search pipeline for larger workloads.
    Every ranking decision is made with workload.get_opt_seq_cost(), so search
    is based on the actual conflict-aware makespan rather than a proxy metric.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    # A schedule prefix is immutable in the cache.  Prefix caching is useful
    # during beam expansion, while complete-order caching helps local search.
    cost_cache = {}

    def evaluate(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(list(key))
        return cost_cache[key]

    # For small instances, return a provably optimal ordering.
    if num_txns <= 7:
        import itertools

        best_cost = float("inf")
        best_schedule = None
        for sequence in itertools.permutations(range(num_txns)):
            cost = evaluate(sequence)
            if cost < best_cost:
                best_cost = cost
                best_schedule = list(sequence)
        return best_cost, best_schedule

    # Keep several competing prefixes.  Unlike the original algorithm, this
    # never randomly discards a potentially useful next transaction.
    beam_width = max(8, min(48, max(1, num_seqs) * 5))
    beam = [(evaluate([txn]), (txn,)) for txn in range(num_txns)]
    beam.sort(key=lambda entry: (entry[0], entry[1]))
    beam = beam[:beam_width]

    all_txns = tuple(range(num_txns))

    for _depth in range(1, num_txns):
        expanded = []

        for _prefix_cost, prefix in beam:
            used = set(prefix)
            for txn in all_txns:
                if txn in used:
                    continue
                candidate = prefix + (txn,)
                expanded.append((evaluate(candidate), candidate))

        # Different parent prefixes can never produce the same ordering at the
        # same depth, so sorting and truncating is sufficient here.
        expanded.sort(key=lambda entry: (entry[0], entry[1]))
        beam = expanded[:beam_width]

    best_cost, best_tuple = beam[0]
    best_schedule = list(best_tuple)

    # Improve several independent beam results.  An insertion move is more
    # expressive than an adjacent swap: it can move a transaction across an
    # entire conflicting region in a single optimization step.
    local_seed_count = min(len(beam), max(3, min(10, num_seqs)))
    for seed_cost, seed_tuple in beam[:local_seed_count]:
        current = list(seed_tuple)
        current_cost = seed_cost
        improved = True
        rounds = 0

        while improved and rounds < max(2, num_txns):
            improved = False
            rounds += 1
            round_best_cost = current_cost
            round_best_schedule = current

            # Full insertion neighborhoods are affordable for the workloads
            # this scheduler targets.  For very large workloads, evaluate a
            # structured subset that still includes long-distance moves.
            if num_txns <= 30:
                source_positions = range(num_txns)
                destination_positions = range(num_txns)
            else:
                # Evenly spaced positions retain coverage for long-distance
                # moves, but conflict bottlenecks are rarely evenly spaced.
                # Measure each transaction's actual contribution by removing
                # it temporarily; this uses the same makespan objective as the
                # search rather than relying on operation-count heuristics.
                stride = max(1, num_txns // 12)
                anchor_positions = list(range(0, num_txns, stride))
                if anchor_positions[-1] != num_txns - 1:
                    anchor_positions.append(num_txns - 1)

                removal_impacts = []
                for position in range(num_txns):
                    reduced = current[:position] + current[position + 1:]
                    removal_impacts.append(
                        (current_cost - evaluate(reduced), position)
                    )

                impact_budget = min(num_txns, max(6, num_txns // 16))
                removal_impacts.sort(key=lambda entry: (-entry[0], entry[1]))
                impact_positions = [
                    position
                    for _impact, position in removal_impacts[:impact_budget]
                ]

                # High-impact transactions are useful sources, and positions
                # on either side of them are useful destinations because an
                # improving move commonly needs to cross their conflict region.
                source_positions = sorted(
                    set(anchor_positions).union(impact_positions)
                )
                destination_set = set(anchor_positions)
                for position in impact_positions:
                    destination_set.add(position)
                    if position > 0:
                        destination_set.add(position - 1)
                    if position < num_txns - 1:
                        destination_set.add(position + 1)
                destination_positions = sorted(destination_set)

            for source in source_positions:
                stripped = current[:source] + current[source + 1:]
                moved_txn = current[source]

                for destination in destination_positions:
                    # destination is interpreted in the original schedule's
                    # coordinate system, then clamped after removal.
                    insert_at = min(destination, len(stripped))
                    if insert_at == source:
                        continue

                    candidate = (
                        stripped[:insert_at]
                        + [moved_txn]
                        + stripped[insert_at:]
                    )
                    candidate_cost = evaluate(candidate)

                    if candidate_cost < round_best_cost:
                        round_best_cost = candidate_cost
                        round_best_schedule = candidate

            # A single insertion can be unable to cross a conflict bottleneck:
            # two transactions may need to retain their relative order while
            # moving together.  Relocating short contiguous blocks supplies
            # that coordinated move without the cost of arbitrary permutations.
            for block_length in range(2, min(4, num_txns) + 1):
                if num_txns <= 30:
                    block_starts = range(num_txns - block_length + 1)
                    block_destinations = range(num_txns - block_length + 1)
                else:
                    block_starts = [
                        start
                        for start in source_positions
                        if start + block_length <= num_txns
                    ]
                    block_destinations = [
                        destination
                        for destination in destination_positions
                        if destination <= num_txns - block_length
                    ]

                for start in block_starts:
                    block = current[start:start + block_length]
                    stripped = (
                        current[:start] + current[start + block_length:]
                    )

                    # Destinations are positions in the schedule after the
                    # block has been removed.  Reinserting at ``start`` is
                    # the unchanged schedule and is skipped.
                    for destination in block_destinations:
                        if destination == start:
                            continue
                        candidate = (
                            stripped[:destination]
                            + block
                            + stripped[destination:]
                        )
                        candidate_cost = evaluate(candidate)

                        if candidate_cost < round_best_cost:
                            round_best_cost = candidate_cost
                            round_best_schedule = candidate

            # An insertion-only optimum is not necessarily an ordering
            # optimum.  In particular, exchanging two transactions can
            # simultaneously remove two opposing conflict delays even when
            # neither transaction has an individually improving insertion.
            swap_positions = list(source_positions)
            for left_index, left in enumerate(swap_positions):
                for right in swap_positions[left_index + 1:]:
                    candidate = current.copy()
                    candidate[left], candidate[right] = (
                        candidate[right],
                        candidate[left],
                    )
                    candidate_cost = evaluate(candidate)

                    if candidate_cost < round_best_cost:
                        round_best_cost = candidate_cost
                        round_best_schedule = candidate

            if round_best_cost < current_cost:
                current = round_best_schedule
                current_cost = round_best_cost
                improved = True

        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = current

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