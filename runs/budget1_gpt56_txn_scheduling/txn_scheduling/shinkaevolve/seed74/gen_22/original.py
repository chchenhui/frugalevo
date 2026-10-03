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

    # Descend with several complementary permutation neighborhoods.  All
    # accept/reject decisions use the simulator objective, not transaction
    # length or a conflict-count approximation.
    if num_txns <= 30:
        search_positions = list(range(num_txns))
    else:
        stride = max(1, num_txns // 12)
        search_positions = list(range(0, num_txns, stride))
        if search_positions[-1] != num_txns - 1:
            search_positions.append(num_txns - 1)

    def descend(start_schedule, start_cost):
        current = list(start_schedule)
        current_cost = start_cost
        rounds = 0

        while rounds < max(2, num_txns):
            rounds += 1
            round_best_cost = current_cost
            round_best_schedule = current

            for source in search_positions:
                stripped = current[:source] + current[source + 1:]
                moved_txn = current[source]
                for destination in search_positions:
                    insert_at = min(destination, len(stripped))
                    if insert_at == source:
                        continue
                    candidate = (
                        stripped[:insert_at] + [moved_txn] + stripped[insert_at:]
                    )
                    candidate_cost = evaluate(candidate)
                    if candidate_cost < round_best_cost:
                        round_best_cost = candidate_cost
                        round_best_schedule = candidate

            for left_offset, left in enumerate(search_positions):
                for right in search_positions[left_offset + 1:]:
                    candidate = current.copy()
                    candidate[left], candidate[right] = candidate[right], candidate[left]
                    candidate_cost = evaluate(candidate)
                    if candidate_cost < round_best_cost:
                        round_best_cost = candidate_cost
                        round_best_schedule = candidate

            for left_offset, left in enumerate(search_positions):
                for right in search_positions[left_offset + 2:]:
                    candidate = (
                        current[:left]
                        + current[left:right + 1][::-1]
                        + current[right + 1:]
                    )
                    candidate_cost = evaluate(candidate)
                    if candidate_cost < round_best_cost:
                        round_best_cost = candidate_cost
                        round_best_schedule = candidate

            if round_best_cost >= current_cost:
                break
            current = round_best_schedule
            current_cost = round_best_cost

        return current_cost, current

    # First optimize distinct elite beam schedules.  Retaining more than one
    # basin matters because equal prefix costs can lead to very different
    # complete conflict patterns.
    local_seed_count = min(len(beam), max(4, min(10, num_seqs)))
    elite = []
    seen_local_orders = set()
    for seed_cost, seed_tuple in beam[:local_seed_count]:
        local_cost, local_schedule = descend(list(seed_tuple), seed_cost)
        local_key = tuple(local_schedule)
        if local_key not in seen_local_orders:
            seen_local_orders.add(local_key)
            elite.append((local_cost, local_schedule))
        if local_cost < best_cost:
            best_cost = local_cost
            best_schedule = local_schedule

    # A local optimum may require several relative-order changes before any
    # improvement becomes visible.  Deterministic kicks create such changes,
    # and each perturbed permutation is immediately reoptimized by the same
    # strict objective-based descent.
    elite.sort(key=lambda entry: (entry[0], tuple(entry[1])))
    kick_elites = elite[:min(4, len(elite))]
    for _elite_cost, base in kick_elites:
        n = len(base)
        anchors = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1]))

        # Medium reversals alter a concentrated conflict region without
        # destroying the remainder of an already good schedule.
        for left in anchors[:-1]:
            right = min(n - 1, left + max(2, n // 3))
            if right <= left:
                continue
            kicked = base[:left] + base[left:right + 1][::-1] + base[right + 1:]
            kicked_cost, kicked_schedule = descend(kicked, evaluate(kicked))
            if kicked_cost < best_cost:
                best_cost = kicked_cost
                best_schedule = kicked_schedule

        # Relocate short blocks across distant anchors.  This is a coordinated
        # version of insertion and can cross an insertion-local optimum.
        block_size = max(2, min(4, n // 5))
        for source in anchors[:-1]:
            end = min(n, source + block_size)
            block = base[source:end]
            remainder = base[:source] + base[end:]
            destination = n - 1 - source
            insert_at = min(destination, len(remainder))
            kicked = remainder[:insert_at] + block + remainder[insert_at:]
            kicked_cost, kicked_schedule = descend(kicked, evaluate(kicked))
            if kicked_cost < best_cost:
                best_cost = kicked_cost
                best_schedule = kicked_schedule

        # A distant exchange supplies a different perturbation shape from
        # reversal and relocation while always remaining a valid permutation.
        for left, right in zip(anchors, reversed(anchors)):
            if left >= right:
                continue
            kicked = base.copy()
            kicked[left], kicked[right] = kicked[right], kicked[left]
            kicked_cost, kicked_schedule = descend(kicked, evaluate(kicked))
            if kicked_cost < best_cost:
                best_cost = kicked_cost
                best_schedule = kicked_schedule

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