import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Search transaction permutations using the simulator makespan directly."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def evaluate(sequence):
        key = tuple(sequence)
        value = cache.get(key)
        if value is None:
            value = workload.get_opt_seq_cost(list(key))
            cache[key] = value
        return value

    if n <= 7:
        import itertools
        best_cost = float("inf")
        best = None
        for order in itertools.permutations(range(n)):
            cost = evaluate(order)
            if cost < best_cost:
                best_cost, best = cost, list(order)
        return best_cost, best

    all_txns = tuple(range(n))
    beam_width = max(8, min(48, max(1, num_seqs) * 5))
    beam = [(evaluate((txn,)), (txn,)) for txn in all_txns]
    beam.sort(key=lambda entry: (entry[0], entry[1]))
    beam = beam[:beam_width]

    for _ in range(1, n):
        expanded = []
        for _cost, prefix in beam:
            used = set(prefix)
            for txn in all_txns:
                if txn not in used:
                    candidate = prefix + (txn,)
                    expanded.append((evaluate(candidate), candidate))
        expanded.sort(key=lambda entry: (entry[0], entry[1]))
        beam = expanded[:beam_width]

    best_cost, best_tuple = beam[0]
    best_schedule = list(best_tuple)

    seed_entries = list(beam[:min(len(beam), max(3, min(10, num_seqs)))])
    seed_keys = {order for _cost, order in seed_entries}

    def add_seed(schedule):
        order = tuple(schedule)
        if order not in seed_keys:
            seed_keys.add(order)
            seed_entries.append((evaluate(order), order))

    # Greedy insertion complements append-only beam construction.  At every
    # step it selects both a remaining transaction and its best insertion
    # point using the true partial conflict makespan.
    greedy_starts = [order[0] for _cost, order in beam[:min(4, len(beam))]]
    for start in greedy_starts:
        current = [start]
        remaining = set(all_txns)
        remaining.remove(start)
        while remaining:
            if n <= 30:
                slots = range(len(current) + 1)
            else:
                step = max(1, len(current) // 10)
                slots = sorted(set(
                    list(range(0, len(current) + 1, step)) +
                    [len(current)]
                ))

            choice_cost = float("inf")
            choice = None
            for txn in sorted(remaining):
                for slot in slots:
                    candidate = current[:slot] + [txn] + current[slot:]
                    cost = evaluate(candidate)
                    if cost < choice_cost:
                        choice_cost = cost
                        choice = candidate
            current = choice
            remaining.remove(current[[i for i, x in enumerate(current)
                                      if x not in current[:i]][-1]]
                             if False else next(
                                 txn for txn in all_txns if txn not in
                                 set(current[:-1]) or txn == current[-1]
                             ))
            # The expression above is intentionally avoided below by deriving
            # the selected item through set difference, which also handles any
            # insertion position.
            previous = set(current)
            # Reconstruct remaining safely from the permutation prefix.
            remaining = set(all_txns) - previous
        add_seed(current)

    # Deterministic large moves supply escape points for strict local descent.
    base = list(best_tuple)
    width = min(n, max(2, n // 3))
    for start in sorted({0, max(0, (n - width) // 2), n - width}):
        candidate = base.copy()
        candidate[start:start + width] = reversed(candidate[start:start + width])
        add_seed(candidate)

    block_size = min(max(2, n // 5), n - 1)
    moved = base.copy()
    source = max(0, (n - block_size) // 3)
    target = min(n - block_size, (2 * n) // 3)
    block = moved[source:source + block_size]
    del moved[source:source + block_size]
    moved[min(target, len(moved)):min(target, len(moved))] = block
    add_seed(moved)

    swapped = base.copy()
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    add_seed(swapped)

    def positions_for(current, current_cost):
        if n <= 30:
            positions = list(range(n))
            return positions, positions

        stride = max(1, n // 12)
        anchors = list(range(0, n, stride))
        if anchors[-1] != n - 1:
            anchors.append(n - 1)

        impacts = []
        for pos in range(n):
            reduced = current[:pos] + current[pos + 1:]
            impacts.append((current_cost - evaluate(reduced), pos))
        impacts.sort(key=lambda item: (-item[0], item[1]))

        important = [pos for _impact, pos in impacts[:max(8, n // 12)]]
        sources = sorted(set(anchors).union(important))
        destinations = set(anchors)
        for pos in important:
            destinations.add(pos)
            if pos:
                destinations.add(pos - 1)
            if pos + 1 < n:
                destinations.add(pos + 1)
        return sources, sorted(destinations)

    for seed_cost, seed_tuple in seed_entries:
        current = list(seed_tuple)
        current_cost = seed_cost

        for _round in range(max(2, n)):
            round_cost = current_cost
            round_best = current
            sources, destinations = positions_for(current, current_cost)

            for source in sources:
                stripped = current[:source] + current[source + 1:]
                txn = current[source]
                for destination in destinations:
                    insert_at = min(destination, len(stripped))
                    if insert_at == source:
                        continue
                    candidate = stripped[:insert_at] + [txn] + stripped[insert_at:]
                    cost = evaluate(candidate)
                    if cost < round_cost:
                        round_cost, round_best = cost, candidate

            swap_positions = list(range(n)) if n <= 30 else list(sources)
            for left_index, left in enumerate(swap_positions):
                for right in swap_positions[left_index + 1:]:
                    candidate = current.copy()
                    candidate[left], candidate[right] = candidate[right], candidate[left]
                    cost = evaluate(candidate)
                    if cost < round_cost:
                        round_cost, round_best = cost, candidate

                    if right - left > 1:
                        candidate = (current[:left] +
                                     list(reversed(current[left:right + 1])) +
                                     current[right + 1:])
                        cost = evaluate(candidate)
                        if cost < round_cost:
                            round_cost, round_best = cost, candidate

            for length in range(2, min(4, n - 1) + 1):
                starts = (range(n - length + 1) if n <= 30 else
                          [p for p in sources if p <= n - length])
                destinations2 = (range(n - length + 1) if n <= 30 else
                                 sorted(set(min(p, n - length)
                                            for p in destinations).union(
                                                {0, n - length})))
                for start in starts:
                    block = current[start:start + length]
                    stripped = current[:start] + current[start + length:]
                    for destination in destinations2:
                        insert_at = min(destination, len(stripped))
                        if insert_at == start:
                            continue
                        candidate = stripped[:insert_at] + block + stripped[insert_at:]
                        cost = evaluate(candidate)
                        if cost < round_cost:
                            round_cost, round_best = cost, candidate

            if round_cost >= current_cost:
                break
            current_cost, current = round_cost, round_best

        if current_cost < best_cost:
            best_cost, best_schedule = current_cost, current

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