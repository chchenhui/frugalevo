import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction schedule using conflict-aware beam search,
    diverse prefix retention, and variable-neighborhood local optimization.

    Every ranking and refinement decision uses the simulator's exact
    get_opt_seq_cost objective.
    """
    n = workload.num_txns

    if n == 0:
        return 0, []

    cost_cache = {}

    def evaluate(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(list(key))
        return cost_cache[key]

    # Exhaustive search is practical at this size and guarantees optimality.
    if n <= 7:
        import itertools

        best_cost = float("inf")
        best_schedule = None
        for sequence in itertools.permutations(range(n)):
            cost = evaluate(sequence)
            if cost < best_cost:
                best_cost = cost
                best_schedule = list(sequence)
        return best_cost, best_schedule

    all_txns = tuple(range(n))
    beam_width = max(8, min(48, max(1, int(num_seqs)) * 5))

    # Start by evaluating all possible first transactions.
    beam = [(evaluate((txn,)), (txn,)) for txn in all_txns]
    beam.sort(key=lambda entry: (entry[0], entry[1]))
    beam = beam[:beam_width]

    for depth in range(1, n):
        expanded = []

        for _, prefix in beam:
            used = set(prefix)
            for txn in all_txns:
                if txn not in used:
                    candidate = prefix + (txn,)
                    expanded.append((evaluate(candidate), candidate))

        expanded.sort(key=lambda entry: (entry[0], entry[1]))

        # Retain the best overall prefixes, then reserve part of the beam for
        # different endpoints.  Different final transactions often represent
        # different unresolved read/write conflict chains.
        elite_count = max(1, (beam_width * 3) // 4)
        selected = expanded[:elite_count]
        selected_keys = {prefix for _, prefix in selected}
        seen_last = {prefix[-1] for _, prefix in selected}

        for candidate in expanded[elite_count:]:
            prefix = candidate[1]
            if prefix[-1] not in seen_last:
                selected.append(candidate)
                selected_keys.add(prefix)
                seen_last.add(prefix[-1])
                if len(selected) >= beam_width:
                    break

        if len(selected) < beam_width:
            for candidate in expanded[elite_count:]:
                if candidate[1] not in selected_keys:
                    selected.append(candidate)
                    selected_keys.add(candidate[1])
                    if len(selected) >= beam_width:
                        break

        beam = selected

        if workload.debug:
            print(
                "beam depth:",
                depth + 1,
                "best prefix:",
                list(beam[0][1]),
                "cost:",
                beam[0][0],
            )

    if n <= 30:
        positions = list(range(n))
    else:
        stride = max(1, n // 12)
        positions = list(range(0, n, stride))
        if positions[-1] != n - 1:
            positions.append(n - 1)

    def descend(initial_schedule, initial_cost=None, max_rounds=None):
        """
        Best-improvement variable-neighborhood descent.

        Insertion repairs long conflict chains, swaps repair paired ordering
        mistakes, and reversals alter several conflict directions together.
        """
        current = list(initial_schedule)
        current_cost = evaluate(current) if initial_cost is None else initial_cost

        if max_rounds is None:
            max_rounds = max(2, n)

        for _ in range(max_rounds):
            round_best_cost = current_cost
            round_best_schedule = current

            for source in positions:
                stripped = current[:source] + current[source + 1:]
                moved = current[source]

                for destination in positions:
                    insert_at = min(destination, len(stripped))
                    if insert_at == source:
                        continue

                    candidate = (
                        stripped[:insert_at]
                        + [moved]
                        + stripped[insert_at:]
                    )
                    candidate_cost = evaluate(candidate)

                    if candidate_cost < round_best_cost:
                        round_best_cost = candidate_cost
                        round_best_schedule = candidate

            for left_offset, left in enumerate(positions):
                for right in positions[left_offset + 1:]:
                    candidate = current.copy()
                    candidate[left], candidate[right] = candidate[right], candidate[left]
                    candidate_cost = evaluate(candidate)

                    if candidate_cost < round_best_cost:
                        round_best_cost = candidate_cost
                        round_best_schedule = candidate

                for right in positions[left_offset + 2:]:
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

    # Refine several beam endings, rather than assuming the lowest prefix-cost
    # completion is always the best local-search starting point.
    seed_count = min(len(beam), max(4, min(12, int(num_seqs) + 2)))
    local_optima = []

    best_cost = float("inf")
    best_schedule = None

    for seed_cost, seed_schedule in beam[:seed_count]:
        cost, schedule = descend(list(seed_schedule), seed_cost)
        local_optima.append((cost, schedule))

        if cost < best_cost:
            best_cost = cost
            best_schedule = schedule

    # Keep distinct local optima as sources for directed perturbations.
    local_optima.sort(key=lambda entry: (entry[0], entry[1]))
    perturb_sources = []
    seen_sources = set()

    for cost, schedule in local_optima:
        key = tuple(schedule)
        if key not in seen_sources:
            perturb_sources.append((cost, schedule))
            seen_sources.add(key)
        if len(perturb_sources) >= min(3, len(local_optima)):
            break

    kick_costs = {}

    def add_kick(candidate):
        key = tuple(candidate)
        if key not in kick_costs:
            kick_costs[key] = evaluate(candidate)

    # Directed perturbations retain much of a good local ordering while
    # changing conflict directions that strict descent cannot cross.
    for _, source_schedule in perturb_sources:
        for left_offset, left in enumerate(positions):
            for right in positions[left_offset + 1:]:
                swapped = source_schedule.copy()
                swapped[left], swapped[right] = swapped[right], swapped[left]
                add_kick(swapped)

                if right - left >= 2:
                    add_kick(
                        source_schedule[:left]
                        + source_schedule[left:right + 1][::-1]
                        + source_schedule[right + 1:]
                    )

        # Move two adjacent transactions together.  This preserves useful
        # internal ordering while moving a conflict-producing block.
        for source in range(n - 1):
            block = source_schedule[source:source + 2]
            remainder = source_schedule[:source] + source_schedule[source + 2:]

            for destination in positions:
                insert_at = min(destination, len(remainder))
                if insert_at != source:
                    add_kick(
                        remainder[:insert_at]
                        + block
                        + remainder[insert_at:]
                    )

    ranked_kicks = sorted(
        ((cost, list(schedule)) for schedule, cost in kick_costs.items()),
        key=lambda entry: (entry[0], entry[1]),
    )

    kick_count = min(len(ranked_kicks), max(4, min(10, int(num_seqs))))
    for kick_cost, kick_schedule in ranked_kicks[:kick_count]:
        cost, schedule = descend(kick_schedule, kick_cost)

        if cost < best_cost:
            best_cost = cost
            best_schedule = schedule

    if workload.debug:
        print("best schedule:", best_schedule)
        print("best makespan:", best_cost)

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