import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction ordering using a cost-aware beam search
    followed by insertion-based local optimization.

    The simulator's get_opt_seq_cost is used as the only objective signal:
    no proxy based on transaction size, read count, or write count is used.
    """
    n = workload.num_txns

    if n == 0:
        return 0, []
    if n == 1:
        schedule = [0]
        return workload.get_opt_seq_cost(schedule), schedule

    # Prefixes and complete schedules are often revisited by the local search.
    # Memoizing them makes the more exhaustive search substantially cheaper.
    cost_cache = {}

    def sequence_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(list(key))
        return cost_cache[key]

    # A wider beam is valuable for small workloads, where exhaustive-looking
    # prefix exploration is inexpensive.  Limit it for large workloads so that
    # expansion remains bounded.
    requested_width = max(2, int(num_seqs))
    size_limited_width = max(2, 240 // max(1, n))
    beam_width = min(8, requested_width, size_limited_width)
    beam_width = max(2, beam_width)

    all_txns = tuple(range(n))

    # Each entry is (prefix_cost, prefix_tuple, remaining_tuple).
    # Start from an empty prefix; the first expansion evaluates every possible
    # starting transaction, rather than choosing one randomly.
    beam = [(0, tuple(), all_txns)]

    for depth in range(n):
        expanded = []

        for _, prefix, remaining in beam:
            for txn in remaining:
                next_prefix = prefix + (txn,)
                next_remaining = tuple(x for x in remaining if x != txn)
                next_cost = sequence_cost(next_prefix)
                expanded.append((next_cost, next_prefix, next_remaining))

        # Stable lexical tie breaking makes results reproducible and avoids
        # random runs selecting weaker schedules.
        expanded.sort(key=lambda entry: (entry[0], entry[1]))

        # Keep a small amount of endpoint diversity.  Prefix-cost landscapes
        # can contain many near-identical paths; retaining distinct recent
        # choices prevents the beam from collapsing immediately onto one path.
        selected = []
        seen_last = set()

        for candidate in expanded:
            last_txn = candidate[1][-1]
            if last_txn not in seen_last:
                selected.append(candidate)
                seen_last.add(last_txn)
                if len(selected) >= beam_width:
                    break

        if len(selected) < beam_width:
            selected_keys = {entry[1] for entry in selected}
            for candidate in expanded:
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

    def improve_by_insertion(initial_schedule):
        """
        Best-improvement insertion descent.

        An insertion removes transaction i and inserts it at position j.
        This can repair delayed read/write conflict chains that require more
        than a single adjacent swap.  Search effort is capped for large n.
        """
        current = tuple(initial_schedule)
        current_cost = sequence_cost(current)

        # Small workloads receive full best-insertion search.  Larger workloads
        # use a deterministic bounded neighborhood plus adjacent moves.
        if n <= 30:
            evaluation_budget = max(400, 4 * n * n)
        else:
            evaluation_budget = max(600, 12 * n)

        used = 0
        rounds = 0
        max_rounds = 4 if n <= 30 else 2

        while rounds < max_rounds and used < evaluation_budget:
            rounds += 1
            best_schedule = current
            best_cost = current_cost

            if n <= 30:
                source_indices = range(n)
            else:
                # Include every transaction but examine a limited set of
                # destination positions around and across the schedule.
                source_indices = range(n)

            for source in source_indices:
                reduced = current[:source] + current[source + 1:]

                if n <= 30:
                    destinations = range(n)
                else:
                    step = max(1, n // 8)
                    destinations = {
                        0,
                        n - 1,
                        max(0, source - step),
                        source,
                        min(n - 1, source + step),
                    }
                    destinations.update(range(0, n, step))

                for destination in destinations:
                    if used >= evaluation_budget:
                        break

                    candidate = (
                        reduced[:destination]
                        + (current[source],)
                        + reduced[destination:]
                    )

                    if candidate == current:
                        continue

                    used += 1
                    candidate_cost = sequence_cost(candidate)

                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_schedule = candidate

                if used >= evaluation_budget:
                    break

            if best_schedule == current:
                break

            current = best_schedule
            current_cost = best_cost

        return current_cost, list(current)

    # The beam's leading schedules can have different final conflict behavior
    # despite similar prefix scores.  Refine several of them and retain the
    # best complete ordering.
    finalists = sorted(beam, key=lambda entry: (entry[0], entry[1]))
    finalist_count = min(len(finalists), max(2, min(4, beam_width)))

    best_cost = float("inf")
    best_schedule = None

    for _, schedule, _ in finalists[:finalist_count]:
        improved_cost, improved_schedule = improve_by_insertion(schedule)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_schedule = improved_schedule

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