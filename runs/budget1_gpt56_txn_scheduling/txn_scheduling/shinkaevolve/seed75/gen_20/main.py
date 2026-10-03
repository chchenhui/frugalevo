import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using exact enumeration for small
    workloads, cost-based beam construction for larger workloads, and local
    permutation improvement.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    # The workload simulator remains the single source of truth for all
    # conflict and makespan decisions. Prefixes and local moves recur often,
    # so caching materially reduces the number of simulator invocations.
    cost_cache = {}

    def cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    # Exhaustive search is inexpensive for small workloads and guarantees the
    # globally optimal transaction ordering rather than merely a local optimum.
    if n <= 8:
        import itertools

        best_value = None
        best_order = None
        for order in itertools.permutations(range(n)):
            value = cost(order)
            if best_value is None or value < best_value:
                best_value = value
                best_order = list(order)
        return best_value, best_order

    # Keep more alternatives than the original independent greedy restarts.
    # A bounded width keeps the search practical for larger input workloads.
    requested = max(1, int(num_seqs))
    beam_width = min(24, max(4, requested * 2))

    # A beam state is (partial_makespan, tuple_of_transactions).  Start from
    # every transaction where possible, since first-position conflicts can
    # strongly affect the eventual critical path.
    initial = list(range(n))
    random.shuffle(initial)
    initial_states = [(cost([txn]), (txn,)) for txn in initial]
    initial_states.sort(key=lambda state: state[0])
    beam = initial_states[:beam_width]

    for depth in range(1, n):
        expanded = []
        seen = set()

        for _, prefix in beam:
            used = set(prefix)
            choices = [txn for txn in range(n) if txn not in used]
            random.shuffle(choices)

            for txn in choices:
                candidate = prefix + (txn,)
                if candidate in seen:
                    continue
                seen.add(candidate)
                expanded.append((cost(candidate), candidate))

        # Actual partial makespan is the primary criterion.  Retaining a
        # distinct final transaction first adds useful diversity among equal or
        # near-equal prefixes, avoiding premature convergence of the beam.
        expanded.sort(key=lambda state: state[0])
        beam = []
        used_last = set()

        for state in expanded:
            last_txn = state[1][-1]
            if last_txn not in used_last:
                beam.append(state)
                used_last.add(last_txn)
                if len(beam) >= beam_width:
                    break

        if len(beam) < beam_width:
            selected = {state[1] for state in beam}
            for state in expanded:
                if state[1] not in selected:
                    beam.append(state)
                    selected.add(state[1])
                    if len(beam) >= beam_width:
                        break

    beam.sort(key=lambda state: state[0])

    def best_relocation(schedule, current_cost):
        """
        Search all single-transaction remove/insert moves. This directly
        corrects greedy placement errors caused by delayed read/write conflicts.
        """
        best_value = current_cost
        best_order = schedule

        for source in range(n):
            txn = schedule[source]
            reduced = schedule[:source] + schedule[source + 1:]

            for destination in range(n):
                if destination == source:
                    continue
                candidate = reduced[:destination] + [txn] + reduced[destination:]
                value = cost(candidate)
                if value < best_value:
                    best_value = value
                    best_order = candidate

        return best_value, best_order

    def best_swap(schedule, current_cost):
        """
        Swaps complement relocations when two conflict-heavy transactions need
        to cross each other together to reduce the critical path.
        """
        best_value = current_cost
        best_order = schedule

        for left in range(n - 1):
            for right in range(left + 1, n):
                candidate = schedule[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                value = cost(candidate)
                if value < best_value:
                    best_value = value
                    best_order = candidate

        return best_value, best_order

    def improve(schedule, initial_cost):
        current_schedule = schedule[:]
        current_cost = initial_cost
        max_passes = max(2, min(8, n))

        for _ in range(max_passes):
            relocated_cost, relocated_schedule = best_relocation(
                current_schedule, current_cost
            )
            if relocated_cost < current_cost:
                current_cost, current_schedule = relocated_cost, relocated_schedule
                continue

            swapped_cost, swapped_schedule = best_swap(
                current_schedule, current_cost
            )
            if swapped_cost < current_cost:
                current_cost, current_schedule = swapped_cost, swapped_schedule
                continue

            break

        return current_cost, current_schedule

    # Improve several independently promising completed schedules, not only
    # the single best beam result. A slightly worse construction can belong to
    # a better local-search basin.
    finalists = beam[:min(len(beam), max(3, requested))]
    best_value = None
    best_schedule = None

    for candidate_cost, candidate_tuple in finalists:
        value, schedule = improve(list(candidate_tuple), candidate_cost)
        if best_value is None or value < best_value:
            best_value = value
            best_schedule = schedule

    return cost(best_schedule), best_schedule

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
