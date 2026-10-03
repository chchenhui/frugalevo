import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Build forward and reverse cost-guided beams, then descend by relocations and swaps."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(order):
        key = tuple(order)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(order)
        return cache[key]

    # A beam keeps several promising partial serializations instead of making
    # one irreversible greedy choice.  Its width is tied to num_seqs so the
    # search remains practical on larger workloads.
    width = max(1, min(n, max(4, num_seqs)))
    complete_orders = []

    for backwards in (False, True):
        states = [(txn,) for txn in range(n)]

        while len(states[0]) < n:
            next_states = []
            for state in states:
                used = set(state)
                for txn in range(n):
                    if txn not in used:
                        next_states.append((txn,) + state if backwards
                                           else state + (txn,))

            # Random tie breaking preserves useful diversity when many partial
            # schedules have exactly the same simulated makespan.
            random.shuffle(next_states)
            next_states.sort(key=cost)
            states = next_states[:width]

        complete_orders.extend(states)

    # Different beam paths can converge to the same complete ordering.
    unique_orders = []
    seen = set()
    for order in complete_orders:
        if order not in seen:
            seen.add(order)
            unique_orders.append(list(order))

    unique_orders.sort(key=cost)
    best_order = unique_orders[0]
    best_cost = cost(best_order)

    # Improve several beam finalists, not merely the currently best one.
    # Relocation is especially effective for moving a transaction away from
    # the conflicting writers/readers that determine the critical path.
    for initial in unique_orders[:min(4, len(unique_orders))]:
        order = initial[:]
        value = cost(order)

        for _ in range(3):
            move_order, move_cost = order, value

            for source in range(n):
                reduced = order[:source] + order[source + 1:]
                txn = order[source]
                for destination in range(n):
                    if destination == source:
                        continue
                    trial = reduced[:destination] + [txn] + reduced[destination:]
                    trial_cost = cost(trial)
                    if trial_cost < move_cost:
                        move_order, move_cost = trial, trial_cost

            if move_cost < value:
                order, value = move_order, move_cost
                continue

            # When no single relocation helps, an exchange can cross two
            # conflict groups simultaneously and escape that local minimum.
            for left in range(n - 1):
                for right in range(left + 1, n):
                    trial = order[:]
                    trial[left], trial[right] = trial[right], trial[left]
                    trial_cost = cost(trial)
                    if trial_cost < move_cost:
                        move_order, move_cost = trial, trial_cost

            if move_cost >= value:
                break
            order, value = move_order, move_cost

        if value < best_cost:
            best_order, best_cost = order, value

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
