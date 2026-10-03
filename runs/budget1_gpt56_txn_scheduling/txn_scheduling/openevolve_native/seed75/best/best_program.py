import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use bidirectional makespan-guided beams followed by relocation, swap, and block-reversal descent."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(order):
        key = tuple(order)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(order)
        return cache[key]

    # Prefix and suffix construction expose different conflict patterns:
    # writers delayed by a forward build are often naturally discovered by a
    # reverse build.  Keep a small beam rather than committing to one greedy
    # serialization.
    width = max(1, min(n, max(4, num_seqs)))
    complete_orders = []

    for backwards in (False, True):
        states = [(txn,) for txn in range(n)]
        while len(states[0]) < n:
            candidates = []
            for state in states:
                used = set(state)
                for txn in range(n):
                    if txn not in used:
                        candidates.append((txn,) + state if backwards
                                          else state + (txn,))
            random.shuffle(candidates)
            candidates.sort(key=cost)
            states = candidates[:width]
        complete_orders.extend(states)

    orders = []
    seen = set()
    for order in complete_orders:
        if order not in seen:
            seen.add(order)
            orders.append(list(order))
    orders.sort(key=cost)

    best_order = orders[0][:]
    best_cost = cost(best_order)

    # Search more than one beam result.  Relocation changes a transaction's
    # relationship with every operation between its old and new positions;
    # swaps and reversals then escape minima requiring coordinated changes.
    for initial in orders[:min(6, len(orders))]:
        order = initial[:]
        value = cost(order)

        for _ in range(6):
            candidate, candidate_cost = order, value

            for source in range(n):
                txn = order[source]
                reduced = order[:source] + order[source + 1:]
                for destination in range(n):
                    if destination == source:
                        continue
                    trial = reduced[:destination] + [txn] + reduced[destination:]
                    trial_cost = cost(trial)
                    if trial_cost < candidate_cost:
                        candidate, candidate_cost = trial, trial_cost

            if candidate_cost < value:
                order, value = candidate, candidate_cost
                continue

            for left in range(n - 1):
                for right in range(left + 1, n):
                    trial = order[:]
                    trial[left], trial[right] = trial[right], trial[left]
                    trial_cost = cost(trial)
                    if trial_cost < candidate_cost:
                        candidate, candidate_cost = trial, trial_cost

            if candidate_cost < value:
                order, value = candidate, candidate_cost
                continue

            # Reversing a contiguous conflict cluster can improve schedules
            # where neither endpoint transaction can profitably move alone.
            for left in range(n - 1):
                for right in range(left + 2, n):
                    trial = order[:left] + order[left:right + 1][::-1] + order[right + 1:]
                    trial_cost = cost(trial)
                    if trial_cost < candidate_cost:
                        candidate, candidate_cost = trial, trial_cost

            if candidate_cost >= value:
                break
            order, value = candidate, candidate_cost

        if value < best_cost:
            best_order, best_cost = order, value

    # For small workloads, use the locally optimized schedule as an incumbent
    # for branch-and-bound.  The cost of a transaction prefix is a lower bound
    # on every completion: adding transactions can only add operations and
    # conflict constraints.  This turns the heuristic result into an exact
    # search when pruning is effective, while retaining a strict node limit
    # for predictable runtime on difficult ten-transaction instances.
    if n <= 10:
        nodes = 0
        node_limit = 750000

        def branch(prefix, remaining, prefix_cost):
            """Explore promising completions, pruning prefixes above incumbent."""
            nonlocal best_cost, best_order, nodes
            if nodes >= node_limit:
                return
            if not remaining:
                if prefix_cost < best_cost:
                    best_cost, best_order = prefix_cost, prefix[:]
                return

            children = []
            for txn in remaining:
                child = prefix + [txn]
                child_cost = cost(child)
                nodes += 1
                if child_cost < best_cost:
                    children.append((child_cost, txn))

            # Trying low-cost prefixes first obtains stronger incumbents early,
            # causing substantially more branches to be discarded.
            children.sort()
            for child_cost, txn in children:
                if nodes >= node_limit:
                    return
                branch(
                    prefix + [txn],
                    tuple(item for item in remaining if item != txn),
                    child_cost,
                )

        branch([], tuple(range(n)), cost([]))

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
