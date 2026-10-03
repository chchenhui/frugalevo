import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Build a conflict-cost beam of transaction prefixes, then optimize by relocation."""
    n = workload.num_txns
    if not n:
        return 0, []

    cache = {}

    def cost(order):
        key = tuple(order)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(order)
        return cache[key]

    # Unlike independent greedy restarts, a beam preserves several promising
    # prefix decisions.  This is useful when a locally cheap next transaction
    # creates expensive read/write dependencies later in the order.
    width = min(8, max(3, num_seqs))
    beam = [([], tuple(range(n)))]
    for depth in range(n):
        expanded = []
        for order, remaining in beam:
            choices = remaining if len(remaining) <= 45 else remaining[:22]
            for txn in choices:
                trial = order + [txn]
                expanded.append((cost(trial), trial,
                                 tuple(x for x in remaining if x != txn)))
        expanded.sort(key=lambda item: item[0])

        # Equal prefix costs are common; retain distinct last transactions so
        # the beam does not collapse into effectively one greedy ordering.
        beam = []
        seen_last = set()
        for _, order, remaining in expanded:
            if order[-1] not in seen_last or len(beam) < width // 2:
                beam.append((order, remaining))
                seen_last.add(order[-1])
            if len(beam) == width:
                break

    best_order = min((order for order, _ in beam), key=cost)
    best_cost = cost(best_order)

    # Best-improvement relocation evaluates actual makespan, not an operation
    # count proxy.  Relocation includes adjacent swaps and long moves.
    for _ in range(8):
        next_cost, next_order = best_cost, best_order
        if n <= 32:
            moves = ((source, target) for source in range(n)
                     for target in range(n) if source != target)
        else:
            moves = ((source, target) for source in range(n)
                     for target in range(n) if abs(source - target) > 1)

        for source, target in moves:
            txn = best_order[source]
            reduced = best_order[:source] + best_order[source + 1:]
            trial = reduced[:target] + [txn] + reduced[target:]
            value = cost(trial)
            if value < next_cost:
                next_cost, next_order = value, trial

        if next_cost == best_cost:
            break
        best_cost, best_order = next_cost, next_order

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
