import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use a beam of lowest-cost order prefixes, then exact local improvement."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(order):
        """Return the cached simulator makespan for an order or order prefix."""
        key = tuple(order)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(order)
        return cache[key]

    def improve(order):
        """Perform bounded exact swap and relocation first-improvement descent."""
        best = cost(order)
        for _ in range(max(2, min(n, 8))):
            moved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = order[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    value = cost(candidate)
                    if value < best:
                        order, best, moved = candidate, value, True
                        break
                if moved:
                    break
            if moved:
                continue

            for i in range(n):
                item = order[i]
                reduced = order[:i] + order[i + 1:]
                for j in range(n):
                    candidate = reduced[:j] + [item] + reduced[j:]
                    value = cost(candidate)
                    if value < best:
                        order, best, moved = candidate, value, True
                        break
                if moved:
                    break
            if not moved:
                break
        return best, order

    # Unlike independent greedy starts, a beam keeps the globally strongest
    # partial schedules at every position and can recover from a weak first txn.
    width = max(1, num_seqs)
    beam = [[]]
    for _ in range(n):
        expanded = []
        for prefix in beam:
            used = set(prefix)
            for txn in range(n):
                if txn not in used:
                    candidate = prefix + [txn]
                    expanded.append((cost(candidate), candidate))
        # Randomizing first preserves useful diversity when prefix costs tie.
        random.shuffle(expanded)
        expanded.sort(key=lambda entry: entry[0])
        beam = [order for _, order in expanded[:width]]

    best_cost = float("inf")
    best_order = None
    for seed in beam:
        # Reversal remains valuable because read/write conflict direction matters.
        for order in (seed, list(reversed(seed))):
            value, candidate = improve(order)
            if value < best_cost:
                best_cost, best_order = value, candidate

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
