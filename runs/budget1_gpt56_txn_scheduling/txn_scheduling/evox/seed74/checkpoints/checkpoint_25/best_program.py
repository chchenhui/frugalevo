import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Build a conflict-scored beam, then use swap, insertion, reversal, and
    perturb-and-redescend search evaluated by the authoritative simulator."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(order):
        """Return the cached simulator makespan for a partial or full order."""
        key = tuple(order)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(order)
        return cache[key]

    # Preserve several cheap partial conflict orientations instead of making
    # one irreversible greedy choice.
    width = min(n, max(2, min(8, num_seqs)))
    starts = list(range(n))
    random.shuffle(starts)
    beam = [[txn] for txn in starts[:width]]

    for _ in range(1, n):
        expanded = []
        for order in beam:
            used = set(order)
            remaining = [txn for txn in range(n) if txn not in used]
            if len(remaining) > 24:
                remaining = random.sample(remaining, 24)
            for txn in remaining:
                candidate = order + [txn]
                expanded.append((cost(candidate), candidate))

        random.shuffle(expanded)
        expanded.sort(key=lambda entry: entry[0])
        beam = [order for _, order in expanded[:width]]

    def improve(order):
        """Take best simulator-scored swap, insertion, or block reversal moves."""
        value = cost(order)
        rounds = 6 if n <= 30 else 3

        for _ in range(rounds):
            best_order, best_value = order, value
            if n <= 30:
                swaps = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
                moves = [(i, j) for i in range(n) for j in range(n) if i != j]
                reversals = [(i, j) for i in range(n - 2)
                             for j in range(i + 2, n)]
            else:
                swaps = set()
                for _ in range(5 * n):
                    i, j = random.sample(range(n), 2)
                    swaps.add((min(i, j), max(i, j)))
                moves = [tuple(random.sample(range(n), 2)) for _ in range(4 * n)]
                reversals = []
                for _ in range(3 * n):
                    i, j = sorted(random.sample(range(n), 2))
                    if j - i > 1:
                        reversals.append((i, j))

            for i, j in swaps:
                candidate = order[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_value = cost(candidate)
                if candidate_value < best_value:
                    best_order, best_value = candidate, candidate_value

            for source, target in moves:
                candidate = order[:]
                candidate.insert(target, candidate.pop(source))
                candidate_value = cost(candidate)
                if candidate_value < best_value:
                    best_order, best_value = candidate, candidate_value

            # A reversal changes the precedence direction of every pair in a
            # conflict cluster, which can shorten a whole dependency chain.
            for left, right in reversals:
                candidate = order[:]
                candidate[left:right + 1] = reversed(candidate[left:right + 1])
                candidate_value = cost(candidate)
                if candidate_value < best_value:
                    best_order, best_value = candidate, candidate_value

            if best_value >= value:
                break
            order, value = best_order, best_value

        return value, order

    elite_count = len(beam) if n <= 30 else min(len(beam), 3)
    best_value, best_order = None, None
    for order in beam[:elite_count]:
        value, order = improve(order)
        if best_value is None or value < best_value:
            best_value, best_order = value, order

    # Larger disruptions expose alternative orderings of groups of mutually
    # conflicting writers/readers; only improved fully simulated schedules win.
    restarts = 7 if n <= 30 else 3
    for restart in range(restarts):
        candidate = best_order[:]
        kicks = 2 + restart % 3 if n <= 30 else 3
        for _ in range(kicks):
            if n > 3 and random.random() < 0.4:
                left, right = sorted(random.sample(range(n), 2))
                candidate[left:right + 1] = reversed(candidate[left:right + 1])
            else:
                source, target = random.sample(range(n), 2)
                candidate.insert(target, candidate.pop(source))
        value, candidate = improve(candidate)
        if value < best_value:
            best_value, best_order = value, candidate

    return best_value, best_order

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
