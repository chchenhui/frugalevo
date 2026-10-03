import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Construct a simulator-scored prefix beam, then perform variable-neighborhood
    descent and perturbed restarts using the final authoritative makespan."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(order):
        key = tuple(order)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(order)
        return cache[key]

    # Every possible next transaction is scored.  Sampling the remaining
    # choices can discard the writer/reader that must be placed early to break
    # the final critical conflict chain.  The wider beam retains several such
    # dependency orientations through the complete prefix construction.
    width = min(n, max(4, min(12, num_seqs + 2)))
    starts = list(range(n))
    random.shuffle(starts)
    beam = [[txn] for txn in starts[:width]]

    for _ in range(1, n):
        expanded = []
        for order in beam:
            used = set(order)
            for txn in range(n):
                if txn not in used:
                    candidate = order + [txn]
                    expanded.append((cost(candidate), candidate))

        # Shuffle affects only ties while preventing a fixed transaction-ID
        # preference from eliminating structurally distinct equal-cost prefixes.
        random.shuffle(expanded)
        expanded.sort(key=lambda entry: entry[0])
        beam = [order for _, order in expanded[:width]]

    def descend(order, rounds):
        """Apply the best strict improvement among swap, insertion, and reversal."""
        value = cost(order)

        for _ in range(rounds):
            best_value, best_order = value, order

            if n <= 30:
                swaps = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
                moves = [(i, j) for i in range(n) for j in range(n) if i != j]
                reversals = [(i, j) for i in range(n - 1)
                             for j in range(i + 2, n + 1)]
            else:
                swaps = set()
                for _ in range(8 * n):
                    i, j = sorted(random.sample(range(n), 2))
                    swaps.add((i, j))
                moves = [tuple(random.sample(range(n), 2)) for _ in range(7 * n)]
                reversals = []
                for _ in range(4 * n):
                    i, j = sorted(random.sample(range(n + 1), 2))
                    if j - i > 1:
                        reversals.append((i, j))

            for i, j in swaps:
                candidate = order[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_value = cost(candidate)
                if candidate_value < best_value:
                    best_value, best_order = candidate_value, candidate

            for source, target in moves:
                candidate = order[:]
                candidate.insert(target, candidate.pop(source))
                candidate_value = cost(candidate)
                if candidate_value < best_value:
                    best_value, best_order = candidate_value, candidate

            # Reversing a run changes several dependency orientations together.
            # This reaches schedules that require multiple individually
            # non-improving swaps or insertions.
            for left, right in reversals:
                candidate = order[:]
                candidate[left:right] = reversed(candidate[left:right])
                candidate_value = cost(candidate)
                if candidate_value < best_value:
                    best_value, best_order = candidate_value, candidate

            if best_value >= value:
                break
            value, order = best_value, best_order

        return value, order

    elites = beam if n <= 30 else beam[:min(4, len(beam))]
    basins = []
    best_value, best_order = None, None

    for order in elites:
        value, improved = descend(order, 7 if n <= 30 else 4)
        basins.append((value, improved))
        if best_value is None or value < best_value:
            best_value, best_order = value, improved

    # Strict descent can stop at a local conflict orientation.  Perturbing
    # elite schedules and descending again provides inexpensive escape paths
    # while the cache prevents repeated simulator evaluations.
    basins.sort(key=lambda entry: entry[0])
    restarts = 10 if n <= 30 else 6
    for restart in range(restarts):
        _, seed = basins[restart % len(basins)]
        candidate = seed[:]

        perturbations = 1 + restart // 3 if n <= 30 else 2 + restart // 3
        for _ in range(perturbations):
            if n > 3 and random.random() < 0.45:
                left, right = sorted(random.sample(range(n + 1), 2))
                if right - left > 1:
                    candidate[left:right] = reversed(candidate[left:right])
            else:
                source, target = random.sample(range(n), 2)
                candidate.insert(target, candidate.pop(source))

        value, improved = descend(candidate, 7 if n <= 30 else 4)
        basins.append((value, improved))
        if value < best_value:
            best_value, best_order = value, improved

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
