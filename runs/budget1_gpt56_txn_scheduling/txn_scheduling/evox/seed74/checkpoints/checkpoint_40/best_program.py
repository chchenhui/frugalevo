import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use an exact-prefix beam and simulator-scored variable-neighborhood
    descent to minimize read/write-conflict makespan."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(order):
        """Return the cached authoritative simulator cost of an order."""
        key = tuple(order)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(order)
        return cache[key]

    # Do not randomly discard transactions while extending a prefix.  A writer
    # omitted from a sampled extension can be precisely the transaction whose
    # early placement breaks the eventual critical dependency chain.  The
    # number of simulator calls remains bounded by beam_width * n * n.
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

        # Shuffle before sorting only randomizes genuine equal-cost prefixes;
        # it avoids a persistent transaction-ID bias without weakening scoring.
        random.shuffle(expanded)
        expanded.sort(key=lambda entry: entry[0])
        beam = [order for _, order in expanded[:width]]

    def improve(order, rounds):
        """Apply best improving swaps, relocations, and block reversals."""
        value = cost(order)

        for _ in range(rounds):
            best_order, best_value = order, value

            if n <= 35:
                swaps = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
                moves = [(i, j) for i in range(n) for j in range(n) if i != j]
                reversals = [(i, j) for i in range(n - 1)
                             for j in range(i + 2, n)]
            else:
                swaps = set()
                for _ in range(9 * n):
                    i, j = random.sample(range(n), 2)
                    swaps.add((min(i, j), max(i, j)))
                moves = [tuple(random.sample(range(n), 2))
                         for _ in range(8 * n)]
                reversals = [tuple(sorted(random.sample(range(n), 2)))
                             for _ in range(3 * n)]

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

            # A reversal changes several related conflict orientations at once,
            # allowing descent to leave basins requiring more than one move.
            for left, right in reversals:
                if right - left < 2:
                    continue
                candidate = order[:]
                candidate[left:right + 1] = reversed(candidate[left:right + 1])
                candidate_value = cost(candidate)
                if candidate_value < best_value:
                    best_order, best_value = candidate, candidate_value

            if best_value >= value:
                break
            order, value = best_order, best_value

        return value, order

    # Prefix cost is not a complete estimate of final cost, so descend several
    # finished beam survivors rather than trusting only the leading one.
    elite_count = min(len(beam), 5 if n <= 35 else 4)
    best_value, best_order = None, None
    for order in beam[:elite_count]:
        value, order = improve(order, 6 if n <= 35 else 4)
        if best_value is None or value < best_value:
            best_value, best_order = value, order

    # Iterated local search retains the good schedule while perturbing enough
    # precedence decisions to expose a different conflict-chain orientation.
    for restart in range(5 if n <= 35 else 4):
        candidate = best_order[:]
        for _ in range(2 + restart % 3):
            if n > 4 and random.random() < 0.4:
                left, right = sorted(random.sample(range(n), 2))
                candidate[left:right + 1] = reversed(candidate[left:right + 1])
            else:
                source, target = random.sample(range(n), 2)
                candidate.insert(target, candidate.pop(source))
        value, candidate = improve(candidate, 5 if n <= 35 else 3)
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
