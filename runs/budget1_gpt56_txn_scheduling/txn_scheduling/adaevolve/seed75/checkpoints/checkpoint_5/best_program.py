import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Construct conflict-aware greedy schedules, then improve them by insertion and swap search.

    Prefix candidates are scored with the simulator's real makespan.  Later
    restarts use a small restricted candidate list, allowing the search to escape
    a greedy prefix choice that looks good locally but creates a long future
    conflict chain.  A memoized insertion descent and one non-adjacent swap
    neighborhood then optimize the best complete permutation.
    """
    n = workload.num_txns
    if n <= 1:
        seq = list(range(n))
        return workload.get_opt_seq_cost(seq), seq

    cache = {}

    def cost(seq):
        key = tuple(seq)
        value = cache.get(key)
        if value is None:
            value = workload.get_opt_seq_cost(seq)
            cache[key] = value
        return value

    restarts = max(1, min(int(num_seqs), n))
    best_cost = float("inf")
    best_seq = None

    for restart in range(restarts):
        rng = random.Random(7919 + restart)
        start = (restart * n) // restarts
        seq = [start]
        remaining = [txn for txn in range(n) if txn != start]

        while remaining:
            scored = sorted((cost(seq + [txn]), txn) for txn in remaining)

            # Keep several purely greedy starts, but let later starts explore
            # near-best prefix choices that may yield a better final schedule.
            if restart < max(1, restarts // 2):
                width = 1
            else:
                width = min(3, len(scored))
            chosen = scored[rng.randrange(width)][1]

            seq.append(chosen)
            remaining.remove(chosen)

        candidate_cost = cost(seq)
        if candidate_cost < best_cost:
            best_cost, best_seq = candidate_cost, seq

    # Relocation directly repairs transactions positioned on an expensive
    # read/write dependency chain.  Applying improvements during a pass allows
    # later transactions to benefit from earlier relocations.
    for _ in range(2):
        improved = False
        for source in range(n):
            txn = best_seq[source]
            base = best_seq[:source] + best_seq[source + 1:]
            move_cost = best_cost
            move_seq = best_seq
            for destination in range(n):
                if destination == source:
                    continue
                candidate = base[:destination] + [txn] + base[destination:]
                candidate_cost = cost(candidate)
                if candidate_cost < move_cost:
                    move_cost, move_seq = candidate_cost, candidate
            if move_cost < best_cost:
                best_cost, best_seq = move_cost, move_seq
                improved = True
        if not improved:
            break

    # A non-adjacent exchange changes both transactions' conflict relationships
    # with the intervening region, which cannot always be improved by one move.
    swap_cost = best_cost
    swap_seq = best_seq
    for left in range(n - 1):
        for right in range(left + 1, n):
            candidate = best_seq[:]
            candidate[left], candidate[right] = candidate[right], candidate[left]
            candidate_cost = cost(candidate)
            if candidate_cost < swap_cost:
                swap_cost, swap_seq = candidate_cost, candidate

    if swap_cost < best_cost:
        best_cost, best_seq = swap_cost, swap_seq
        for source in range(n):
            txn = best_seq[source]
            base = best_seq[:source] + best_seq[source + 1:]
            move_cost = best_cost
            move_seq = best_seq
            for destination in range(n):
                if destination == source:
                    continue
                candidate = base[:destination] + [txn] + base[destination:]
                candidate_cost = cost(candidate)
                if candidate_cost < move_cost:
                    move_cost, move_seq = candidate_cost, candidate
            if move_cost < best_cost:
                best_cost, best_seq = move_cost, move_seq

    return best_cost, best_seq

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
