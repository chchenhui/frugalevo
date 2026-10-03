import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use simulator-scored beam construction, insertion descent, and swaps.

    The beam retains several low-makespan partial orders at every depth, avoiding
    the irreversible prefix decisions of independent greedy restarts.  Complete
    beam finalists are then improved using exact relocation and exchange moves.
    """
    n = workload.num_txns
    if n <= 1:
        seq = list(range(n))
        return workload.get_opt_seq_cost(seq), seq

    cache = {}

    def cost(seq):
        key = tuple(seq)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(seq)
        return cache[key]

    def relocate(seq, value):
        """Repeatedly apply exact best relocation improvements."""
        for _ in range(3):
            changed = False
            for source in range(n):
                txn = seq[source]
                base = seq[:source] + seq[source + 1:]
                move_value, move_seq = value, seq
                for destination in range(n):
                    candidate = base[:destination] + [txn] + base[destination:]
                    candidate_value = cost(candidate)
                    if candidate_value < move_value:
                        move_value, move_seq = candidate_value, candidate
                if move_value < value:
                    value, seq = move_value, move_seq
                    changed = True
            if not changed:
                break
        return value, seq

    # Every retained prefix is evaluated by the actual conflict simulator.  This
    # preserves alternatives whose slightly worse early prefix enables much more
    # overlap among later conflicting transactions.
    width = max(2, min(8, int(num_seqs), n))
    beam = [(cost([txn]), [txn]) for txn in range(n)]
    beam.sort(key=lambda pair: (pair[0], pair[1]))
    beam = beam[:width]

    for _ in range(1, n):
        candidates = []
        for _, prefix in beam:
            used = set(prefix)
            for txn in range(n):
                if txn not in used:
                    candidate = prefix + [txn]
                    candidates.append((cost(candidate), candidate))
        candidates.sort(key=lambda pair: (pair[0], pair[1]))
        beam = candidates[:width]

    best_cost, best_seq = float("inf"), None
    for initial_cost, initial_seq in beam[:min(4, len(beam))]:
        candidate_cost, candidate_seq = relocate(initial_seq, initial_cost)

        # Alternate exact relocation with two larger neighborhoods.  Reversing a
        # region changes many conflict directions at once, escaping local minima
        # where no individual transaction move or ordinary exchange helps.
        for _ in range(3):
            neighborhood_cost, neighborhood_seq = candidate_cost, candidate_seq

            for left in range(n - 1):
                for right in range(left + 1, n):
                    swapped = candidate_seq[:]
                    swapped[left], swapped[right] = swapped[right], swapped[left]
                    swapped_cost = cost(swapped)
                    if swapped_cost < neighborhood_cost:
                        neighborhood_cost, neighborhood_seq = swapped_cost, swapped

                    reversed_seq = (candidate_seq[:left] +
                                    candidate_seq[left:right + 1][::-1] +
                                    candidate_seq[right + 1:])
                    reversed_cost = cost(reversed_seq)
                    if reversed_cost < neighborhood_cost:
                        neighborhood_cost, neighborhood_seq = reversed_cost, reversed_seq

            if neighborhood_cost >= candidate_cost:
                break
            candidate_cost, candidate_seq = relocate(
                neighborhood_seq, neighborhood_cost
            )

        if candidate_cost < best_cost:
            best_cost, best_seq = candidate_cost, candidate_seq

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
