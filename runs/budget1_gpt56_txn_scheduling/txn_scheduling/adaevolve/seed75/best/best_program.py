import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use exact-cost beam/local search followed by deterministic randomized
    destroy-and-greedy-repair large-neighborhood search."""
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
        """Perform bounded exact best-insertion descent on a full permutation."""
        for _ in range(3):
            improved = False
            for source in range(n):
                txn = seq[source]
                base = seq[:source] + seq[source + 1:]
                move_cost, move_seq = value, seq
                for destination in range(n):
                    trial = base[:destination] + [txn] + base[destination:]
                    trial_cost = cost(trial)
                    if trial_cost < move_cost:
                        move_cost, move_seq = trial_cost, trial
                if move_cost < value:
                    value, seq, improved = move_cost, move_seq, True
            if not improved:
                break
        return value, seq

    width = max(2, min(8, int(num_seqs), n))
    beam = [(cost([txn]), [txn]) for txn in range(n)]
    beam.sort(key=lambda x: (x[0], x[1]))
    beam = beam[:width]
    for _ in range(1, n):
        expanded = []
        for _, prefix in beam:
            used = set(prefix)
            for txn in range(n):
                if txn not in used:
                    trial = prefix + [txn]
                    expanded.append((cost(trial), trial))
        expanded.sort(key=lambda x: (x[0], x[1]))
        beam = expanded[:width]

    best_cost, best_seq = float("inf"), None
    for initial_cost, initial_seq in beam[:min(4, len(beam))]:
        value, seq = relocate(initial_seq, initial_cost)
        for _ in range(2):
            move_cost, move_seq = value, seq
            for left in range(n - 1):
                for right in range(left + 1, n):
                    trial = seq[:]
                    trial[left], trial[right] = trial[right], trial[left]
                    trial_cost = cost(trial)
                    if trial_cost < move_cost:
                        move_cost, move_seq = trial_cost, trial
                    trial = seq[:left] + seq[left:right + 1][::-1] + seq[right + 1:]
                    trial_cost = cost(trial)
                    if trial_cost < move_cost:
                        move_cost, move_seq = trial_cost, trial
            if move_cost >= value:
                break
            value, seq = relocate(move_seq, move_cost)
        if value < best_cost:
            best_cost, best_seq = value, seq

    # Destroying several positions changes many conflict directions together.
    # Every repair decision remains based solely on the exact simulator cost.
    rng = random.Random(104729 * n + int(num_seqs))
    seed_seq, seed_cost = best_seq[:], best_cost
    attempts = max(6, min(16, 2 * int(num_seqs) + 2))
    maximum = max(2, min(n - 1, max(3, n // 3)))

    for attempt in range(attempts):
        remove_count = 2 + (attempt % max(1, maximum - 1))
        remove_count = min(remove_count, n - 1)
        removed_positions = rng.sample(range(n), remove_count)
        removed = [seed_seq[pos] for pos in removed_positions]
        removed_set = set(removed)
        partial = [txn for txn in seed_seq if txn not in removed_set]
        rng.shuffle(removed)

        # Random reinsertion order explores distinct orientations.  Tied exact
        # insertion costs are deliberately randomized rather than index-biased.
        for txn in removed:
            candidates = []
            lowest = float("inf")
            for position in range(len(partial) + 1):
                trial = partial[:position] + [txn] + partial[position:]
                trial_cost = cost(trial)
                if trial_cost < lowest:
                    lowest, candidates = trial_cost, [trial]
                elif trial_cost == lowest:
                    candidates.append(trial)
            partial = candidates[rng.randrange(len(candidates))]

        repaired_cost = cost(partial)
        if repaired_cost < best_cost:
            best_cost, best_seq = repaired_cost, partial[:]

        # A deterministic, limited uphill walk provides a different source
        # permutation for later large repairs without sacrificing the incumbent.
        if repaired_cost <= seed_cost or attempt % 3 == 2:
            seed_seq, seed_cost = partial[:], repaired_cost

    best_cost, best_seq = relocate(best_seq, best_cost)
    return workload.get_opt_seq_cost(best_seq), best_seq

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
