import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use simulator-scored beam construction followed by insertion descent."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(seq):
        key = tuple(seq)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(seq)
        return cache[key]

    if n <= 8:
        import itertools
        best = min(itertools.permutations(range(n)), key=cost)
        return cost(best), list(best)

    # A fixed seed makes improvements reproducible while still providing
    # randomized candidate diversity for large transaction sets.
    rng = random.Random(7919 + 97 * n)
    width = max(8, min(20, max(1, num_seqs) * 2))
    sample_size = n if n <= 36 else 24

    # Retain competing low-cost prefixes so a locally attractive early
    # conflict orientation cannot force the entire schedule into a poor basin.
    beam = [([], tuple(range(n)))]
    for _ in range(n):
        expanded = []
        for prefix, remaining in beam:
            choices = remaining
            if len(remaining) > sample_size:
                choices = tuple(rng.sample(list(remaining), sample_size))
            for txn in choices:
                seq = prefix + [txn]
                rest = tuple(item for item in remaining if item != txn)
                expanded.append((cost(seq), seq, rest))

        expanded.sort(key=lambda item: item[0])
        beam = []
        seen = set()
        for _, seq, rest in expanded:
            marker = tuple(seq)
            if marker not in seen:
                beam.append((seq, rest))
                seen.add(marker)
                if len(beam) == width:
                    break

    candidates = [(cost(seq), seq) for seq, _ in beam]

    # Add independent greedy schedules, which supply useful diversity when
    # prefix-only beam pruning rejects a transaction that pays off later.
    starts = list(range(n))
    rng.shuffle(starts)
    for start in starts[:max(1, min(n, num_seqs))]:
        seq = [start]
        remaining = [item for item in range(n) if item != start]
        while remaining:
            choices = remaining if len(remaining) <= sample_size else rng.sample(
                remaining, sample_size)
            txn = min(choices, key=lambda item: cost(seq + [item]))
            seq.append(txn)
            remaining.remove(txn)
        candidates.append((cost(seq), seq))

    candidates.sort(key=lambda item: item[0])
    best_cost, best_seq = candidates[0]

    def descend(initial):
        """Perform simulator-scored relocation and adjacent-swap descent."""
        seq = initial[:]
        current = cost(seq)

        # For small workloads inspect the complete insertion neighborhood.
        # For larger workloads, inspect a substantially wider sampled
        # neighborhood than the construction phase without excessive cost.
        for _ in range(min(n, 8)):
            move_cost, move_seq = current, None
            if n <= 28:
                positions = range(n)
                destination_count = n
            else:
                positions = rng.sample(range(n), min(n, 24))
                destination_count = min(n, 24)

            for old_pos in positions:
                reduced = seq[:old_pos] + seq[old_pos + 1:]
                if n <= 28:
                    destinations = range(n)
                else:
                    destinations = rng.sample(range(n), destination_count)
                for new_pos in destinations:
                    trial = reduced[:new_pos] + [seq[old_pos]] + reduced[new_pos:]
                    trial_cost = cost(trial)
                    if trial_cost < move_cost:
                        move_cost, move_seq = trial_cost, trial

            if move_seq is None:
                break
            seq, current = move_seq, move_cost

        # Adjacent reversals are inexpensive and can improve an insertion
        # optimum by changing one directly conflicting transaction pair.
        for _ in range(3):
            changed = False
            for pos in range(n - 1):
                trial = seq[:]
                trial[pos], trial[pos + 1] = trial[pos + 1], trial[pos]
                trial_cost = cost(trial)
                if trial_cost < current:
                    seq, current, changed = trial, trial_cost, True
            if not changed:
                break
        return current, seq

    # Refine more independent constructions.  Each move is evaluated with the
    # simulator's actual makespan, rather than a proxy such as transaction
    # length or write count.
    refined = []
    for _, initial in candidates[:min(5, len(candidates))]:
        current, seq = descend(initial)
        refined.append((current, seq))
        if current < best_cost:
            best_cost, best_seq = current, seq

    # A local optimum can require temporarily worsening a conflict direction.
    # Apply small insertion kicks to the best distinct local optima, then run
    # descent again; this is an inexpensive iterated-local-search escape step.
    refined.sort(key=lambda item: item[0])
    for _, initial in refined[:min(2, len(refined))]:
        for _ in range(2):
            kicked = initial[:]
            for _ in range(2):
                old_pos = rng.randrange(n)
                txn = kicked.pop(old_pos)
                kicked.insert(rng.randrange(n), txn)
            current, seq = descend(kicked)
            if current < best_cost:
                best_cost, best_seq = current, seq

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
