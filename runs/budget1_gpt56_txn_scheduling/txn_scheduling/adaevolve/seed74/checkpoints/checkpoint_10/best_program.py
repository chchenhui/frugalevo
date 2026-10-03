import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Construct simulator-scored beam schedules, then use relocation descent and kicked restarts."""
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

    # A stable seed prevents benchmark quality from depending on an unlucky
    # random sample, while still selecting a diverse subset on large instances.
    rng = random.Random(7919 + 97 * n)
    width = max(8, min(20, max(1, num_seqs) * 2))
    sample_size = n if n <= 36 else 24

    # Keep multiple low-makespan partial orders.  Prefixes are scored by the
    # actual simulator rather than an operation-count conflict proxy.
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
        beam = [(seq, rest) for _, seq, rest in expanded[:width]]

    candidates = [(cost(seq), seq) for seq, _ in beam]

    # Greedy schedules started from distinct transactions complement beam
    # pruning when an initially expensive transaction pays off later.
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
        """Use best relocation descent, traversing unseen equal-cost plateaus."""
        seq = initial[:]
        current = cost(seq)
        best_cost, best_seq = current, seq[:]
        seen = {tuple(seq)}

        # Makespan is often integer-valued, so many different conflict
        # orientations have identical cost.  Strict-only descent stops at the
        # first such plateau even when a lower basin is reachable through it.
        # Keep a small reservoir of unseen equal moves and use one only when no
        # improving relocation exists.
        for _ in range(min(3 * n, 30)):
            move_cost, move_seq = current, None
            equal_moves = []
            positions = range(n) if n <= 28 else rng.sample(range(n), min(n, 24))

            for old_pos in positions:
                reduced = seq[:old_pos] + seq[old_pos + 1:]
                destinations = range(n) if n <= 28 else rng.sample(
                    range(n), min(n, 24))
                for new_pos in destinations:
                    trial = reduced[:new_pos] + [seq[old_pos]] + reduced[new_pos:]
                    marker = tuple(trial)
                    if marker in seen:
                        continue
                    trial_cost = cost(trial)
                    if trial_cost < move_cost:
                        move_cost, move_seq = trial_cost, trial
                    elif trial_cost == current:
                        # Reservoir sampling limits memory while retaining
                        # diverse ways to cross a flat conflict landscape.
                        if len(equal_moves) < 12:
                            equal_moves.append(trial)
                        elif rng.randrange(len(equal_moves) + 1) < 12:
                            equal_moves[rng.randrange(12)] = trial

            if move_seq is not None:
                seq, current = move_seq, move_cost
            elif equal_moves:
                seq = rng.choice(equal_moves)
                current = cost(seq)
            else:
                break

            seen.add(tuple(seq))
            if current < best_cost:
                best_cost, best_seq = current, seq[:]

        # Test direct pair reversals after plateau traversal, since a newly
        # reached ordering can expose a one-edge critical-path improvement.
        for _ in range(3):
            changed = False
            for pos in range(n - 1):
                trial = best_seq[:]
                trial[pos], trial[pos + 1] = trial[pos + 1], trial[pos]
                trial_cost = cost(trial)
                if trial_cost < best_cost:
                    best_seq, best_cost, changed = trial, trial_cost, True
            if not changed:
                break
        return best_cost, best_seq

    refined = []
    for _, initial in candidates[:min(5, len(candidates))]:
        current, seq = descend(initial)
        refined.append((current, seq))
        if current < best_cost:
            best_cost, best_seq = current, seq

    # Local optima may require temporarily changing several conflict
    # orientations.  Small relocation kicks provide this escape cheaply.
    refined.sort(key=lambda item: item[0])
    for _, initial in refined[:min(2, len(refined))]:
        for _ in range(2):
            kicked = initial[:]
            for _ in range(2):
                txn = kicked.pop(rng.randrange(n))
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
