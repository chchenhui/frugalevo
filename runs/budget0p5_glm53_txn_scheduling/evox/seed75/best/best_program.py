import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Greedy construction + fast first-improvement local search + iterated
    local search (ILS) on the true makespan.

    1. Greedy: append the transaction minimizing the actual makespan
       (workload.get_opt_seq_cost), sampling candidates when many remain.
    2. First-improvement hill climbing over insertion moves (fast: accepts
       the first improving move found, saving expensive evaluations).
    3. ILS: perturb the incumbent with a few random relocations, re-climb,
       keep improvements; repeat until the time budget is exhausted.
    """
    import time
    n = workload.num_txns
    deadline = time.time() + 80.0

    def greedy_once(num_samples):
        start_txn = random.randint(0, n - 1)
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            candidates = remaining
            if len(candidates) > num_samples:
                candidates = random.sample(remaining, num_samples)
            best_cost = None
            best_txn = None
            for t in candidates:
                cost = workload.get_opt_seq_cost(txn_seq + [t])
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_txn = t
            txn_seq.append(best_txn)
            remaining.remove(best_txn)
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    def local_search(seq, cost, plateau_budget=6):
        """First-improvement insertion + swap hill climbing with limited
        plateau walking; restarts the scan after each accepted move so
        earlier positions get re-evaluated (deeper descent per pass)."""
        seq = list(seq)
        improved = True
        plateaus = 0
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    break
                # --- insertion moves ---
                t = seq.pop(i)
                for j in range(n):
                    if j == i:
                        continue
                    seq.insert(j, t)
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost or (c == cost and plateaus < plateau_budget
                                    and random.random() < 0.3):
                        if c < cost:
                            plateaus = 0
                        else:
                            plateaus += 1
                        cost = c
                        improved = True
                        break  # accept immediately, re-scan from scratch
                    del seq[j]
                else:
                    seq.insert(i, t)
                    # --- swap moves ---
                    for j in range(n):
                        if j == i:
                            continue
                        seq[i], seq[j] = seq[j], seq[i]
                        c = workload.get_opt_seq_cost(seq)
                        if c < cost or (c == cost and plateaus < plateau_budget
                                        and random.random() < 0.3):
                            if c < cost:
                                plateaus = 0
                            else:
                                plateaus += 1
                            cost = c
                            improved = True
                            break
                        seq[i], seq[j] = seq[j], seq[i]
                    continue
                break  # improving move applied; restart outer scan
        return cost, seq

    def perturb(seq, k):
        """Mix of random relocations, swaps, and one segment reversal for
        stronger diversification."""
        s = list(seq)
        if len(s) < 3:
            return s
        for _ in range(k):
            r = random.random()
            if r < 0.5:
                i = random.randrange(len(s))
                j = random.randrange(len(s))
                t = s.pop(i)
                s.insert(j, t)
            else:
                i = random.randrange(len(s) - 1)
                j = random.randrange(i + 1, len(s))
                s[i], s[j] = s[j], s[i]
        if random.random() < 0.4:
            i = random.randrange(len(s) - 2)
            j = random.randrange(i + 2, min(i + 12, len(s)) + 1)
            s[i:j] = reversed(s[i:j])
        return s

    best_cost, best_seq = None, None
    for r in range(max(1, num_seqs)):
        if time.time() >= deadline:
            break
        c, s = greedy_once(12)
        c, s = local_search(s, c)
        if best_cost is None or c < best_cost:
            best_cost, best_seq = c, s

    # ILS with adaptive perturbation strength and occasional greedy restarts
    strength, stagnant = 3, 0
    while time.time() < deadline:
        if stagnant >= 30:
            c, s = greedy_once(12)
            stagnant, strength = 0, 3
        else:
            s = perturb(best_seq, random.randint(strength, strength + 2))
        c = workload.get_opt_seq_cost(s)
        c, s = local_search(s, c)
        if c < best_cost:
            best_cost, best_seq = c, s
            stagnant, strength = 0, 3
        else:
            stagnant += 1
            if stagnant % 8 == 0:
                strength = min(strength + 2, 10)
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
