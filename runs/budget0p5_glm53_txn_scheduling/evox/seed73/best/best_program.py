import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Simulated annealing over transaction permutations with makespan-aware
    Metropolis acceptance, random swap/insertion moves, geometric cooling
    with periodic reheat, and best-so-far tracking. Seeded with the identity
    permutation and a quick randomized-greedy construction. This explores
    far more of the search space per second than full-neighborhood descent.
    """
    import time as _time
    import math as _math
    rng = random.Random(12345)
    deadline = _time.time() + 118  # per-workload budget; 3 workloads < 360s
    n = workload.num_txns
    cost_fn = workload.get_opt_seq_cost

    # --- quick greedy seed (one pass, sampled candidates) ---
    remaining = list(range(n))
    rng.shuffle(remaining)
    seq = []
    while remaining and _time.time() < deadline - 100:
        best_c, best_t = None, remaining[0]
        sample = remaining if len(remaining) <= 12 else rng.sample(remaining, 12)
        for t in sample:
            c = cost_fn(seq + [t])
            if best_c is None or c < best_c:
                best_c, best_t = c, t
        seq.append(best_t)
        remaining.remove(best_t)
    seq += remaining  # append leftovers if time ran out

    cur = seq
    cur_cost = cost_fn(cur)
    best, best_cost = cur[:], cur_cost

    # --- SA parameters ---
    T0 = max(1.0, cur_cost * 0.02)   # initial temperature: ~2% of makespan
    T = T0
    T_MIN = 0.05
    ALPHA = 0.9995                    # cooling rate
    REHEAT_EVERY = 20000              # periodic reheat to escape deep basins

    it = 0
    while _time.time() < deadline:
        it += 1
        # propose: random swap or random insertion (50/50)
        s = cur[:]
        if rng.random() < 0.5:
            i, j = rng.randrange(n), rng.randrange(n)
            s[i], s[j] = s[j], s[i]
        else:
            i = rng.randrange(n)
            t = s.pop(i)
            j = rng.randrange(n)
            s.insert(j, t)
        c = cost_fn(s)
        delta = c - cur_cost
        if delta <= 0 or rng.random() < _math.exp(-delta / max(T, 1e-9)):
            cur, cur_cost = s, c
            if c < best_cost:
                best_cost, best = c, s[:]
        # cool down
        T = max(T_MIN, T * ALPHA)
        if it % REHEAT_EVERY == 0:
            T = T0 * 0.5
            cur, cur_cost = best[:], best_cost  # restart from best on reheat
    return best_cost, best

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
