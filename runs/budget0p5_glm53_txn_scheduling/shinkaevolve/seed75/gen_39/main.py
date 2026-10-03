import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

def get_best_schedule(workload, num_seqs, time_budget=25.0):
    """
    Randomized greedy restarts with a per-restart exploration schedule
    (sample_rate 0.95 -> 0.85, num_samples 6 -> 10), cost caching, and
    swap + reinsertion local search on the best greedy seeds.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    start_time = time.time()
    deadline = start_time + time_budget

    cost_cache = {}
    def cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            if len(cost_cache) < 2000000:
                cost_cache[key] = c
        return c

    def greedy_cost(num_samples, sample_rate, rng):
        # randomized greedy using true makespan cost
        start_txn = rng.randrange(n)
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            if rng.random() > sample_rate and len(remaining) > 1:
                idx = rng.randint(0, len(remaining) - 1)
                txn_seq.append(remaining.pop(idx))
                continue
            num = min(num_samples, len(remaining))
            sampled = rng.sample(remaining, num)
            best_cost = None
            best_txn = sampled[0]
            for t in sampled:
                c = cost(txn_seq + [t])
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_txn = t
            txn_seq.append(best_txn)
            remaining.remove(best_txn)
        return cost(txn_seq), txn_seq

    def local_search(seq):
        cur = seq.copy()
        cur_cost = cost(cur)
        while time.time() < deadline:
            improved = False
            # pairwise swaps
            for i in range(len(cur)):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, len(cur)):
                    cand = cur.copy()
                    cand[i], cand[j] = cand[j], cand[i]
                    c = cost(cand)
                    if c < cur_cost:
                        cur, cur_cost = cand, c
                        improved = True
            # reinsertion moves
            for i in range(len(cur)):
                if time.time() >= deadline:
                    break
                t = cur.pop(i)
                placed = False
                for j in range(len(cur) + 1):
                    cand = cur.copy()
                    cand.insert(j, t)
                    c = cost(cand)
                    if c < cur_cost:
                        cur, cur_cost = cand, c
                        improved = True
                        placed = True
                        break
                if not placed:
                    cur.insert(i, t)
            if not improved:
                break
        return cur_cost, cur

    best_cost, best_seq = None, None
    top_candidates = []
    max_restarts = 20
    restart = 0
    rng = random.Random(12345)

    while time.time() < deadline:
        frac = min(1.0, restart / max_restarts)
        sample_rate = 0.95 - 0.10 * frac
        num_samples = 6 if restart < max_restarts / 2 else 10
        c, seq = greedy_cost(num_samples, sample_rate, rng)
        if best_cost is None or c < best_cost:
            best_cost, best_seq = c, seq
        top_candidates.append((c, seq))
        top_candidates.sort(key=lambda x: x[0])
        if len(top_candidates) > 3:
            top_candidates.pop()
        restart += 1
        if restart >= 60:
            break

    # deep local search from the best polished greedy seed
    if top_candidates:
        c, seq = local_search(list(top_candidates[0][1]))
        if c < best_cost:
            best_cost, best_seq = c, seq

    if best_seq is None:
        best_seq = list(range(n))
        best_cost = cost(best_seq)
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