import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

def get_best_schedule(workload, num_seqs, time_budget=25.0):
    """
    Sampled greedy restarts, each immediately polished by a cheap
    best-improvement swap sweep, then refined by full insertion local search.
    Restarts repeat until a time deadline; best sequence overall is returned.
    """
    n = workload.num_txns
    deadline = time.time() + time_budget

    cost_cache = {}
    def cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            if len(cost_cache) < 2000000:
                cost_cache[key] = c
        return c

    # ---- Phase 1: sampled greedy construction ----
    def greedy_sampled(sample_rate=0.9):
        start_txn = random.randint(0, n - 1)
        seq = [start_txn]
        remaining = [t for t in range(n) if t != start_txn]
        while remaining:
            if time.time() >= deadline:
                # finish arbitrarily with remaining order
                seq.extend(remaining)
                remaining = []
                break
            if random.random() > sample_rate:
                # diversity move: append a random txn without evaluation
                idx = random.randint(0, len(remaining) - 1)
                seq.append(remaining.pop(idx))
                continue
            best_t, best_c = None, None
            for t in remaining:
                c = cost(tuple(seq) + (t,))
                if best_c is None or c < best_c:
                    best_c, best_t = c, t
            seq.append(best_t)
            remaining.remove(best_t)
        return seq, cost(seq)

    # ---- Phase 2a: cheap full best-improvement swap sweep ----
    def swap_sweep(seq, cur_cost):
        seq = list(seq)
        improved = True
        while improved:
            improved = False
            best_delta_i, best_delta_j = -1, -1
            best_c = cur_cost
            for i in range(n):
                if time.time() >= deadline:
                    return seq, cur_cost
                for j in range(i + 1, n):
                    seq[i], seq[j] = seq[j], seq[i]
                    c = cost(seq)
                    if c < best_c - 1e-9:
                        best_c = c
                        best_delta_i, best_delta_j = i, j
                    seq[i], seq[j] = seq[j], seq[i]
            if best_delta_i >= 0:
                seq[best_delta_i], seq[best_delta_j] = \
                    seq[best_delta_j], seq[best_delta_i]
                cur_cost = best_c
                improved = True
        return seq, cur_cost

    # ---- Phase 2b: full insertion local search (first-improvement) ----
    def insertion_search(seq, cur_cost, max_rounds=25):
        seq = list(seq)
        rounds = 0
        improved = True
        while improved and rounds < max_rounds and time.time() < deadline:
            improved = False
            rounds += 1
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    return seq, cur_cost
                t = seq.pop(i)
                base = seq
                for j in range(n):
                    if j == i:
                        continue
                    cand = base[:j] + [t] + base[j:]
                    c = cost(cand)
                    if c < cur_cost - 1e-9:
                        cur_cost = c
                        seq = cand
                        improved = True
                        break
                else:
                    seq = base[:i] + [t] + base[i:]
        return seq, cur_cost

    # ---- Phase 3: restart loop within time budget ----
    best_cost = float('inf')
    best_seq = None
    restart = 0
    while time.time() < deadline:
        restart += 1
        seq, c = greedy_sampled(0.9)
        # cheap swap polish before deep insertion search
        seq, c = swap_sweep(seq, c)
        seq, c = insertion_search(seq, c)
        if c < best_cost:
            best_cost = c
            best_seq = list(seq)

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