import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Multi-restart greedy construction with least-makespan next-txn selection,
    followed by 2-opt swap local search on the best schedule found.
    """
    n = workload.num_txns

    def greedy_construct(start_txn):
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            best_t, best_cost = None, float('inf')
            test = txn_seq + [0]
            for t in remaining:
                test[-1] = t
                c = workload.get_opt_seq_cost(test)
                if c < best_cost:
                    best_cost, best_t = c, t
            txn_seq.append(best_t)
            remaining.remove(best_t)
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    def two_opt(seq, max_rounds=30):
        cost = workload.get_opt_seq_cost(seq)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < cost:
                        cost, seq = c, cand
                        improved = True
        return cost, seq

    best_cost, best_seq = float('inf'), None
    starts = list(range(n))
    random.shuffle(starts)
    num_restarts = min(n, 25)
    for s in starts[:num_restarts]:
        c, sq = greedy_construct(s)
        if c < best_cost:
            best_cost, best_seq = c, sq

    # local search refinement on best schedule
    best_cost, best_seq = two_opt(best_seq)

    # simulated annealing refinement: low temperature, mostly-downhill,
    # accepts rare worsening moves to escape 2-opt local optima
    import math
    import time

    def simulated_annealing(seq, cost, time_budget=30.0):
        start = time.time()
        T = max(1.0, cost * 0.02)
        cooling = 0.999
        cur_seq, cur_cost = seq[:], cost
        best_sa_seq, best_sa_cost = seq[:], cost
        while T > 1e-3 and (time.time() - start) < time_budget:
            for _ in range(60):
                i = random.randrange(n)
                if random.random() < 0.5 and i < n - 1:
                    j = i + 1
                else:
                    j = random.randrange(n)
                if i == j:
                    continue
                cand = cur_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = workload.get_opt_seq_cost(cand)
                delta = c - cur_cost
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    cur_seq, cur_cost = cand, c
                    if cur_cost < best_sa_cost:
                        best_sa_cost, best_sa_seq = cur_cost, cur_seq[:]
            T *= cooling
        return best_sa_cost, best_sa_seq

    best_cost, best_seq = simulated_annealing(best_seq, best_cost)

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