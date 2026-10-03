import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using greedy construction + 2-opt descent +
    simulated-annealing iterated local search.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import time
    import math

    n = workload.num_txns

    def cost_of(seq):
        return workload.get_opt_seq_cost(seq)

    def get_full_greedy(perturb_rate):
        start_txn = random.randint(0, n - 1)
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            min_cost = None
            min_txn = -1
            if random.random() < perturb_rate:
                idx = random.randint(0, len(remaining) - 1)
                min_txn = remaining[idx]
                min_cost = cost_of(txn_seq + [min_txn])
            else:
                for t in remaining:
                    c = cost_of(txn_seq + [t])
                    if min_cost is None or c < min_cost:
                        min_cost = c
                        min_txn = t
            txn_seq.append(min_txn)
            remaining.remove(min_txn)
        return cost_of(txn_seq), txn_seq

    def two_opt_descent(seq, cost, deadline):
        # first-improvement 2-opt: reverse segment [i:j]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if time.time() >= deadline:
                        return cost, seq
                    cand = seq[:]
                    cand[i:j] = reversed(cand[i:j])
                    c = cost_of(cand)
                    if c < cost:
                        seq, cost = cand, c
                        improved = True
                        break
                else:
                    continue
                break
        return cost, seq

    def local_search(seq, cost, deadline):
        # 2-opt plus random relocation/swap descent
        cost, seq = two_opt_descent(seq, cost, deadline)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for _ in range(150):
                if time.time() >= deadline:
                    break
                cand = seq[:]
                r = random.random()
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if r < 0.4 and i != j:
                    t = cand.pop(i)
                    cand.insert(j, t)
                elif i != j:
                    cand[i], cand[j] = cand[j], cand[i]
                else:
                    continue
                c = cost_of(cand)
                if c < cost:
                    seq, cost = cand, c
                    improved = True
        return cost, seq

    def perturb(seq):
        # ruin-and-recreate: remove segment, greedy reinsertion
        k = random.randint(3, min(8, n - 1))
        start = random.randint(0, n - k)
        removed = seq[start:start + k]
        cand = seq[:start] + seq[start + k:]
        for t in removed:
            best_c = None
            best_pos = 0
            for pos in range(len(cand) + 1):
                trial = cand[:pos] + [t] + cand[pos:]
                c = cost_of(trial)
                if best_c is None or c < best_c:
                    best_c = c
                    best_pos = pos
                if time.time() >= deadline:
                    break
            cand = cand[:best_pos] + [t] + cand[best_pos:]
        return cand

    time_budget = 20.0
    deadline = time.time() + time_budget

    best_cost, best_seq = get_full_greedy(0.05)
    best_cost, best_seq = local_search(best_seq, best_cost, deadline)

    # simulated annealing over the incumbent with occasional restarts
    cur_cost, cur_seq = best_cost, best_seq[:]
    T = 2.0
    alpha = 0.985
    since_best = 0
    while time.time() < deadline:
        if random.random() < 0.5:
            cand = perturb(cur_seq)
        else:
            cand = cur_seq[:]
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                if random.random() < 0.5:
                    t = cand.pop(i)
                    cand.insert(j, t)
                else:
                    cand[i], cand[j] = cand[j], cand[i]
        c = cost_of(cand)
        if c < cur_cost or random.random() < math.exp(-(c - cur_cost) / max(T, 1e-9)):
            cur_seq, cur_cost = cand, c
            if c < best_cost:
                best_cost, best_seq = c, cand[:]
                since_best = 0
        else:
            since_best += 1
        T *= alpha
        if T < 0.05 or since_best > 300:
            # reheat from best
            T = 1.0
            cur_cost, cur_seq = best_cost, best_seq[:]
            since_best = 0
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