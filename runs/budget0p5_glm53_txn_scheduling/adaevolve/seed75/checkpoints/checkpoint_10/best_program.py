import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Multi-restart exact-cost greedy + hybrid local search (2-opt + or-opt)
    plus iterated local search (perturb-and-repolish).

    1. Construct schedules greedily: at each step append the remaining
       transaction that yields the lowest TRUE makespan (via
       workload.get_opt_seq_cost), evaluated over all candidates.
    2. Multiple restarts (deterministic spread + randomized greedy variants
       with random tie-breaking) to escape poor greedy basins.
    3. Polish the best schedule with first-improvement local search using
       both pairwise swaps (2-opt) and single-transaction reinsertion
       (or-opt) moves until local optimum or time budget expiry.
    4. Iterated local search: perturb the best schedule with random double
       swaps and re-polish, accepting improvements, until time runs out.
    """
    import time

    time_budget = 110.0  # seconds for this workload
    start_time = time.time()
    n = workload.num_txns

    def greedy(start_txn, noise=0.0):
        seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            scored = []
            for t in remaining:
                c = workload.get_opt_seq_cost(seq + [t])
                scored.append((c, random.random() if noise else t, t))
            scored.sort()
            pick = scored[0][2]
            # with noise, occasionally take 2nd-best to diversify
            if noise and len(scored) > 1 and random.random() < noise:
                pick = scored[1][2]
            seq.append(pick)
            remaining.remove(pick)
        return workload.get_opt_seq_cost(seq), seq

    def local_search(seq, cost):
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            # 2-opt swaps
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if time.time() - start_time >= time_budget:
                        return cost, seq
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
            # or-opt: move a single txn to another position
            for i in range(n):
                if time.time() - start_time >= time_budget:
                    return cost, seq
                t = seq.pop(i)
                for j in range(n):
                    if j == i:
                        continue
                    seq.insert(j, t)
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                        break
                    seq.pop(j)
                else:
                    seq.insert(i, t)
            # segment reversal: reverse block [i..j]
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if time.time() - start_time >= time_budget:
                        return cost, seq
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i:j + 1] = seq[i:j + 1][::-1]
        return cost, seq

    best_cost = None
    best_seq = None

    # Deterministic spread of starting transactions
    starts = list(range(n))
    if n > 12:
        step = max(1, n // 12)
        starts = list(range(0, n, step))[:12]

    for s in starts:
        if time.time() - start_time >= time_budget * 0.5:
            break
        c, seq = greedy(s)
        if best_cost is None or c < best_cost:
            best_cost, best_seq = c, seq

    # Randomized restarts with noise while time remains
    while time.time() - start_time < time_budget * 0.7:
        s = random.randint(0, n - 1)
        c, seq = greedy(s, noise=0.15)
        if best_cost is None or c < best_cost:
            best_cost, best_seq = c, seq

    # Polish best schedule with hybrid local search
    best_cost, best_seq = local_search(best_seq, best_cost)

    # Diversified pool of good local optima for perturbation seeding
    pool = [(best_cost, best_seq)]

    # Perturb-and-repolish (iterated local search): escape local optima
    # via random double swaps or segment reversals, seeded from the pool
    # (not only the incumbent), then re-optimize with local search.
    while time.time() - start_time < time_budget:
        if random.random() < 0.5:
            base = best_seq
        else:
            base = random.choice(pool)[1]
        cand = base[:]
        if random.random() < 0.5:
            for _ in range(2):
                i, j = random.sample(range(n), 2)
                cand[i], cand[j] = cand[j], cand[i]
        else:
            i, j = sorted(random.sample(range(n), 2))
            cand[i:j + 1] = cand[i:j + 1][::-1]
        c, cand = local_search(cand, workload.get_opt_seq_cost(cand))
        if c < best_cost:
            best_cost, best_seq = c, cand
        # maintain a small diverse pool of good local optima
        if len(pool) < 5 or c < pool[-1][0]:
            pool.append((c, cand))
            pool.sort(key=lambda x: x[0])
            pool = pool[:5]

    assert len(set(best_seq)) == workload.num_txns
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
