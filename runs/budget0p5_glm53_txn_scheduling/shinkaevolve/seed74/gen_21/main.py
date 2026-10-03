import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Hybrid search: multi-restart greedy + 2-opt + iterated
    ruin-and-recreate + simulated annealing polish, all under a time budget.
    """
    n = workload.num_txns
    DEADLINE = time.time() + 25.0

    def cost(seq):
        return workload.get_opt_seq_cost(seq)

    def greedy_construct(start_txn):
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            best_t, best_c = None, float('inf')
            for t in remaining:
                c = cost(txn_seq + [t])
                if c < best_c:
                    best_c, best_t = c, t
            txn_seq.append(best_t)
            remaining.remove(best_t)
        return cost(txn_seq), txn_seq

    def two_opt(seq, c, max_rounds=8):
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < DEADLINE:
            improved = False
            rounds += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    cc = cost(cand)
                    if cc < c:
                        c, seq = cc, cand
                        improved = True
                if time.time() > DEADLINE:
                    break
        return c, seq

    def ruin_recreate(seq, c, k=None):
        # remove k random txns, greedily re-insert at best position
        if k is None:
            k = random.randint(3, 6)
        seq = seq[:]
        removed = random.sample(seq, min(k, max(2, n // 4)))
        for t in removed:
            seq.remove(t)
        for t in removed:
            best_pos, best_c = 0, float('inf')
            for pos in range(len(seq) + 1):
                cand = seq[:pos] + [t] + seq[pos:]
                cc = cost(cand)
                if cc < best_c:
                    best_c, best_pos = cc, pos
            seq.insert(best_pos, t)
        return cost(seq), seq

    def annealing(seq, c, time_budget, T0):
        start = time.time()
        T = T0
        cur, cur_c = seq[:], c
        while T > 1e-3 and (time.time() - start) < time_budget and time.time() < DEADLINE:
            for _ in range(40):
                i = random.randrange(n)
                j = random.randrange(n)
                if i == j:
                    continue
                cand = cur[:]
                cand[i], cand[j] = cand[j], cand[i]
                cc = cost(cand)
                d = cc - cur_c
                if d <= 0 or random.random() < math.exp(-d / max(T, 1e-6)):
                    cur, cur_c = cand, cc
                    if cur_c < best[0]:
                        best[0], best[1] = cur_c, cur[:]
            T *= 0.995
        return best[0], best[1]

    # --- initial greedy restarts (fewer, cheaper) ---
    best = [float('inf'), None]
    starts = list(range(n))
    random.shuffle(starts)
    for s in starts[:min(n, 12)]:
        if time.time() > DEADLINE - 8.0:
            break
        c, sq = greedy_construct(s)
        if c < best[0]:
            best[0], best[1] = c, sq

    best[0], best[1] = two_opt(best[1], best[0])

    # --- iterated ruin-and-recreate with 2-opt polishing ---
    while time.time() < DEADLINE - 5.0:
        c2, sq2 = ruin_recreate(best[1], best[0])
        c2, sq2 = two_opt(sq2, c2, max_rounds=3)
        if c2 < best[0]:
            best[0], best[1] = c2, sq2
        else:
            # occasional acceptance of worsening to diversify
            if random.random() < 0.05:
                best[0], best[1] = c2, sq2

    # --- final SA polish ---
    remaining_time = max(1.0, DEADLINE - time.time())
    best[0], best[1] = annealing(best[1], best[0], remaining_time,
                                 T0=max(1.0, best[0] * 0.05))

    return best[0], best[1]

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