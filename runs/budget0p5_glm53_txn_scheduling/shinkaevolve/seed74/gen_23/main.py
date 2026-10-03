import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using greedy cost sampling + ruin-and-recreate perturbation.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns

    def greedy_construct(sample_rate=1.0, num_samples=10):
        start = random.randint(0, n - 1)
        seq = [start]
        remaining = [x for x in range(n) if x != start]
        while remaining:
            if random.random() > sample_rate:
                idx = random.randint(0, len(remaining) - 1)
                seq.append(remaining.pop(idx))
                continue
            best_cost = None
            best_t = None
            cand_pool = random.sample(remaining, min(num_samples, len(remaining)))
            for t in cand_pool:
                c = workload.get_opt_seq_cost(seq + [t])
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_t = t
            seq.append(best_t)
            remaining.remove(best_t)
        return workload.get_opt_seq_cost(seq), seq

    def ruin_and_recreate(seq, k):
        # remove a random contiguous block of k txns, reinsert greedily at best position
        if len(seq) <= k:
            return None
        start = random.randint(0, len(seq) - k - 1)
        removed = seq[start:start + k]
        base = seq[:start] + seq[start + k:]
        for t in removed:
            best_cost = None
            best_pos = None
            for pos in range(len(base) + 1):
                cand = base[:pos] + [t] + base[pos:]
                c = workload.get_opt_seq_cost(cand)
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_pos = pos
            base.insert(best_pos, t)
        return workload.get_opt_seq_cost(base), base

    deadline = time.time() + 25.0

    # initial solutions
    best_cost, best_seq = greedy_construct()
    # greedy full-cost deterministic start (first txn in order)
    seq = []
    remaining = list(range(n))
    while remaining:
        best_c = None
        best_t = None
        for t in remaining:
            c = workload.get_opt_seq_cost(seq + [t])
            if best_c is None or c < best_c:
                best_c = c
                best_t = t
        seq.append(best_t)
        remaining.remove(best_t)
        if time.time() > deadline:
            break
    c = workload.get_opt_seq_cost(seq)
    if len(set(seq)) == n and c < best_cost:
        best_cost, best_seq = c, seq

    # iterate ruin-and-recreate on the best, with occasional fresh greedy restarts
    while time.time() < deadline:
        if random.random() < 0.25:
            c, s = greedy_construct(0.9, 8)
        else:
            res = ruin_and_recreate(best_seq, random.randint(3, 5))
            if res is None:
                continue
            c, s = res
        if c < best_cost:
            best_cost, best_seq = c, s

    # adjacent-swap hill climbing
    improved = True
    while improved:
        improved = False
        for i in range(len(best_seq) - 1):
            if time.time() > deadline:
                break
            cand = best_seq.copy()
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = workload.get_opt_seq_cost(cand)
            if c < best_cost:
                best_cost, best_seq = c, cand
                improved = True
                break

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