import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Multi-restart greedy construction + hybrid local search.

    1. Greedy construction: start from a random transaction, then repeatedly
       append the remaining transaction that yields the lowest actual makespan
       (evaluated with workload.get_opt_seq_cost on the full prefix+candidate).
    2. Local search: alternate general pairwise-swap moves and best-insertion
       moves (remove a transaction, reinsert at its best position) until no
       improvement is found.
    3. Iterated local search: perturb the incumbent with a few random swaps
       and re-optimize, mixing in fresh greedy restarts, within a wall-clock
       budget, keeping the best schedule found. Local search kept lean
       (swaps + best-insertion) so more restarts fit in the time budget.
    """
    import time
    start_time = time.time()
    time_budget = 100.0
    n = workload.num_txns

    cost_cache = {}

    def cost_of(seq):
        key = tuple(seq)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(list(seq))
        return cost_cache[key]

    def greedy_construct():
        start_txn = random.randint(0, n - 1)
        seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        cur_cost = cost_of(seq)
        while remaining:
            best_cost = None
            best_txn = -1
            for t in remaining:
                c = cost_of(seq + [t])
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_txn = t
            seq.append(best_txn)
            remaining.remove(best_txn)
            cur_cost = best_cost
        return cur_cost, seq

    def local_search(seq, cost):
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            # general pairwise swaps (first-improvement)
            for i in range(len(seq) - 1):
                if time.time() - start_time >= time_budget:
                    return cost, seq
                for j in range(i + 1, len(seq)):
                    cand = seq.copy()
                    cand[i], cand[j] = cand[j], cand[i]
                    c = cost_of(cand)
                    if c < cost:
                        seq, cost = cand, c
                        improved = True
                        break
            # best-insertion moves
            for i in range(len(seq)):
                if time.time() - start_time >= time_budget:
                    return cost, seq
                t = seq.pop(i)
                best_c, best_j = None, i
                for j in range(len(seq) + 1):
                    cand = seq[:j] + [t] + seq[j:]
                    c = cost_of(cand)
                    if best_c is None or c < best_c:
                        best_c, best_j = c, j
                seq.insert(best_j, t)
                if best_c < cost:
                    cost = best_c
                    improved = True
        return cost, seq

    best_cost, best_seq = greedy_construct()
    best_cost, best_seq = local_search(best_seq, best_cost)
    while time.time() - start_time < time_budget:
        if random.random() < 0.75:
            # iterated local search: perturb incumbent, re-optimize
            s = best_seq.copy()
            for _ in range(random.randint(2, 5)):
                i = random.randint(0, len(s) - 1)
                j = random.randint(0, len(s) - 1)
                s[i], s[j] = s[j], s[i]
            c = cost_of(s)
            c, s = local_search(s, c)
        else:
            c, s = greedy_construct()
            c, s = local_search(s, c)
        if c < best_cost:
            best_cost, best_seq = c, s

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
