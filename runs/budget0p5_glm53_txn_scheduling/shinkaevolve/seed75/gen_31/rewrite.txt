import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

def get_best_schedule(workload, num_seqs, time_budget=25.0):
    """
    Multi-restart greedy construction with restart-scheduled exploration rate
    (0.95 -> 0.85) and sample count (8 -> 12), followed by annealing insertion
    local search, within a time budget.
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

    def greedy_construct(sample_rate, num_samples, seed):
        rng = random.Random(seed)
        seq = [rng.randrange(n)]
        remaining = [t for t in range(n) if t != seq[0]]
        while remaining:
            if rng.random() > sample_rate:
                # occasional random move for diversification
                idx = rng.randint(0, len(remaining) - 1)
                seq.append(remaining.pop(idx))
                continue
            k = min(len(remaining), num_samples)
            candidates = remaining if n <= 40 else rng.sample(remaining, k)
            best_t, best_c = None, None
            for t in candidates:
                c = cost(tuple(seq) + (t,))
                if best_c is None or c < best_c:
                    best_c, best_t = c, t
            seq.append(best_t)
            remaining.remove(best_t)
        return seq

    def insertion_search(seq, restart_time):
        cur = list(seq)
        cur_cost = cost(cur)
        best_cost = cur_cost
        best = list(cur)
        temp = max(1.0, cur_cost * 0.02)
        alpha = 0.98
        stall = 0
        while time.time() < deadline and time.time() < restart_time:
            improved = False
            # one full insertion descent pass
            for i in range(n):
                if time.time() >= deadline or time.time() >= restart_time:
                    break
                t = cur.pop(i)
                base = cur
                best_j = i
                best_c = cost(base[:i] + [t] + base[i:])
                # try nearby + a few random positions
                positions = set(range(max(0, i - 10), min(n, i + 11)))
                for _ in range(6):
                    positions.add(random.randint(0, n - 1))
                for j in sorted(positions):
                    if j == i:
                        continue
                    cand = base[:j] + [t] + base[j:]
                    c = cost(cand)
                    if c < best_c:
                        best_c = c
                        best_j = j
                cur = base[:best_j] + [t] + base[best_j:]
                cur_cost = best_c
                if cur_cost < best_cost - 1e-9:
                    best_cost = cur_cost
                    best = list(cur)
                    improved = True
            temp *= alpha
            if not improved:
                stall += 1
                # annealing escape: random insertions, accept worsening w.p. temp
                for _ in range(5):
                    if time.time() >= deadline or time.time() >= restart_time:
                        break
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    t = cur.pop(i)
                    cur.insert(j, t)
                    c = cost(cur)
                    if c < cur_cost or c - cur_cost <= temp:
                        cur_cost = c
                        if c < best_cost:
                            best_cost = c
                            best = list(cur)
                    else:
                        t = cur.pop(j)
                        cur.insert(i, t)
                if stall > 6:
                    cur = list(best)
                    for _ in range(3):
                        i = random.randint(0, n - 1)
                        j = random.randint(0, n - 1)
                        t = cur.pop(i)
                        cur.insert(j, t)
                    cur_cost = cost(cur)
                    stall = 0
        return best_cost, best

    # ---- Main loop: restart-scheduled greedy rates 0.95 -> 0.85 ----
    best_cost = None
    best_seq = None
    restart = 0
    max_restarts = 20
    while time.time() < deadline:
        frac = min(1.0, restart / max_restarts)
        sample_rate = 0.95 - 0.10 * frac
        num_samples = 8 if restart < 10 else 12
        # budget each restart a slice of remaining time
        remaining_time = deadline - time.time()
        slice_time = max(0.5, remaining_time / 2.0)
        restart_deadline = min(deadline, time.time() + slice_time)

        seq = greedy_construct(sample_rate, num_samples, seed=restart * 7919 + 13)
        c, s = insertion_search(seq, restart_deadline)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_seq = list(s)
        restart += 1
        if restart >= max_restarts and time.time() < deadline:
            # continue refining best with extra restarts at 0.85 rate
            restart = max_restarts - 1

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