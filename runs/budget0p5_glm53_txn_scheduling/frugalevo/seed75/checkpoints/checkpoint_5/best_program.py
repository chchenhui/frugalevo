import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

def get_best_schedule(workload, num_seqs):
    """
    Refined beam-search construction with restarts, prefix-cost memoization,
    and insertion+swap local-search polish.

    Approach: multiple beam searches (width B, expansion k by true simulator
    cost of prefixes) with different random starting prefixes are run until a
    per-workload deadline. Prefix costs are cached in a dict so sibling beam
    nodes sharing a parent prefix are not re-evaluated, freeing budget for
    wider beams. The best complete permutation is then polished by insertion
    moves (remove element, reinsert elsewhere) and adjacent-range swaps,
    accepting strict improvements only. Greedy completion plus an identity
    fallback guarantee a valid permutation at all times; hard deadline checks
    keep the whole run well inside the 360s global budget.
    """
    n = workload.num_txns
    if n == 0:
        return 0, []
    deadline = time.time() + 100.0
    B, K = 12, 4
    cache = {}

    def cost(seq):
        key = tuple(seq)
        c = cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            cache[key] = c
        return c

    # Fallback: identity permutation (always valid).
    best_seq = list(range(n))
    best_cost = cost(best_seq)

    def beam_run():
        nonlocal best_seq, best_cost
        starts = random.sample(range(n), min(B, n))
        beam = [((s,), set(range(n)) - {s}) for s in starts]
        step = 1
        while step < n and beam and time.time() < deadline:
            expanded = []
            for seq, remaining in beam:
                rem = sorted(remaining)
                scored = sorted((cost(seq + (t,)), t) for t in rem)
                top = scored[:K]
                if random.random() < 0.2 and len(scored) > K:
                    top = top[:-1] + [random.choice(scored[K:])]
                for c, t in top:
                    expanded.append((c, seq + (t,), remaining - {t}))
            if not expanded:
                break
            expanded.sort(key=lambda x: x[0])
            beam = [(s, r) for _, s, r in expanded[:B]]
            step += 1

        # Greedily complete unfinished beam nodes.
        for seq, remaining in beam:
            remaining = sorted(remaining)
            while remaining:
                scored = sorted((cost(seq + (t,)), t) for t in remaining)
                seq = seq + (scored[0][1],)
                remaining.remove(scored[0][1])
                if time.time() > deadline:
                    seq = seq + tuple(remaining)
                    remaining = []
            c = cost(seq)
            if c < best_cost:
                best_cost, best_seq = c, list(seq)

    # Restart beam search while budget remains (first run always executes).
    runs = 0
    while time.time() < deadline and runs < 4:
        beam_run()
        runs += 1

    # Polish: insertion moves (move element i to position j), then pairwise
    # swaps; accept strict improvements only, under deadline checks.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = best_seq[:]
                v = cand.pop(i)
                cand.insert(j, v)
                c = cost(cand)
                if c < best_cost:
                    best_cost, best_seq = c, cand
                    improved = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = cost(cand)
                if c < best_cost:
                    best_cost, best_seq = c, cand
                    improved = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    assert len(set(best_seq)) == n
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
