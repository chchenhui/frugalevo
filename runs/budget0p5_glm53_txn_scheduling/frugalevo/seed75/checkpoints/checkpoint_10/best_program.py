import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

def get_best_schedule(workload, num_seqs):
    """
    Beam-seeded local search over complete permutations with 2-opt descents
    and double-bridge restarts.

    Approach: a strong beam-search seed (width B, expansion by true simulator
    prefix cost, memoized) is polished by a first-improvement descent over the
    complete permutation using three move families — relocation (pop and
    reinsert), pairwise swap, and 2-opt segment reversal — each evaluated with
    the true `get_opt_seq_cost` and accepted only on strict improvement, so
    descents are monotone. The 2-opt family adds structural moves the plain
    relocation+swap descent lacks, correcting misordered blocks of hot-key
    transactions that greedy construction places badly. Restart mechanism:
    double-bridge perturbation (four-cut reassembly) of the incumbent, then
    re-descent; accept only if it beats the incumbent. Hard per-workload
    deadline (~100s), tuple-key cost cache, and identity fallback guarantee
    validity and keep all three workloads inside 360s.
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

    # Beam construction to seed the local search.
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

    # Greedily complete unfinished beam nodes; track the best seed.
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

    def descend(seq, cur_cost):
        """First-improvement descent: relocations, swaps, 2-opt reversals."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Relocation moves.
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    cand = seq[:]
                    v = cand.pop(i)
                    cand.insert(j, v)
                    c = cost(cand)
                    if c < cur_cost:
                        seq, cur_cost = cand, c
                        improved = True
                        break
                    if time.time() > deadline:
                        break
                if improved or time.time() > deadline:
                    break
            if time.time() > deadline:
                break
            # Pairwise swaps.
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = seq[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = cost(cand)
                    if c < cur_cost:
                        seq, cur_cost = cand, c
                        improved = True
                        break
                    if time.time() > deadline:
                        break
                if improved or time.time() > deadline:
                    break
            if time.time() > deadline:
                break
            # 2-opt segment reversals.
            for i in range(n - 1):
                for j in range(i + 2, n):
                    cand = seq[:]
                    cand[i:j + 1] = cand[i:j + 1][::-1]
                    c = cost(cand)
                    if c < cur_cost:
                        seq, cur_cost = cand, c
                        improved = True
                        break
                    if time.time() > deadline:
                        break
                if improved or time.time() > deadline:
                    break
        return seq, cur_cost

    best_seq, best_cost = descend(best_seq, best_cost)

    # Restart loop: double-bridge perturbation of the incumbent, re-descend.
    while time.time() < deadline:
        seq = best_seq[:]
        if n >= 4:
            a, b, c2 = sorted(random.sample(range(1, n), 3))
            seq = (seq[:a] + seq[b:c2] + seq[a:b] + seq[c2:])
        else:
            i = random.randrange(n - 1)
            j = random.randrange(i + 1, n)
            seq[i:j + 1] = seq[i:j + 1][::-1]
        seq, c = descend(seq, cost(seq))
        if c < best_cost:
            best_cost, best_seq = c, seq[:]

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
