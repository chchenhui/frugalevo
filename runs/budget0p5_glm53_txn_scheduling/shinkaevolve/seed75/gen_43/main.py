import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Phase 1: many cheap sampled-greedy restarts (no refinement).
    Phase 2: keep a top-3 pool of solutions that are pairwise positionally
             diverse (<70% overlap) so the refinement starts from a robust basin.
    Phase 3: single deep iterated local search (perturb + insertion/swap
             descent) from the best pooled solution, with a cost cache.
    """
    import time
    deadline = time.time() + 100

    cost_cache = {}
    def cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            cost_cache[key] = c
        return c

    def get_greedy_cost_sampled(num_samples, sample_rate):
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)]
        remaining_txns.remove(start_txn)
        for i in range(0, workload.num_txns - 1):
            min_cost = 100000
            min_txn = -1
            holdout_txns = []
            done = False

            sample = random.random()
            if sample > sample_rate:
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns[idx]
                txn_seq.append(t)
                remaining_txns.pop(idx)
                continue

            for j in range(0, num_samples):
                idx = 0
                if len(remaining_txns) > 1:
                    idx = random.randint(0, len(remaining_txns) - 1)
                else:
                    done = True
                t = remaining_txns[idx]
                holdout_txns.append(remaining_txns.pop(idx))
                test_seq = txn_seq + [t]
                c = cost(test_seq)
                if c < min_cost:
                    min_cost = c
                    min_txn = t
                if done:
                    break
            assert(min_txn != -1)
            txn_seq.append(min_txn)
            holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)

        assert len(set(txn_seq)) == workload.num_txns
        overall_cost = cost(txn_seq)
        return overall_cost, txn_seq

    def overlap(a, b):
        same = sum(1 for x, y in zip(a, b) if x == y)
        return same / float(len(a))

    # ---- Phase 1 & 2: greedy restarts into a diverse top-3 pool ----
    pool = []  # list of (cost, seq), sorted ascending, max 3
    pool_deadline = deadline - 45  # reserve 45s for deep refinement
    while time.time() < pool_deadline:
        c, seq = get_greedy_cost_sampled(10, 0.9)
        # diversity check vs existing pool members
        if all(overlap(seq, s) < 0.7 for _, s in pool):
            pool.append((c, seq))
            pool.sort(key=lambda x: x[0])
            del pool[3:]
        if len(pool) == 3 and time.time() > pool_deadline - 25:
            break

    if not pool:
        c, seq = get_greedy_cost_sampled(10, 1.0)
        pool = [(c, seq)]

    # pick pool entry with lowest diversity-weighted rank
    def pool_score(idx):
        c, s = pool[idx]
        div_bonus = sum(1.0 - overlap(s, s2) for k, (_, s2) in enumerate(pool) if k != idx)
        return c - 2.0 * (div_bonus / max(1, len(pool) - 1))
    start = min(range(len(pool)), key=pool_score)
    best_cost, best_seq = pool[start]
    best_seq = list(best_seq)

    # ---- Phase 3: deep iterated local search ----
    n = len(best_seq)

    def descend(seq, cur_cost):
        seq = list(seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline:
                    return seq, cur_cost
                # insertion moves (limited window + a few random targets)
                t = seq.pop(i)
                targets = set(range(max(0, i - 10), min(n, i + 11)))
                for _ in range(6):
                    targets.add(random.randint(0, n - 1))
                best_j, best_c = i, cur_cost
                for j in sorted(targets):
                    if j == i:
                        continue
                    cand = seq[:j] + [t] + seq[j:]
                    c = cost(cand)
                    if c < best_c - 1e-9:
                        best_c, best_j = c, j
                seq = seq[:best_j] + [t] + seq[best_j:]
                if best_j != i:
                    cur_cost = best_c
                    improved = True
            # swap moves (nearby window)
            for i in range(n):
                if time.time() >= deadline:
                    return seq, cur_cost
                for j in range(i + 1, min(n, i + 12)):
                    seq[i], seq[j] = seq[j], seq[i]
                    c = cost(seq)
                    if c < cur_cost - 1e-9:
                        cur_cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
        return seq, cur_cost

    def perturb(seq, strength):
        seq = list(seq)
        for _ in range(strength):
            i = random.randint(0, len(seq) - 1)
            j = random.randint(0, len(seq) - 1)
            t = seq.pop(i)
            seq.insert(j, t)
        return seq

    best_seq, best_cost = descend(best_seq, best_cost)
    while time.time() < deadline:
        cand = perturb(best_seq, random.randint(2, 5))
        cand, cand_cost = descend(cand, cost(cand))
        if cand_cost < best_cost - 1e-9:
            best_cost = cand_cost
            best_seq = cand

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