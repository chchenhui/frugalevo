import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using greedy cost sampling strategy.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    def get_greedy_cost_sampled(num_samples, sample_rate):
        # greedy with random starting point
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)]
        remaining_txns.remove(start_txn)
        running_cost = workload.txns[start_txn][0][3]
        # min_costs = []
        # key_map, total_cost = workload.get_incremental_seq_cost(start_txn, {}, 0)
        for i in range(0, workload.num_txns - 1):
            min_cost = 100000 # MAX
            min_relative_cost = 10
            min_txn = -1
            # min_index = 0
            holdout_txns = []
            done = False
            key_maps = []

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
                if workload.debug:
                    print(remaining_txns, holdout_txns)
                txn_len = workload.txns[t][0][3]
                test_seq = txn_seq.copy()
                test_seq.append(t)
                cost = 0
                cost = workload.get_opt_seq_cost(test_seq)
                if cost < min_cost:
                # if relative_cost < min_relative_cost:
                    min_cost = cost
                    min_txn = t
                    # min_relative_cost = relative_cost
                    # min_index = j
                if done:
                    break
            assert(min_txn != -1)
            running_cost = min_cost
            txn_seq.append(min_txn)
            holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)

            if workload.debug:
                print("min: ", min_txn, remaining_txns, holdout_txns, txn_seq)
        if workload.debug:
            print(txn_seq)
            print(len(set(txn_seq)))
        assert len(set(txn_seq)) == workload.num_txns
        # print(txn_seq)

        overall_cost = workload.get_opt_seq_cost(txn_seq)

        return overall_cost, txn_seq

    # Local search: windowed pairwise swaps plus insertion moves, all
    # evaluated with the true objective (get_opt_seq_cost). Insertion
    # reaches orderings swaps cannot, escaping greedy's mistakes.
    def refine(seq, cost):
        improved = True
        n = len(seq)
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, min(i + 10, n)):
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
            for i in range(n):
                for j in range(max(0, i - 10), min(i + 11, n)):
                    if j == i:
                        continue
                    t = seq.pop(i)
                    seq.insert(j, t)
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        t = seq.pop(j)
                        seq.insert(i, t)
        return cost, seq

    # Adaptive perturbation: random swaps plus a segment reversal whose
    # strength grows the longer the incumbent has not improved, so the
    # search diversifies more aggressively when stuck.
    def perturb(seq, strength):
        s = seq[:]
        n = len(s)
        for _ in range(strength):
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, min(i + 10, n - 1))
            s[i], s[j] = s[j], s[i]
        seg = min(6 + 2 * strength, n)
        i = random.randint(0, max(0, n - seg))
        s[i:i + seg] = reversed(s[i:i + seg])
        return s

    import time as _time
    import math
    deadline = _time.time() + 105  # per-workload budget; 3 workloads < 360s

    # Greedy restarts + refinement build a strong incumbent...
    best_cost = float('inf')
    best_seq = None
    for _ in range(6):
        cost, seq = get_greedy_cost_sampled(12, 1.0)
        cost, seq = refine(seq, cost)
        if cost < best_cost:
            best_cost, best_seq = cost, seq

    # ...then iterated local search with annealing acceptance: perturb,
    # re-refine, accept improvements always and slightly worse solutions
    # with a decaying probability (current solution drifts, incumbent is
    # kept separately), enabling escape from deep local optima.
    cur_cost, cur_seq = best_cost, best_seq[:]
    stuck = 0
    t0 = _time.time()
    while _time.time() < deadline:
        strength = 2 + min(6, stuck // 3)
        cand = perturb(cur_seq, strength)
        cost = workload.get_opt_seq_cost(cand)
        cost, cand = refine(cand, cost)
        if cost < best_cost:
            best_cost, best_seq = cost, cand
            cur_cost, cur_seq = cost, cand
            stuck = 0
        else:
            stuck += 1
            frac = (_time.time() - t0) / 105.0
            temp = max(0.5, 8.0 * (1.0 - frac))
            if cost < cur_cost or random.random() < math.exp(-(cost - cur_cost) / temp):
                cur_cost, cur_seq = cost, cand
            if cost == best_cost and random.random() < 0.2:
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
