import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using multi-restart greedy cost sampling
    followed by swap + reinsertion local search on top candidates.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    def get_greedy_cost_sampled(num_samples, sample_rate):
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)]
        remaining_txns.remove(start_txn)
        for i in range(0, workload.num_txns - 1):
            min_cost = float('inf')
            min_txn = -1
            holdout_txns = []
            done = False

            sample = random.random()
            if sample > sample_rate and remaining_txns:
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns.pop(idx)
                txn_seq.append(t)
                continue

            for j in range(0, num_samples):
                if len(remaining_txns) == 0:
                    break
                if len(remaining_txns) == 1:
                    done = True
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns.pop(idx)
                holdout_txns.append(t)
                test_seq = txn_seq.copy()
                test_seq.append(t)
                cost = workload.get_opt_seq_cost(test_seq)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
                if done:
                    break
            if min_txn == -1 and holdout_txns:
                min_txn = holdout_txns[0]
            txn_seq.append(min_txn)
            if min_txn in holdout_txns:
                holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)

        assert len(set(txn_seq)) == workload.num_txns
        overall_cost = workload.get_opt_seq_cost(txn_seq)
        return overall_cost, txn_seq

    # local improvement: pairwise swaps + reinsertion until no improvement
    def swap_improve(seq):
        cur = seq.copy()
        cur_cost = workload.get_opt_seq_cost(cur)
        improved = True
        while improved:
            improved = False
            for i in range(len(cur)):
                for j in range(i + 1, len(cur)):
                    cand = cur.copy()
                    cand[i], cand[j] = cand[j], cand[i]
                    c = workload.get_opt_seq_cost(cand)
                    if c < cur_cost:
                        cur, cur_cost = cand, c
                        improved = True
            # reinsertion moves
            for i in range(len(cur)):
                t = cur.pop(i)
                for j in range(len(cur) + 1):
                    cand = cur.copy()
                    cand.insert(j, t)
                    c = workload.get_opt_seq_cost(cand)
                    if c < cur_cost:
                        cur, cur_cost = cand, c
                        improved = True
                        break
                else:
                    cur.insert(i, t)
                    continue
                break
        return cur_cost, cur

    # Phase 1: multi-restart greedy, keep top candidates
    candidates = []
    for _ in range(20):
        c, s = get_greedy_cost_sampled(6, 0.95)
        candidates.append((c, s))
    candidates.sort(key=lambda x: x[0])

    # Phase 2: local search on top-3 greedy candidates
    best_cost = float('inf')
    best_seq = None
    for c, s in candidates[:3]:
        c2, s2 = swap_improve(s)
        if c2 < best_cost:
            best_cost, best_seq = c2, s2

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