import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using time-budgeted multi-restart randomized greedy
    plus first-improvement swap local search.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    try:
        import time as _time
    except ImportError:
        _time = None

    n = workload.num_txns

    def greedy_run(num_samples, sample_rate):
        start_txn = random.randint(0, n - 1)
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, n)]
        remaining_txns.remove(start_txn)
        for i in range(0, n - 1):
            min_cost = float('inf')
            min_txn = -1
            holdout_txns = []
            done = False

            if random.random() > sample_rate and remaining_txns:
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
                test_seq = txn_seq + [t]
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
        assert len(set(txn_seq)) == n
        overall_cost = workload.get_opt_seq_cost(txn_seq)
        return overall_cost, txn_seq

    def swap_improve_first(seq, deadline):
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
                    if _time is not None and _time.time() > deadline:
                        return cur_cost, cur
                if _time is not None and _time.time() > deadline:
                    return cur_cost, cur
            # reinsertion moves (first improvement)
            restart_insert = True
            while restart_insert:
                restart_insert = False
                for i in range(len(cur)):
                    t = cur.pop(i)
                    placed = False
                    for j in range(len(cur) + 1):
                        cand = cur.copy()
                        cand.insert(j, t)
                        c = workload.get_opt_seq_cost(cand)
                        if c < cur_cost:
                            cur, cur_cost = cand, c
                            improved = True
                            placed = True
                            restart_insert = True
                            break
                        if _time is not None and _time.time() > deadline:
                            cur.insert(min(i, len(cur)), t)
                            return cur_cost, cur
                    if not placed:
                        cur.insert(i, t)
                    if restart_insert:
                        break
                if _time is not None and _time.time() > deadline:
                    return cur_cost, cur
        return cur_cost, cur

    best_cost = float('inf')
    best_seq = None

    if _time is not None:
        start_time = _time.time()
        budget = 2.5
        deadline = start_time + budget
    else:
        deadline = float('inf')

    # quick initial solution in case of timeout
    cost, seq = greedy_run(6, 1.0)
    if cost < best_cost:
        best_cost, best_seq = cost, seq

    restart = 0
    while True:
        if _time is not None and _time.time() > deadline:
            break
        if restart > 500:
            break
        restart += 1
        # alternate greedy parameters for diversity
        if restart % 3 == 0:
            cost, seq = greedy_run(8, 0.9)
        elif restart % 3 == 1:
            cost, seq = greedy_run(5, 1.0)
        else:
            cost, seq = greedy_run(6, 0.95)
        if cost < best_cost * 0.999:
            cost, seq = swap_improve_first(seq, deadline)
        if cost < best_cost:
            best_cost, best_seq = cost, seq

    # final polish on best
    if _time is not None:
        polish_deadline = deadline + 0.5
    else:
        polish_deadline = float('inf')
    c, s = swap_improve_first(best_seq, polish_deadline)
    if c < best_cost:
        best_cost, best_seq = c, s

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