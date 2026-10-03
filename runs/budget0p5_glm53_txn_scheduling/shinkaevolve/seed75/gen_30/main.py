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
    def greedy_from(start_txn):
        # Greedy construction: at each step, pick the transaction whose
        # addition to the current prefix yields the lowest makespan.
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(0, workload.num_txns)]
        remaining_txns.remove(start_txn)

        while remaining_txns:
            min_cost = float("inf")
            min_txn = -1
            for t in remaining_txns:
                test_seq = txn_seq + [t]
                cost = workload.get_opt_seq_cost(test_seq)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
            txn_seq.append(min_txn)
            remaining_txns.remove(min_txn)

        assert len(set(txn_seq)) == workload.num_txns
        overall_cost = workload.get_opt_seq_cost(txn_seq)
        return overall_cost, txn_seq

    _cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = _cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            _cost_cache[key] = c
        return c

    def local_search(seq, cost, max_iters=30):
        # Hybrid local search: alternate between full pairwise swap sweeps
        # and full insertion (relocate) sweeps. The two move types explore
        # different neighborhoods, so alternating them helps escape local
        # optima of either class. Best-improvement acceptance with the
        # shared cost cache keeps repeated evaluations cheap.
        n = len(seq)
        for _ in range(max_iters):
            improved = False
            # --- swap sweep ---
            best_pair = None
            best_pair_cost = cost
            for i in range(n - 1):
                for j in range(i + 1, n):
                    trial = seq[:]
                    trial[i], trial[j] = trial[j], trial[i]
                    c = eval_cost(trial)
                    if c < best_pair_cost:
                        best_pair_cost = c
                        best_pair = (i, j)
            if best_pair is not None:
                i, j = best_pair
                seq[i], seq[j] = seq[j], seq[i]
                cost = best_pair_cost
                improved = True
            # --- insertion sweep ---
            best_move = None
            best_move_cost = cost
            for i in range(n):
                t = seq.pop(i)
                for j in range(n):
                    seq.insert(j, t)
                    c = eval_cost(seq)
                    if c < best_move_cost:
                        best_move_cost = c
                        best_move = (i, j)
                    del seq[j]
                seq.insert(i, t)
            if best_move is not None:
                i, j = best_move
                t = seq.pop(i)
                seq.insert(j, t)
                cost = best_move_cost
                improved = True
            if not improved:
                break
        return cost, seq

    best_cost = float("inf")
    best_seq = None
    num_restarts = max(1, num_seqs)
    for i in range(num_restarts):
        if i < workload.num_txns:
            start_txn = i
        else:
            start_txn = random.randint(0, workload.num_txns - 1)
        cost, seq = greedy_from(start_txn)
        cost, seq = local_search(seq, cost)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

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