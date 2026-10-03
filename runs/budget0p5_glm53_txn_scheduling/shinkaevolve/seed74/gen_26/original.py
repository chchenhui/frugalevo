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

    def swap_pass(best_cost, best_sched):
        improved = False
        i = 1
        while i < len(best_sched) - 1:
            cand = best_sched[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = workload.get_opt_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_sched = cand
                improved = True
            else:
                i += 1
        return best_cost, best_sched, improved

    def or_opt_pass(best_cost, best_sched):
        # Or-opt: remove a segment of length 1-3 and reinsert it elsewhere.
        improved = False
        n = len(best_sched)
        for seg_len in (1, 2, 3):
            for i in range(n - seg_len + 1):
                seg = best_sched[i:i + seg_len]
                rest = best_sched[:i] + best_sched[i + seg_len:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    cand = rest[:j] + seg + rest[j:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < best_cost:
                        best_cost = c
                        best_sched = cand
                        improved = True
                        # restart scanning for this seg_len with new schedule
                        return best_cost, best_sched, improved
        return best_cost, best_sched, improved

    def local_search(schedule, max_rounds=100):
        best_cost = workload.get_opt_seq_cost(schedule)
        best_sched = list(schedule)
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            best_cost, best_sched, imp1 = swap_pass(best_cost, best_sched)
            best_cost, best_sched, imp2 = or_opt_pass(best_cost, best_sched)
            if not imp1 and not imp2:
                break
        return best_cost, best_sched

    best_cost = None
    best_sched = None
    for _ in range(30):
        cost, sched = get_greedy_cost_sampled(8, 0.9)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_sched = sched

    best_cost, best_sched = local_search(best_sched)
    return best_cost, best_sched

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