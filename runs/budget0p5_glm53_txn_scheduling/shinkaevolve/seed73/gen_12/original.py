import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using full-lookahead greedy with multistart restarts.

    At each step, every remaining transaction is evaluated by computing the
    makespan of the partial schedule extended with it, and the best one is
    chosen. Multiple random restarts (one per distinct starting transaction)
    diversify the search and the best complete schedule is returned.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import time

    def greedy_from(start_txn):
        txn_seq = [start_txn]
        remaining = [x for x in range(workload.num_txns) if x != start_txn]
        while remaining:
            best_cost = None
            best_txn = None
            for t in remaining:
                test_seq = txn_seq + [t]
                cost = workload.get_opt_seq_cost(test_seq)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_txn = t
            txn_seq.append(best_txn)
            remaining.remove(best_txn)
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    start_time = time.time()
    best_cost = None
    best_seq = None
    order = list(range(workload.num_txns))
    random.shuffle(order)
    for start_txn in order:
        if time.time() - start_time > 25 and best_seq is not None:
            break
        cost, seq = greedy_from(start_txn)
        if best_cost is None or cost < best_cost:
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