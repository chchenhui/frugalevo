import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using full-lookahead greedy with randomized
    multistart restarts and tie-breaking (crossover of full-lookahead
    greedy and randomized sampling strategies).

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import time

    def greedy_from(start_txn, sample_cap):
        txn_seq = [start_txn]
        remaining = [x for x in range(workload.num_txns) if x != start_txn]
        while remaining:
            candidates = remaining
            if sample_cap and len(remaining) > sample_cap:
                candidates = random.sample(remaining, sample_cap)
            best_cost = None
            best_txns = []
            for t in candidates:
                test_seq = txn_seq + [t]
                cost = workload.get_opt_seq_cost(test_seq)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_txns = [t]
                elif cost == best_cost:
                    best_txns.append(t)
            best_txn = random.choice(best_txns)
            txn_seq.append(best_txn)
            remaining.remove(best_txn)
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    start_time = time.time()
    best_cost = None
    best_seq = None

    # First pass: full-lookahead greedy from every distinct start txn.
    order = list(range(workload.num_txns))
    random.shuffle(order)
    for start_txn in order:
        if time.time() - start_time > 20 and best_seq is not None:
            break
        cost, seq = greedy_from(start_txn, sample_cap=None)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_seq = seq

    # Second pass: randomized-sampling restarts within remaining budget.
    while time.time() - start_time < 27:
        start_txn = random.randint(0, workload.num_txns - 1)
        cost, seq = greedy_from(start_txn, sample_cap=15)
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