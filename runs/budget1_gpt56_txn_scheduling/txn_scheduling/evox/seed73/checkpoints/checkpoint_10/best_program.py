import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use exhaustive conflict-cost greedy starts followed by insertion descent."""
    n = workload.num_txns
    if not n:
        return 0, []

    best_cost, best_seq = float("inf"), None

    # Sampling candidates can miss the one transaction that avoids extending
    # the current critical conflict path.  Evaluate every legal next choice.
    # Starting from every transaction also gives deterministic and substantially
    # more diverse seed schedules than random candidate samples.
    for start in range(n):
        seq = [start]
        remaining = list(range(n))
        remaining.remove(start)

        while remaining:
            chosen = min(
                remaining,
                key=lambda txn: workload.get_opt_seq_cost(seq + [txn])
            )
            seq.append(chosen)
            remaining.remove(chosen)

        cost = workload.get_opt_seq_cost(seq)
        if cost < best_cost:
            best_cost, best_seq = cost, seq

    # A relocation can move a transaction across an entire conflicting group.
    # Continue best-improving relocation until a true insertion local optimum
    # is reached; every accepted candidate is scored by the actual makespan.
    while True:
        move_cost, move_seq = best_cost, None
        for source in range(n):
            reduced = best_seq[:]
            txn = reduced.pop(source)
            for target in range(n):
                trial = reduced[:]
                trial.insert(target, txn)
                cost = workload.get_opt_seq_cost(trial)
                if cost < move_cost:
                    move_cost, move_seq = cost, trial

        if move_seq is None:
            break
        best_cost, best_seq = move_cost, move_seq

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
