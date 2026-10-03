import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use exact-cost beam construction followed by full-schedule insertion improvement."""
    n = workload.num_txns
    if n <= 1:
        seq = list(range(n))
        return workload.get_opt_seq_cost(seq), seq

    # Unlike sampled greedy selection, retain several promising partial orders.
    # Every expansion is evaluated with the simulator's actual conflict model.
    width = max(1, min(8, num_seqs))
    beam = [(0, [], list(range(n)))]

    for _ in range(n):
        expanded = []
        for _, prefix, remaining in beam:
            for txn in remaining:
                candidate = prefix + [txn]
                cost = workload.get_opt_seq_cost(candidate)
                expanded.append(
                    (cost, candidate, [other for other in remaining if other != txn])
                )
        expanded.sort(key=lambda state: state[0])
        beam = expanded[:width]

    best_cost, best_seq, _ = min(beam, key=lambda state: state[0])

    # Prefix cost cannot completely predict conflicts introduced much later.
    # Relocating one transaction evaluates every resulting complete schedule.
    for _ in range(3):
        next_cost = best_cost
        next_seq = best_seq
        for source in range(n):
            txn = best_seq[source]
            without_txn = best_seq[:source] + best_seq[source + 1:]
            for destination in range(n):
                candidate = without_txn[:destination] + [txn] + without_txn[destination:]
                cost = workload.get_opt_seq_cost(candidate)
                if cost < next_cost:
                    next_cost, next_seq = cost, candidate
        if next_cost >= best_cost:
            break
        best_cost, best_seq = next_cost, next_seq

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
