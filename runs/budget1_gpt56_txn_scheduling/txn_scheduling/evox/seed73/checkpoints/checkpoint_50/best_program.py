import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Keep a conflict-cost beam of prefixes, then insertion-descend diverse finalists."""
    n = workload.num_txns
    if not n:
        return 0, []

    # Unlike independent greedy starts, a beam may retain a prefix whose next
    # move is temporarily more expensive but which avoids a later critical
    # read/write-conflict chain.  Width n preserves the broad initial search
    # of the previous method while pruning weak alternatives at every depth.
    width = max(1, min(n, max(num_seqs, 2)))
    states = [(0, [])]

    for _ in range(n):
        expanded = []
        for _, seq in states:
            used = set(seq)
            for txn in range(n):
                if txn not in used:
                    trial = seq + [txn]
                    expanded.append((workload.get_opt_seq_cost(trial), trial))
        expanded.sort(key=lambda state: state[0])
        states = expanded[:width]

    # The best complete prefix is not always in the best insertion basin.
    # Descending a few beam finalists provides deterministic basin diversity
    # without the much larger cost of fully optimizing every candidate.
    best_cost, best_seq = float("inf"), None
    for initial_cost, initial_seq in states[:min(3, len(states))]:
        cost, seq = initial_cost, initial_seq[:]

        while True:
            move_cost, move_seq = cost, None
            for source in range(n):
                reduced = seq[:]
                txn = reduced.pop(source)
                for target in range(n):
                    trial = reduced[:]
                    trial.insert(target, txn)
                    trial_cost = workload.get_opt_seq_cost(trial)
                    if trial_cost < move_cost:
                        move_cost, move_seq = trial_cost, trial

            if move_seq is None:
                break
            cost, seq = move_cost, move_seq

        if cost < best_cost:
            best_cost, best_seq = cost, seq

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
