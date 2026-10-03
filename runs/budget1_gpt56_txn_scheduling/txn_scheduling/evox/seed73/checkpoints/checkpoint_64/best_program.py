import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Search exact-cost beams from both schedule ends, then insertion-descend."""
    n = workload.num_txns
    if not n:
        return 0, []

    width = max(2, min(n, num_seqs))

    def beam(prepend):
        """Keep low-makespan partial orders while growing at one chosen end."""
        states = [(0, [])]
        for _ in range(n):
            expanded = []
            for _, seq in states:
                used = set(seq)
                for txn in range(n):
                    if txn not in used:
                        trial = [txn] + seq if prepend else seq + [txn]
                        expanded.append((workload.get_opt_seq_cost(trial), trial))
            expanded.sort(key=lambda state: state[0])
            states = expanded[:width]
        return states

    def improve(cost, seq):
        """Apply best strict single-transaction insertion moves to convergence."""
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
                return cost, seq
            cost, seq = move_cost, move_seq

    # Appending is naturally biased toward decisions near the start of a
    # schedule.  Prepending supplies equally exact candidates whose important
    # conflict decisions were made from the end, reaching different local
    # minima without randomization.
    forward = beam(False)
    backward = beam(True)
    finalists = forward[:2] + backward[:2]

    best_cost, best_seq = float("inf"), None
    for cost, seq in finalists:
        cost, seq = improve(cost, seq)
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
