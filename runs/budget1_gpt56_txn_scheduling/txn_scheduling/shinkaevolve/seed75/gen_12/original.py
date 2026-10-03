import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan serial transaction order.

    Partial schedules are ranked using the simulator's actual conflict-aware
    makespan, then the best complete order is improved by relocation search.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    # On tiny instances, evaluating every order is cheap and removes all
    # heuristic uncertainty.
    if num_txns <= 8:
        best_cost = float("inf")
        best_schedule = None

        def enumerate_orders(prefix, remaining):
            nonlocal best_cost, best_schedule
            if not remaining:
                cost = workload.get_opt_seq_cost(prefix)
                if cost < best_cost:
                    best_cost = cost
                    best_schedule = prefix
                return
            for txn in remaining:
                enumerate_orders(
                    prefix + [txn],
                    [other for other in remaining if other != txn],
                )

        enumerate_orders([], list(range(num_txns)))
        return best_cost, best_schedule

    beam_width = max(1, num_seqs)
    all_txns = tuple(range(num_txns))
    # Each entry is (prefix makespan, prefix tuple, remaining tuple).
    beam = [(0, (), all_txns)]

    for _ in range(num_txns):
        candidates = []
        for _, prefix, remaining in beam:
            for txn in remaining:
                next_prefix = prefix + (txn,)
                next_remaining = tuple(item for item in remaining if item != txn)
                candidates.append((
                    workload.get_opt_seq_cost(list(next_prefix)),
                    next_prefix,
                    next_remaining,
                ))

        # A prefix is unique by construction, but sorting this way also gives
        # stable, reproducible choices when multiple schedules tie.
        candidates.sort(key=lambda entry: (entry[0], entry[1]))
        beam = candidates[:beam_width]

    best_cost, best_prefix, _ = min(beam, key=lambda entry: (entry[0], entry[1]))
    best_schedule = list(best_prefix)

    # Greedy prefix choices can be locally correct but globally poor.  Search
    # all transaction relocation moves and accept the first improving move;
    # repeat until no conflict-aware makespan improvement remains.
    improved = True
    while improved:
        improved = False
        for source in range(num_txns):
            txn = best_schedule[source]
            without_txn = best_schedule[:source] + best_schedule[source + 1:]
            for destination in range(num_txns):
                if destination == source:
                    continue
                candidate = (
                    without_txn[:destination] + [txn] + without_txn[destination:]
                )
                candidate_cost = workload.get_opt_seq_cost(candidate)
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_schedule = candidate
                    improved = True
                    break
            if improved:
                break

    if workload.debug:
        print("best schedule:", best_schedule, "makespan:", best_cost)
    return best_cost, best_schedule

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