import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Build several conflict-aware greedy orders, then improve the best complete
    order with exact-cost local search.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    # Local search frequently revisits schedules produced by different moves.
    # Caching makes those exact objective evaluations inexpensive.
    cost_cache = {}

    def schedule_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    def greedy_order(start_txn):
        order = [start_txn]
        remaining = list(range(num_txns))
        remaining.remove(start_txn)

        while remaining:
            # For small instances inspect every legal next transaction.  On
            # larger instances retain diversity while still testing far more
            # alternatives than the former fixed sample of ten.
            if len(remaining) <= 32:
                candidates = remaining[:]
            else:
                width = min(len(remaining), max(12, min(28, 2 * num_seqs)))
                candidates = random.sample(remaining, width)

            best_prefix_cost = None
            best_candidates = []
            for txn in candidates:
                candidate_cost = schedule_cost(order + [txn])
                if best_prefix_cost is None or candidate_cost < best_prefix_cost:
                    best_prefix_cost = candidate_cost
                    best_candidates = [txn]
                elif candidate_cost == best_prefix_cost:
                    best_candidates.append(txn)

            # Equal prefix costs can have very different future conflicts, so
            # make ties deliberately diverse across independent restarts.
            chosen = random.choice(best_candidates)
            order.append(chosen)
            remaining.remove(chosen)

        return schedule_cost(order), order

    # Use distinct starting transactions where possible.  A transaction that
    # is poor as the first choice can otherwise never be recovered by an
    # append-only greedy construction.
    starts = list(range(num_txns))
    random.shuffle(starts)
    attempts = max(1, num_seqs)

    best_cost = float("inf")
    best_order = None
    for attempt in range(attempts):
        cost, order = greedy_order(starts[attempt % num_txns])
        if cost < best_cost:
            best_cost, best_order = cost, order

    def refine(order, current_cost):
        # Best-improving adjacent exchanges cheaply remove many ordering
        # inversions caused by greedy prefix decisions.
        for _ in range(min(8, num_txns)):
            move_cost = current_cost
            move_order = None
            for index in range(num_txns - 1):
                candidate = order[:]
                candidate[index], candidate[index + 1] = (
                    candidate[index + 1], candidate[index]
                )
                candidate_cost = schedule_cost(candidate)
                if candidate_cost < move_cost:
                    move_cost, move_order = candidate_cost, candidate
            if move_order is None:
                break
            order, current_cost = move_order, move_cost

        # Insertions can escape an adjacent-swap local minimum.  Exhaustively
        # test them for small workloads and use a bounded unbiased sample for
        # large ones to keep optimization time predictable.
        for _ in range(4):
            if num_txns <= 30:
                moves = [
                    (source, destination)
                    for source in range(num_txns)
                    for destination in range(num_txns)
                    if source != destination
                ]
            else:
                moves = [
                    (random.randrange(num_txns), random.randrange(num_txns))
                    for _ in range(6 * num_txns)
                ]

            move_cost = current_cost
            move_order = None
            for source, destination in moves:
                if source == destination:
                    continue
                candidate = order[:]
                txn = candidate.pop(source)
                candidate.insert(destination, txn)
                candidate_cost = schedule_cost(candidate)
                if candidate_cost < move_cost:
                    move_cost, move_order = candidate_cost, candidate
            if move_order is None:
                break
            order, current_cost = move_order, move_cost

        return current_cost, order

    best_cost, best_order = refine(best_order, best_cost)
    return best_cost, best_order

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