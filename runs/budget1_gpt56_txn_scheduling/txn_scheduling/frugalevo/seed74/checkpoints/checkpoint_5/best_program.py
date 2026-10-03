import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Construct sampled best-insertion schedules and apply bounded local search."""
    import time

    n = workload.num_txns
    if n == 0:
        return 0, []

    deadline = time.monotonic() + 105.0
    cache = {}

    def cost(seq):
        key = tuple(seq)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(list(seq))
        return cache[key]

    def best_insertion_order(start):
        """Insert sampled transactions at the cheapest exact-cost position."""
        order = [start]
        remaining = set(range(n))
        remaining.remove(start)

        while remaining and time.monotonic() < deadline:
            candidates = list(remaining)
            random.shuffle(candidates)
            candidates = candidates[:7]

            best_value = None
            best_order = None
            best_txn = None
            for txn in candidates:
                for position in range(len(order) + 1):
                    trial = order[:]
                    trial.insert(position, txn)
                    value = cost(trial)
                    if best_value is None or value < best_value:
                        best_value = value
                        best_order = trial
                        best_txn = txn

            if best_order is None:
                break
            order = best_order
            remaining.remove(best_txn)

        if remaining:
            order.extend(sorted(remaining))
        return order

    def improve(seq):
        """Accept at most 35 improving swaps or relocations using true cost."""
        current = cost(seq)
        accepted = 0
        while accepted < 35 and time.monotonic() < deadline:
            moves = []
            for i in range(n - 1):
                moves.append(("swap", i, i + 1))
            for i in range(n):
                for j in range(n):
                    if i != j and j != i + 1:
                        moves.append(("move", i, j))
            random.shuffle(moves)

            changed = False
            for kind, i, j in moves:
                if time.monotonic() >= deadline:
                    break
                trial = seq[:]
                if kind == "swap":
                    trial[i], trial[j] = trial[j], trial[i]
                else:
                    item = trial.pop(i)
                    trial.insert(j - (j > i), item)
                value = cost(trial)
                if value < current:
                    seq, current = trial, value
                    accepted += 1
                    changed = True
                    break
            if not changed:
                break
        return seq, current

    best_seq = list(range(n))
    best_cost = cost(best_seq)

    starts = max(1, min(4, int(num_seqs)))
    chosen = random.sample(range(n), min(starts, n))
    for start in chosen:
        if time.monotonic() >= deadline:
            break
        candidate = best_insertion_order(start)
        candidate, value = improve(candidate)
        if value < best_cost:
            best_seq, best_cost = candidate, value

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
