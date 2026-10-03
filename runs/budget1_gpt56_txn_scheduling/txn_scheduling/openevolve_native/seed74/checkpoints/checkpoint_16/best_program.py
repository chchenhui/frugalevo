import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Use diversified true-cost insertion, then relocation and swap descent."""
    n = workload.num_txns
    if n == 0:
        return 0, []

    # Prefix evaluations are reused by insertion and local search.  Bounding
    # this cache avoids retaining very many long permutation tuples on large
    # workloads while preserving the useful repeated evaluations.
    cache = {}

    def cost(order):
        key = tuple(order)
        value = cache.get(key)
        if value is None:
            value = workload.get_opt_seq_cost(list(key))
            if len(cache) < 120000:
                cache[key] = value
        return value

    def insert_construct(arrival, variant):
        """Insert each arriving transaction at its minimum actual-cost slot."""
        order = ()
        for txn in arrival:
            candidates = []
            best = None
            for position in range(len(order) + 1):
                trial = order[:position] + (txn,) + order[position:]
                value = cost(trial)
                if best is None or value < best:
                    best = value
                    candidates = [trial]
                elif value == best:
                    candidates.append(trial)

            # Different tie choices retain useful diversity when incomplete
            # schedules have identical makespans.
            order = candidates[variant % len(candidates)]
        return order

    # Rotations and reversals change the order in which insertion decisions are
    # committed.  Unlike the old beam's truncated candidate set, every
    # transaction can become the first transaction in a construction.
    runs = min(12, max(2, num_seqs))
    seeds = []
    base = tuple(range(n))
    for run in range(runs):
        shift = (run * n) // runs
        arrival = base[shift:] + base[:shift]
        if run % 2:
            arrival = tuple(reversed(arrival))
        order = insert_construct(arrival, run)
        seeds.append((cost(order), order))

    # Refine several independently constructed schedules.  A non-adjacent
    # swap cannot in general be represented by one improving relocation, so
    # both neighborhoods are searched using the simulator's real makespan.
    seeds.sort(key=lambda item: item[0])
    finalists = seeds[:min(3, len(seeds))]
    best_cost, best_order = finalists[0]

    for initial_cost, initial_order in finalists:
        current_cost, current_order = initial_cost, initial_order
        for _ in range(7):
            next_cost, next_order = current_cost, current_order

            for source in range(n):
                txn = current_order[source]
                reduced = current_order[:source] + current_order[source + 1:]
                for target in range(n):
                    if target == source:
                        continue
                    trial = reduced[:target] + (txn,) + reduced[target:]
                    value = cost(trial)
                    if value < next_cost:
                        next_cost, next_order = value, trial

            for left in range(n - 1):
                for right in range(left + 1, n):
                    trial = (current_order[:left] + (current_order[right],) +
                             current_order[left + 1:right] +
                             (current_order[left],) +
                             current_order[right + 1:])
                    value = cost(trial)
                    if value < next_cost:
                        next_cost, next_order = value, trial

            if next_cost >= current_cost:
                break
            current_cost, current_order = next_cost, next_order

        if current_cost < best_cost:
            best_cost, best_order = current_cost, current_order

    return best_cost, list(best_order)

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
