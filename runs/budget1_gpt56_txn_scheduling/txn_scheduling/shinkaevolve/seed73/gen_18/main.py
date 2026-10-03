import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Construct conflict-aware greedy schedules and optimize elite candidates
    through exact-cost local and large-neighborhood search.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []

    cost_cache = {}

    def schedule_cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    def greedy_order(start_txn):
        order = [start_txn]
        remaining = set(range(num_txns))
        remaining.remove(start_txn)

        while remaining:
            best_cost = None
            best_choices = []

            # Exact prefix makespan captures read/write dependency effects
            # directly, unlike transaction-size based ordering heuristics.
            for txn in remaining:
                candidate_cost = schedule_cost(order + [txn])
                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_choices = [txn]
                elif candidate_cost == best_cost:
                    best_choices.append(txn)

            chosen = random.choice(best_choices)
            order.append(chosen)
            remaining.remove(chosen)

        return schedule_cost(order), order

    def improve(order, current_cost, rounds=4):
        """Best-improvement relocation, exchange, and reversal search."""
        n = len(order)

        for _ in range(rounds):
            improved_cost = current_cost
            improved_order = order

            # Relocation changes the precedence of one transaction against
            # an entire region and is especially useful around hot keys.
            for source in range(n):
                txn = order[source]
                shortened = order[:source] + order[source + 1:]
                for destination in range(n):
                    candidate = (
                        shortened[:destination] + [txn] + shortened[destination:]
                    )
                    if candidate == order:
                        continue
                    candidate_cost = schedule_cost(candidate)
                    if candidate_cost < improved_cost:
                        improved_cost = candidate_cost
                        improved_order = candidate

            # Direct exchanges can cross barriers that relocation cannot.
            for left in range(n - 1):
                for right in range(left + 1, n):
                    candidate = order[:]
                    candidate[left], candidate[right] = (
                        candidate[right], candidate[left]
                    )
                    candidate_cost = schedule_cost(candidate)
                    if candidate_cost < improved_cost:
                        improved_cost = candidate_cost
                        improved_order = candidate

            # Reversing a dependency-dense region repairs several precedence
            # decisions in one exact-cost move.
            for first in range(n - 2):
                for last in range(first + 2, n):
                    candidate = (
                        order[:first]
                        + order[first:last + 1][::-1]
                        + order[last + 1:]
                    )
                    candidate_cost = schedule_cost(candidate)
                    if candidate_cost < improved_cost:
                        improved_cost = candidate_cost
                        improved_order = candidate

            if improved_cost >= current_cost:
                break
            order, current_cost = improved_order, improved_cost

        return current_cost, order

    starts = list(range(num_txns))
    random.shuffle(starts)
    attempts = max(1, num_seqs)

    seeds = []
    best_cost = float("inf")
    best_order = None

    for attempt in range(attempts):
        start = starts[attempt % num_txns]
        cost, order = greedy_order(start)
        seeds.append((cost, order))
        if cost < best_cost:
            best_cost, best_order = cost, order

    # Refine several distinct construction paths instead of assuming that the
    # initially cheapest greedy schedule has the best local-search basin.
    elite_count = min(3, len(seeds))
    for seed_cost, seed_order in sorted(seeds, key=lambda item: item[0])[:elite_count]:
        refined_cost, refined_order = improve(seed_order[:], seed_cost)
        if refined_cost < best_cost:
            best_cost, best_order = refined_cost, refined_order

    # Destroy-and-repair permits coordinated movement of multiple transactions
    # that share conflicts and therefore escapes one-move local minima.
    if num_txns >= 3:
        repair_attempts = max(6, min(16, num_seqs * 2))
        max_destroy = min(6, num_txns)

        for attempt in range(repair_attempts):
            destroy_size = random.randint(3, max_destroy)

            if attempt % 3 == 0:
                first = random.randint(0, num_txns - destroy_size)
                removed = best_order[first:first + destroy_size]
                repaired = (
                    best_order[:first] + best_order[first + destroy_size:]
                )
            else:
                removed_positions = set(
                    random.sample(range(num_txns), destroy_size)
                )
                removed = [
                    txn for index, txn in enumerate(best_order)
                    if index in removed_positions
                ]
                repaired = [
                    txn for index, txn in enumerate(best_order)
                    if index not in removed_positions
                ]

            while removed:
                insertion_cost = None
                insertion_choices = []

                for txn in removed:
                    for destination in range(len(repaired) + 1):
                        candidate = (
                            repaired[:destination]
                            + [txn]
                            + repaired[destination:]
                        )
                        candidate_cost = schedule_cost(candidate)

                        if insertion_cost is None or candidate_cost < insertion_cost:
                            insertion_cost = candidate_cost
                            insertion_choices = [(txn, candidate)]
                        elif candidate_cost == insertion_cost:
                            insertion_choices.append((txn, candidate))

                selected_txn, repaired = random.choice(insertion_choices)
                removed.remove(selected_txn)

            repaired_cost = schedule_cost(repaired)
            if repaired_cost < best_cost:
                best_cost, best_order = repaired_cost, repaired
                # Repair can expose new one-move improvements, so polish the
                # newly reached basin before continuing large-neighborhood work.
                best_cost, best_order = improve(best_order, best_cost, rounds=3)

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