import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction order using exact incremental makespan
    evaluations, diversified greedy construction, and permutation descent.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    if n == 0:
        return 0, []

    # The simulator is the authoritative conflict/makespan model.  Caching is
    # particularly valuable because local moves often recreate prior prefixes.
    cost_cache = {}

    def cost(sequence):
        key = tuple(sequence)
        if key not in cost_cache:
            cost_cache[key] = workload.get_opt_seq_cost(sequence)
        return cost_cache[key]

    # num_seqs is used as a true construction budget.  For small workloads,
    # trying every possible starting transaction gives useful diversification.
    seed_count = max(1, int(num_seqs))
    if n <= seed_count:
        starts = list(range(n))
    else:
        # Keep one fixed start for reproducibility while sampling additional
        # independent starts to avoid a transaction-id ordering bias.
        starts = [0]
        pool = list(range(1, n))
        random.shuffle(pool)
        starts.extend(pool[:seed_count - 1])

    def construct_from(start):
        """
        Greedily append the transaction with the smallest actual resulting
        makespan.  Ties are randomized, preserving diversified restarts when
        several conflict-equivalent alternatives exist.
        """
        sequence = [start]
        remaining = set(range(n))
        remaining.remove(start)

        while remaining:
            candidates = list(remaining)
            random.shuffle(candidates)

            best_txn = None
            best_cost = None
            for txn in candidates:
                candidate_cost = cost(sequence + [txn])
                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_txn = txn

            sequence.append(best_txn)
            remaining.remove(best_txn)

        return cost(sequence), sequence

    constructed = [construct_from(start) for start in starts]
    constructed.sort(key=lambda result: result[0])
    best_cost, best_schedule = constructed[0]

    def best_relocation(schedule, current_cost):
        """
        Search the complete one-transaction relocation neighborhood.  Unlike
        append-only greedy construction, this can move an already scheduled
        transaction before the transaction that caused its conflict delay.
        """
        best_value = current_cost
        best_order = schedule

        for source in range(n):
            txn = schedule[source]
            reduced = schedule[:source] + schedule[source + 1:]

            for destination in range(n):
                # Recreating the unmodified order is unnecessary.
                if destination == source:
                    continue

                candidate = reduced[:destination] + [txn] + reduced[destination:]
                candidate_cost = cost(candidate)
                if candidate_cost < best_value:
                    best_value = candidate_cost
                    best_order = candidate

        return best_value, best_order

    def best_swap(schedule, current_cost):
        """
        Swap moves complement relocations and can escape a relocation-local
        ordering where two mutually interacting transactions must move together.
        """
        best_value = current_cost
        best_order = schedule

        for left in range(n - 1):
            for right in range(left + 1, n):
                candidate = schedule[:]
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_cost = cost(candidate)
                if candidate_cost < best_value:
                    best_value = candidate_cost
                    best_order = candidate

        return best_value, best_order

    def best_block_relocation(schedule, current_cost):
        """
        Move an adjacent pair as one ordered block.  This reaches schedules
        that require two cooperating transactions to cross another transaction
        together, which neither a strict single relocation nor a swap need
        expose as an improving intermediate move.
        """
        best_value = current_cost
        best_order = schedule

        for source in range(n - 1):
            block = schedule[source:source + 2]
            reduced = schedule[:source] + schedule[source + 2:]

            for destination in range(n - 1):
                # This insertion position recreates the original order.
                if destination == source:
                    continue

                candidate = reduced[:destination] + block + reduced[destination:]
                candidate_cost = cost(candidate)
                if candidate_cost < best_value:
                    best_value = candidate_cost
                    best_order = candidate

        return best_value, best_order

    # Descend until all exact-cost neighborhoods are locally optimal.  A move
    # in one neighborhood can expose new improving relocations, so stopping
    # after an arbitrary number of passes leaves avoidable conflict delays.
    while True:
        improved_cost, improved_schedule = best_relocation(best_schedule, best_cost)
        if improved_cost < best_cost:
            best_cost, best_schedule = improved_cost, improved_schedule
            continue

        # When one-item inserts stagnate, search complementary neighborhoods
        # and take the strongest exact-cost improvement.  The next iteration
        # returns to relocation search for the newly changed ordering.
        swapped_cost, swapped_schedule = best_swap(best_schedule, best_cost)
        block_cost, block_schedule = best_block_relocation(
            best_schedule, best_cost
        )

        if swapped_cost <= block_cost and swapped_cost < best_cost:
            best_cost, best_schedule = swapped_cost, swapped_schedule
        elif block_cost < best_cost:
            best_cost, best_schedule = block_cost, block_schedule
        else:
            break

    # Return a freshly evaluated value so the result always agrees exactly
    # with the workload's makespan implementation.
    return cost(best_schedule), best_schedule

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