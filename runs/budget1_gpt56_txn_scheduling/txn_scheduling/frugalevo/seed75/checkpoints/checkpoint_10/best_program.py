import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """Destroy critical transactions, exactly reinsert them, then use swap descent."""
    import time

    now = time.perf_counter()
    global _schedule_search_deadline
    if "_schedule_search_deadline" not in globals():
        _schedule_search_deadline = now + 330.0
    deadline = min(_schedule_search_deadline, now + 105.0)

    n = workload.num_txns
    if n == 0:
        return 0, []

    cache = {}

    def cost(seq):
        """Return the simulator cost, caching identical permutations."""
        key = tuple(seq)
        if key not in cache:
            cache[key] = workload.get_opt_seq_cost(list(key))
        return cache[key]

    best_seq = list(range(n))
    best_cost = cost(best_seq)
    ruin_count = min(8, max(3, n // 8), n)

    for attempt in range(4):
        if time.perf_counter() >= deadline:
            break

        impacts = []
        for txn in best_seq:
            if time.perf_counter() >= deadline:
                break
            remainder = [x for x in best_seq if x != txn]
            impacts.append((best_cost - cost(remainder), txn))
        if len(impacts) != n:
            break

        # Remove transactions whose deletion most improves the true cost.
        impacts.sort(key=lambda x: (-x[0], (x[1] - attempt) % max(1, n)))
        removed = {txn for _, txn in impacts[:ruin_count]}
        partial = [txn for txn in best_seq if txn not in removed]

        # Exact destroy-repair: choose both the next transaction and position.
        while removed and time.perf_counter() < deadline:
            choice = None
            for txn in sorted(removed, key=lambda x: (x - attempt) % max(1, n)):
                for pos in range(len(partial) + 1):
                    trial = partial[:pos] + [txn] + partial[pos:]
                    value = cost(trial)
                    candidate = (value, txn, pos, trial)
                    if choice is None or candidate[:3] < choice[:3]:
                        choice = candidate
            if choice is None:
                break
            _, txn, _, partial = choice
            removed.remove(txn)

        # This only matters if the shared deadline interrupted repair.
        partial.extend(sorted(removed))
        candidate = partial
        candidate_cost = cost(candidate)

        # Bounded adjacent-swap descent removes remaining local precedence errors.
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for pos in range(n - 1):
                if time.perf_counter() >= deadline:
                    break
                trial = candidate[:]
                trial[pos], trial[pos + 1] = trial[pos + 1], trial[pos]
                value = cost(trial)
                if value < candidate_cost:
                    candidate, candidate_cost = trial, value
                    improved = True
                    break

        if candidate_cost < best_cost:
            best_seq, best_cost = candidate[:], candidate_cost

    if len(best_seq) != n or set(best_seq) != set(range(n)):
        best_seq = list(range(n))
        best_cost = cost(best_seq)
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
