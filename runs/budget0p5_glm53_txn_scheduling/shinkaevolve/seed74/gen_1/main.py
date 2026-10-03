import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan transaction ordering using iterated local search
    with pairwise swaps and reinsertion moves, evaluated with exact makespan.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    time_budget = 20.0  # seconds of search
    start_time = time.time()

    cost_fn = workload.get_opt_seq_cost

    def local_search(seq):
        """Hill-climb with swap + reinsertion moves until no improvement."""
        current_cost = cost_fn(seq)
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            # pairwise swaps
            for i in range(n - 1):
                if time.time() - start_time > time_budget:
                    break
                for j in range(i + 1, n):
                    seq[i], seq[j] = seq[j], seq[i]
                    c = cost_fn(seq)
                    if c < current_cost:
                        current_cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
            # reinsertion moves (remove t at i, insert at k)
            for i in range(n):
                if time.time() - start_time > time_budget:
                    break
                t = seq.pop(i)
                best_k, best_c = i, current_cost
                for k in range(n):
                    seq.insert(k, t)
                    c = cost_fn(seq)
                    if c < best_c:
                        best_c, best_k = c, k
                    seq.pop(k)
                seq.insert(best_k, t)
                if best_c < current_cost:
                    current_cost = best_c
                    improved = True
        return current_cost, seq

    best_cost = float('inf')
    best_seq = None

    restart = 0
    while time.time() - start_time < time_budget:
        # random restart (or perturbation of best found so far)
        if best_seq is not None and restart > 0 and random.random() < 0.5:
            seq = best_seq[:]
            # perturb: random double-swap
            for _ in range(3):
                a, b = random.randrange(n), random.randrange(n)
                seq[a], seq[b] = seq[b], seq[a]
        else:
            seq = list(range(n))
            random.shuffle(seq)

        cost, seq = local_search(seq)
        if cost < best_cost:
            best_cost, best_seq = cost, seq[:]
        restart += 1

    assert len(set(best_seq)) == n
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
