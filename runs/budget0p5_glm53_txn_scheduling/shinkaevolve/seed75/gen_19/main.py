import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Greedy construction + insertion-move simulated annealing.
    Returns (best makespan found, best schedule).
    """
    n = workload.num_txns
    cost_fn = workload.get_opt_seq_cost
    TIME_BUDGET = 25.0  # seconds per workload call (3 workloads -> ~75s total)

    # ---------- greedy construction (random start, best-cost append) ----------
    def greedy_construct(sample_rate=1.0, rng=None):
        rng = rng or random
        start = rng.randint(0, n - 1)
        seq = [start]
        remaining = [t for t in range(n) if t != start]
        while remaining:
            if rng.random() > sample_rate and len(remaining) > 1:
                # random pick for diversification
                idx = rng.randint(0, len(remaining) - 1)
                seq.append(remaining.pop(idx))
                continue
            best_t, best_c = None, None
            for t in remaining:
                c = cost_fn(seq + [t])
                if best_c is None or c < best_c:
                    best_c, best_t = c, t
            seq.append(best_t)
            remaining.remove(best_t)
        return seq

    # ---------- simulated annealing with insertion moves ----------
    def anneal(seq, deadline):
        cur = list(seq)
        cur_c = cost_fn(cur)
        best, best_c = list(cur), cur_c
        T0 = max(1.0, cur_c * 0.02)   # initial temperature ~2% of cost
        T = T0
        # move budget tied to time
        it = 0
        while time.time() < deadline:
            # do a small batch of moves per time check
            for _ in range(50):
                it += 1
                # temperature decay (geometric, with reheat if stagnant)
                T = T0 * (0.999 ** (it % 2000))
                # random insertion move
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                t = cur.pop(i)
                cur.insert(j, t)
                c = cost_fn(cur)
                delta = c - cur_c
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
                    cur_c = c
                    if c < best_c:
                        best_c = c
                        best = list(cur)
                else:
                    # revert
                    t = cur.pop(j)
                    cur.insert(i, t)
        return best_c, best

    deadline = time.time() + TIME_BUDGET
    rng = random.Random(42)

    # initial solution: greedy (with some randomization to not waste budget)
    seq = greedy_construct(1.0, rng)
    best_c, best_seq = anneal(seq, deadline)
    return best_c, best_seq

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
