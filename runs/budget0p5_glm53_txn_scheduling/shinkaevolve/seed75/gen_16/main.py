import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    GRASP construction (greedy randomized, true incremental makespan cost)
    + hybrid local search alternating insertion sweeps and adjacent-swap sweeps.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    def seq_cost(seq):
        return workload.get_opt_seq_cost(seq)

    def grasp_run(rng, k_pool):
        """Greedy randomized construction: pick among top-k cheapest next txns."""
        seq = [rng.randrange(n)]
        remaining = [x for x in all_txns if x != seq[0]]
        cost = seq_cost(seq)
        while remaining:
            scored = []
            for t in remaining:
                scored.append((seq_cost(seq + [t]), t))
            scored.sort(key=lambda x: x[0])
            pool = scored[:min(k_pool, len(scored))]
            c, t = pool[rng.randrange(len(pool))]
            seq.append(t)
            cost = c
            remaining.remove(t)
        return cost, seq

    def insertion_sweep(seq, cost):
        """Try moving each txn to every other position; keep best improvement."""
        improved = False
        i = 0
        while i < len(seq):
            base = seq.copy()
            t = base.pop(i)
            best_c, best_j = cost, None
            for j in range(len(seq)):
                if j == i:
                    continue
                cand = base.copy()
                cand.insert(j, t)
                c = seq_cost(cand)
                if c < best_c:
                    best_c, best_j = c, j
            if best_j is not None:
                cand = base.copy()
                cand.insert(best_j, t)
                seq, cost = cand, best_c
                improved = True
            else:
                i += 1
        return improved, cost, seq

    def swap_sweep(seq, cost):
        """Adjacent swap sweep: swap neighbors while it improves."""
        improved = False
        i = 0
        while i < len(seq) - 1:
            cand = seq.copy()
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = seq_cost(cand)
            if c < cost:
                seq, cost = cand, c
                improved = True
            else:
                i += 1
        return improved, cost, seq

    def local_search(seq, cost, max_rounds=30):
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            imp1, cost, seq = insertion_sweep(seq, cost)
            imp2, cost, seq = swap_sweep(seq, cost)
            if not (imp1 or imp2):
                break
        return cost, seq

    rng = random.Random(12345)
    best_cost, best_seq = None, None

    # multi-start GRASP with hybrid local search
    num_restarts = 20
    for r in range(num_restarts):
        c, s = grasp_run(rng, 3)
        c, s = local_search(s, c, max_rounds=15 if r < 5 else 8)
        if best_cost is None or c < best_cost:
            best_cost, best_seq = c, s

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