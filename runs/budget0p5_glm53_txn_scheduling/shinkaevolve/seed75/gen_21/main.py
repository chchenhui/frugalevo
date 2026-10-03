import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find schedule minimizing makespan using beam search + GRASP restarts
    + adjacent-swap local search improvement.
    """
    n = workload.num_txns
    all_txns = list(range(n))

    def seq_cost(seq):
        return workload.get_opt_seq_cost(seq)

    def beam_search(width, k_candidates):
        # each beam entry: (cost, seq, remaining)
        beams = [(seq_cost([t]), [t], [x for x in all_txns if x != t])
                 for t in random.sample(all_txns, min(width, n))]
        for _ in range(n - 1):
            candidates = []
            for cost, seq, remaining in beams:
                if not remaining:
                    candidates.append((cost, seq, remaining))
                    continue
                # evaluate all remaining (true incremental cost)
                scored = []
                for t in remaining:
                    c = seq_cost(seq + [t])
                    scored.append((c, t))
                scored.sort(key=lambda x: x[0])
                # restricted candidate list
                rcl = scored[:k_candidates]
                for c, t in rcl:
                    candidates.append((c, seq + [t],
                                       [x for x in remaining if x != t]))
            candidates.sort(key=lambda x: x[0])
            beams = candidates[:width]
        best = min(beams, key=lambda x: x[0])
        return best[0], best[1]

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

    def local_search(seq, cost, max_passes=30):
        """Hybrid local search: best-insertion sweep + adjacent-swap sweep."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            # --- insertion sweep: relocate each txn to best position ---
            for i in range(len(seq)):
                base = seq.copy()
                t = base.pop(i)
                best_c, best_pos = cost, i
                for j in range(len(seq)):
                    if j == i:
                        continue
                    cand = base.copy()
                    cand.insert(j, t)
                    c = seq_cost(cand)
                    if c < best_c:
                        best_c, best_pos = c, j
                if best_pos != i:
                    base.insert(best_pos, t)
                    seq, cost = base, best_c
                    improved = True
            # --- adjacent swap sweep ---
            for i in range(len(seq) - 1):
                cand = seq.copy()
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < cost:
                    seq, cost = cand, c
                    improved = True
        return cost, seq

    rng = random.Random(12345)
    best_cost, best_seq = beam_search(width=4, k_candidates=3)
    best_cost, best_seq = local_search(best_seq, best_cost)

    for _ in range(num_seqs * 2):
        c, s = grasp_run(rng, 3)
        c, s = local_search(s, c, max_passes=10)
        if c < best_cost:
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