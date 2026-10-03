import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Conflict-aware greedy construction + full-neighborhood local search
    (all swaps, all insertions) + iterated local search with ruin-and-recreate
    kicks, all evaluated with the true makespan objective.
    """

    def greedy_construct(rng):
        """Greedy: repeatedly append the transaction whose addition yields
        the lowest true makespan of the partial schedule."""
        remaining = list(range(workload.num_txns))
        rng.shuffle(remaining)
        seq = [remaining.pop(0)]
        while remaining:
            best_c, best_i = None, 0
            # evaluate a random subset when large, all when small
            k = len(remaining)
            sample = remaining if k <= 12 else rng.sample(remaining, 12)
            for t in sample:
                c = workload.get_opt_seq_cost(seq + [t])
                if best_c is None or c < best_c:
                    best_c, best_i = c, t
            seq.append(best_i)
            remaining.remove(best_i)
        return workload.get_opt_seq_cost(seq), seq

    def local_search(seq, cost, deadline):
        """Full-neighborhood descent: every pairwise swap and every
        insertion move, evaluated with the true objective."""
        n = len(seq)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            # all pairwise swaps
            for i in range(n - 1):
                for j in range(i + 1, n):
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost, improved = c, True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
                if _time.time() > deadline:
                    return cost, seq
            # all insertion moves
            for i in range(n):
                t = seq.pop(i)
                for j in range(n):
                    if j == i:
                        continue
                    seq.insert(j, t)
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost, improved = c, True
                        i = j
                        break
                    seq.pop(j)
                else:
                    seq.insert(i, t)
                if _time.time() > deadline:
                    return cost, seq
        return cost, seq

    def ruin_and_recreate(seq, rng, block=8):
        """Kick: remove a contiguous block (or scattered random picks) and
        greedily re-insert each removed transaction at its best position."""
        s = seq[:]
        n = len(s)
        if rng.random() < 0.35 and n > block:
            # scattered ruin: remove random positions
            idxs = sorted(rng.sample(range(n), min(block, n)), reverse=True)
            removed = [s[i] for i in idxs]
            for i in idxs:
                del s[i]
        else:
            i = rng.randint(0, max(0, n - block))
            removed = s[i:i + block]
            del s[i:i + block]
        rng.shuffle(removed)
        for t in removed:
            best_c, best_j = None, 0
            for j in range(len(s) + 1):
                s.insert(j, t)
                c = workload.get_opt_seq_cost(s)
                if best_c is None or c < best_c:
                    best_c, best_j = c, j
                s.pop(j)
            s.insert(best_j, t)
        return s

    import time as _time
    rng = random.Random(12345)
    deadline = _time.time() + 115  # per-workload budget; 3 workloads < 360s

    # Phase 1: a few greedy seeds, refine the best ones
    best_cost, best_seq = float('inf'), None
    while _time.time() < deadline - 75:
        c, s = greedy_construct(rng)
        c, s = local_search(s, c, min(deadline - 60, _time.time() + 15))
        if c < best_cost:
            best_cost, best_seq = c, s[:]

    if best_seq is None:
        best_cost, best_seq = workload.get_opt_seq_cost(list(range(workload.num_txns))), list(range(workload.num_txns))

    # Phase 2: iterated local search with diversified ruin-and-recreate
    # kicks; restart from best when stagnating to escape plateaus.
    cur_cost, cur_seq = best_cost, best_seq[:]
    stagnation = 0
    while _time.time() < deadline:
        cand = ruin_and_recreate(cur_seq, rng, block=rng.randint(3, 12))
        c = workload.get_opt_seq_cost(cand)
        c, cand = local_search(cand, c, deadline)
        if c < best_cost:
            best_cost, best_seq = c, cand[:]
            stagnation = 0
        if c <= cur_cost:
            cur_cost, cur_seq = c, cand
            stagnation += 1 if c == cur_cost else 0
        else:
            stagnation += 1
        if stagnation >= 40:
            # stagnated: restart intensification from the best solution
            cur_cost, cur_seq = best_cost, best_seq[:]
            stagnation = 0
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
