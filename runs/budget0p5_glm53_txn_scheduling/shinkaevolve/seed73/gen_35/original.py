import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using greedy cost sampling strategy.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import time
    import math

    n = workload.num_txns
    if n == 0:
        return 0, []

    start_time = time.time()
    time_budget = 20.0

    best_cost = float('inf')
    best_seq = None

    # ---------- Phase 1: beam search with randomized restarts ----------
    num_restarts = max(1, min(8, 200 // max(1, n)))
    max_beam_width = max(2, min(6, 400 // max(1, n)))

    candidates = []
    for restart in range(num_restarts):
        if time.time() - start_time >= time_budget * 0.5:
            break
        # seed beam with a few starting transactions (incl. txn 0 and random)
        seeds = set()
        seeds.add(0)
        seeds.add(random.randrange(n))
        if n > 1:
            seeds.add(random.randrange(n))
        beam = []
        for s in seeds:
            seq = [s]
            rem = [x for x in range(n) if x != s]
            c = workload.get_opt_seq_cost(seq)
            beam.append((c, seq, rem))
        beam.sort(key=lambda x: x[0])
        beam = beam[:max_beam_width]

        while beam and len(beam[0][1]) < n:
            # adaptive beam width: wide early (branching compounds),
            # narrow toward the end
            depth = len(beam[0][1])
            frac = depth / max(1, n)
            if frac < 0.2:
                width = max_beam_width
            elif frac < 0.6:
                width = max(3, max_beam_width - 2)
            else:
                width = max(2, max_beam_width - 3)
            cands = []
            for c, seq, rem in beam:
                if not rem:
                    continue
                # expand with all remaining transactions (full lookahead)
                for t in rem:
                    new_seq = seq + [t]
                    new_rem = [x for x in rem if x != t]
                    cost = workload.get_opt_seq_cost(new_seq)
                    cands.append((cost, new_seq, new_rem))
            if not cands:
                break
            cands.sort(key=lambda x: x[0])
            beam = cands[:width]

        if beam:
            for c, seq, rem in beam:
                final_cost = workload.get_opt_seq_cost(seq)
                candidates.append((final_cost, seq))
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_seq = seq

    if best_seq is None:
        # fallback: trivial sequence
        best_seq = list(range(n))
        best_cost = workload.get_opt_seq_cost(best_seq)
        candidates.append((best_cost, best_seq[:]))

    # ---------- Phase 2: simulated annealing from top candidates ----------
    candidates.sort(key=lambda x: x[0])
    rng = random.Random(1234)
    for c_cost, c_seq in candidates[:4]:
        if time.time() - start_time >= time_budget:
            break
        seq = c_seq[:]
        cur_cost = c_cost
        local_best_cost, local_best_seq = cur_cost, seq[:]
        # temperature schedule: start hot, cool with elapsed time
        T0 = max(1.0, cur_cost * 0.05)
        rem_budget = max(0.001, time_budget - (time.time() - start_time))
        cand_start = time.time()
        while True:
            elapsed = time.time() - cand_start
            if elapsed >= rem_budget or time.time() - start_time >= time_budget:
                break
            frac = elapsed / rem_budget
            T = T0 * (1.0 - frac) + 1e-6
            # random move: insertion or swap
            if rng.random() < 0.5:
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i == j:
                    continue
                cand = seq[:i] + seq[i+1:]
                cand.insert(j, seq[i])
            else:
                i = rng.randrange(n)
                j = rng.randrange(n)
                if i == j:
                    continue
                cand = seq[:]
                cand[i], cand[j] = cand[j], cand[i]
            c = workload.get_opt_seq_cost(cand)
            delta = c - cur_cost
            if delta < 0 or rng.random() < math.exp(-delta / T):
                seq, cur_cost = cand, c
                if cur_cost < local_best_cost:
                    local_best_cost, local_best_seq = cur_cost, seq[:]
                    # intensify: greedy insertion sweep around new best
                    for i2 in range(n):
                        best_c, best_cand = local_best_cost, None
                        for j2 in range(n):
                            if i2 == j2:
                                continue
                            trial = local_best_seq[:i2] + local_best_seq[i2+1:]
                            trial.insert(j2, local_best_seq[i2])
                            tc = workload.get_opt_seq_cost(trial)
                            if tc < best_c:
                                best_c, best_cand = tc, trial
                        if best_cand is not None:
                            local_best_cost = best_c
                            local_best_seq = best_cand
                    seq, cur_cost = local_best_seq[:], local_best_cost
        if local_best_cost < best_cost:
            best_cost, best_seq = local_best_cost, local_best_seq

    assert best_seq is not None and len(set(best_seq)) == n
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