import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan schedule via beam search with randomized restarts,
    then refine the best candidates with time-budgeted local search
    (insertion + swap moves) evaluated by actual makespan cost.
    """
    import time

    n = workload.num_txns
    if n == 0:
        return 0, []

    start_time = time.time()
    time_budget = 20.0

    best_cost = float('inf')
    best_seq = None

    # ---------- Phase 1: beam search with randomized restarts ----------
    num_restarts = max(1, min(12, 300 // max(1, n)))
    wide_beam = max(6, min(12, 800 // max(1, n)))
    narrow_beam = max(2, min(4, 200 // max(1, n)))

    candidates = []
    for restart in range(num_restarts):
        if time.time() - start_time >= time_budget * 0.6:
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
        beam = beam[:beam_width]

        while beam and len(beam[0][1]) < n:
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
            # Adaptive beam width: wide early (pruning error compounds),
            # narrow late (choices matter less, saves time).
            done_len = len(beam[0][1])
            if done_len < 0.25 * n:
                w = wide_beam
            elif done_len < 0.6 * n:
                w = (wide_beam + narrow_beam) // 2
            else:
                w = narrow_beam
            cands.sort(key=lambda x: x[0])
            beam = cands[:w]

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

    # ---------- Phase 2: local search on top candidates ----------
    candidates.sort(key=lambda x: x[0])
    for c_cost, c_seq in candidates[:3]:
        if time.time() - start_time >= time_budget:
            break
        seq = c_seq[:]
        cur_cost = c_cost
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            # insertion moves
            for i in range(n):
                if time.time() - start_time >= time_budget:
                    break
                for j in range(n):
                    if i == j:
                        continue
                    if time.time() - start_time >= time_budget:
                        break
                    cand = seq[:i] + seq[i+1:]
                    cand.insert(j, seq[i])
                    c = workload.get_opt_seq_cost(cand)
                    if c < cur_cost:
                        seq, cur_cost = cand, c
                        improved = True
                if time.time() - start_time >= time_budget:
                    break
            # swap moves
            for i in range(n):
                if time.time() - start_time >= time_budget:
                    break
                for j in range(i + 1, n):
                    if time.time() - start_time >= time_budget:
                        break
                    cand = seq[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    c = workload.get_opt_seq_cost(cand)
                    if c < cur_cost:
                        seq, cur_cost = cand, c
                        improved = True
        if cur_cost < best_cost:
            best_cost, best_seq = cur_cost, seq

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