import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Hybrid: beam search with randomized restarts + randomized greedy
    constructions, then time-budgeted iterated local search
    (insertion + swap moves with perturbation restarts).
    """
    import time

    n = workload.num_txns
    if n == 0:
        return 0, []

    start_time = time.time()
    time_budget = 20.0

    best_cost = float('inf')
    best_seq = None
    candidates = []

    # ---------- Beam search with randomized restarts, time-budgeted ----------
    # Dynamically run as many restarts as the budget allows, increasing
    # the beam width over time so later restarts explore more broadly.
    restart = 0
    while time.time() - start_time < time_budget:
        beam_width = max(2, min(12, 3 + restart))
        restart += 1
        seeds = {0, random.randrange(n)}
        if n > 1:
            seeds.add(random.randrange(n))
        beam = []
        for s in seeds:
            seq = [s]
            rem = [x for x in range(n) if x != s]
            beam.append((seq, rem))
        while beam and len(beam[0][0]) < n:
            cands = []
            for seq, rem in beam:
                if not rem:
                    continue
                for t in rem:
                    new_seq = seq + [t]
                    new_rem = [x for x in rem if x != t]
                    cost = workload.get_opt_seq_cost(new_seq)
                    cands.append((cost, new_seq, new_rem))
            if not cands:
                break
            cands.sort(key=lambda x: x[0])
            beam = [(c[1], c[2]) for c in cands[:beam_width]]
        if beam:
            for seq, rem in beam:
                final_cost = workload.get_opt_seq_cost(seq)
                candidates.append((final_cost, seq))
                if final_cost < best_cost:
                    best_cost, best_seq = final_cost, seq

    # ---------- Phase 1b: randomized greedy constructions (diversity) ----------
    def greedy_construct(sample_rate):
        start = random.randrange(n)
        seq = [start]
        rem = [x for x in range(n) if x != start]
        while rem:
            if random.random() < sample_rate:
                # pick the best of a few sampled candidates
                k = min(len(rem), 10)
                best_t, best_c = None, float('inf')
                for t in random.sample(rem, k):
                    c = workload.get_opt_seq_cost(seq + [t])
                    if c < best_c:
                        best_c, best_t = c, t
                t = best_t
            else:
                t = random.choice(rem)
            seq.append(t)
            rem.remove(t)
        return workload.get_opt_seq_cost(seq), seq

    while time.time() - start_time < time_budget * 0.4:
        c, seq = greedy_construct(random.choice([0.5, 0.8, 1.0]))
        candidates.append((c, seq))
        if c < best_cost:
            best_cost, best_seq = c, seq

    if best_seq is None:
        best_seq = list(range(n))
        best_cost = workload.get_opt_seq_cost(best_seq)
        candidates.append((best_cost, best_seq[:]))

    # No local search: budget is better spent on more beam restarts.
    candidates.sort(key=lambda x: x[0])
    if candidates and candidates[0][0] < best_cost:
        best_cost, best_seq = candidates[0][0], candidates[0][1]

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