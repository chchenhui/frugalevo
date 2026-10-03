import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Find a low-makespan schedule via repeated randomized beam search,
    running until the wall-clock budget expires. All compute is spent
    on beam search restarts (the dimension that demonstrably works).
    """
    import time

    n = workload.num_txns
    if n == 0:
        return 0, []

    start_time = time.time()
    time_budget = 25.0

    best_cost = float('inf')
    best_seq = None

    beam_width = max(2, min(8, 600 // max(1, n)))
    restart = 0

    while time.time() - start_time < time_budget:
        restart += 1
        # widen the beam slightly as we accumulate restarts
        bw = min(beam_width + restart, max(2, 1200 // max(1, n)))

        # seed beam with a few starting transactions (incl. txn 0 and randoms)
        seeds = set()
        seeds.add(0)
        for _ in range(3):
            seeds.add(random.randrange(n))
        beam = []
        for s in seeds:
            seq = [s]
            rem = [x for x in range(n) if x != s]
            c = workload.get_opt_seq_cost(seq)
            beam.append((c, seq, rem))
        beam.sort(key=lambda x: x[0])
        beam = beam[:bw]

        while beam and len(beam[0][1]) < n:
            if time.time() - start_time >= time_budget:
                break
            cands = []
            for c, seq, rem in beam:
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
            beam = cands[:bw]

        if beam:
            for c, seq, rem in beam:
                final_cost = workload.get_opt_seq_cost(seq)
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_seq = seq
        else:
            break

    if best_seq is None:
        best_seq = list(range(n))
        best_cost = workload.get_opt_seq_cost(best_seq)

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