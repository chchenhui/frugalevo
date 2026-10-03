import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using beam search over greedy cost expansions
    combined with multiple randomized restarts.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns

    def beam_search(start_txn, beam_width):
        # Each beam entry: (cost, sequence_tuple, frozenset of used txns)
        beam = [(workload.get_opt_seq_cost([start_txn]), (start_txn,), frozenset([start_txn]))]
        for _ in range(n - 1):
            candidates = []
            for cost, seq, used in beam:
                remaining = [t for t in range(n) if t not in used]
                for t in remaining:
                    test_seq = list(seq) + [t]
                    c = workload.get_opt_seq_cost(test_seq)
                    candidates.append((c, seq + (t,), used | {t}))
            # keep the best beam_width distinct states (dedupe by used-set)
            candidates.sort(key=lambda x: x[0])
            new_beam = []
            seen = set()
            for c, s, u in candidates:
                key = u
                if key not in seen:
                    seen.add(key)
                    new_beam.append((c, s, u))
                if len(new_beam) >= beam_width:
                    break
            beam = new_beam
        best = min(beam, key=lambda x: x[0])
        return best[0], list(best[1])

    def greedy_cost(num_samples, sample_rate):
        # randomized greedy using true makespan cost
        start_txn = random.randint(0, n - 1)
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            sample = random.random()
            if sample > sample_rate and len(remaining) > 1:
                idx = random.randint(0, len(remaining) - 1)
                txn_seq.append(remaining.pop(idx))
                continue
            num = min(num_samples, len(remaining))
            sampled = random.sample(remaining, num)
            best_cost = None
            best_txn = sampled[0]
            for t in sampled:
                cost = workload.get_opt_seq_cost(txn_seq + [t])
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_txn = t
            txn_seq.append(best_txn)
            remaining.remove(best_txn)
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    best_cost, best_seq = None, None
    # beam search from a few different starting transactions
    beam_width = 3 if n > 30 else 4
    num_beam_starts = min(n, 3)
    starts = random.sample(range(n), num_beam_starts) if n > num_beam_starts else list(range(n))
    for s in starts:
        c, seq = beam_search(s, beam_width)
        if best_cost is None or c < best_cost:
            best_cost, best_seq = c, seq

    # randomized greedy restarts to complement beam search
    for _ in range(8):
        c, seq = greedy_cost(8, 0.9)
        if c < best_cost:
            best_cost, best_seq = c, seq

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