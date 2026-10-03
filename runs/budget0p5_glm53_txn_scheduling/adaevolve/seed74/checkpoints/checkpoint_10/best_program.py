import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Iterated Local Search (ILS) with randomized-greedy construction.

    1. Construct an initial ordering with a sampled greedy pass (append
       the sampled transaction minimizing full-sequence cost).
    2. Refine with first-improvement local search over full pairwise
       swaps AND relocations, accepting equal-cost moves to cross
       plateaus.
    3. Perturb the current incumbent (segment reversal + random swaps)
       with adaptive kick strength, re-refine, and accept improvements,
       plateau moves, and occasionally worse moves (SA-style) to drift
       between basins. Restart from fresh greedy constructions when
       deeply stagnated. Keep the best schedule found overall.
    """
    import time

    n = workload.num_txns
    deadline = time.time() + 90

    def greedy_pass(num_samples):
        start_txn = random.randint(0, n - 1)
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(n) if x != start_txn]
        while remaining_txns:
            k = min(num_samples, len(remaining_txns))
            candidates = random.sample(remaining_txns, k)
            min_cost = float("inf")
            min_txn = candidates[0]
            for t in candidates:
                cost = workload.get_opt_seq_cost(txn_seq + [t])
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
            txn_seq.append(min_txn)
            remaining_txns.remove(min_txn)
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    def local_search(seq, cost):
        """First-improvement over full pairwise swaps + relocations,
        accepting equal-cost moves to traverse plateaus."""
        seq = list(seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # full pairwise swaps (randomized order)
            pairs = [(i, j) for i in range(n - 1)
                     for j in range(i + 1, n)]
            random.shuffle(pairs)
            for i, j in pairs:
                if time.time() >= deadline:
                    return cost, seq
                seq[i], seq[j] = seq[j], seq[i]
                c = workload.get_opt_seq_cost(seq)
                if c <= cost:
                    if c < cost:
                        improved = True
                    cost = c
                else:
                    seq[i], seq[j] = seq[j], seq[i]
            # relocation moves: remove txn at i, insert at j
            for i in range(n):
                if time.time() >= deadline:
                    return cost, seq
                t = seq.pop(i)
                for j in range(n):
                    if time.time() >= deadline:
                        seq.insert(i, t)
                        return cost, seq
                    seq.insert(j, t)
                    c = workload.get_opt_seq_cost(seq)
                    if c <= cost:
                        if c < cost:
                            improved = True
                        cost = c
                        break
                    seq.pop(j)
                else:
                    seq.insert(i, t)
        return cost, seq

    def perturb(seq, strength):
        """Kick: reverse a random segment and apply swaps; strength
        scales the kick size so stagnation triggers bigger jumps."""
        seq = list(seq)
        seg = min(4 + 8 * strength, n - 1)
        a = random.randint(0, n - 2)
        b = random.randint(a + 1, min(a + seg, n - 1))
        seq[a:b + 1] = reversed(seq[a:b + 1])
        for _ in range(2 + 2 * strength):
            i, j = random.randrange(n), random.randrange(n)
            seq[i], seq[j] = seq[j], seq[i]
        return seq

    # construction + initial refinement
    best_cost, best_seq = greedy_pass(12)
    best_cost, best_seq = local_search(best_seq, best_cost)

    # ILS with SA-style incumbent walking
    cur_cost, cur_seq = best_cost, best_seq
    stagnation = 0
    temperature = 0.02
    while time.time() < deadline:
        strength = min(stagnation // 8, 5)
        cand = perturb(cur_seq, strength)
        cost = workload.get_opt_seq_cost(cand)
        cost, cand = local_search(cand, cost)
        if cost < best_cost:
            best_cost, best_seq = cost, cand
            stagnation = 0
        elif cost <= cur_cost:
            stagnation += 1
        else:
            # SA-style acceptance of worse candidates
            if random.random() < temperature:
                stagnation += 1
            else:
                cand, cost = None, None
        if cand is not None:
            cur_cost, cur_seq = cost, cand
        # slow temperature decay with a floor
        temperature = max(temperature * 0.995, 0.002)
        # deep stagnation: fresh greedy restart into a new basin
        if stagnation and stagnation % 60 == 0:
            c, s = greedy_pass(12)
            c, s = local_search(s, c)
            if c < best_cost:
                best_cost, best_seq = c, s
            cur_cost, cur_seq = c, s
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
