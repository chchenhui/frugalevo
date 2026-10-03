import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

import time

def get_best_schedule(workload, num_seqs):
    """
    Approach: breadth-limited beam search over partial permutations with
    last-transaction diversity enforcement, a cached true-cost oracle, and
    an iterated local search tail (or-opt reinsertion + adjacent swaps +
    random 3-perturbation restarts).

    Mechanism (beam-search partial-schedule expansion): the beam maintains
    k diverse partial prefixes alive simultaneously. At each depth, every
    state is expanded with a sampled set of next transactions, each
    successor evaluated with the simulator's TRUE cost
    (workload.get_opt_seq_cost on the extended prefix). Selection enforces
    diversity: at most 2 survivors may share the same last transaction, so
    the beam retains genuinely different orderings rather than clones
    differing only in one slot; the cheapest full permutation wins.

    Refinement tail: identical to before — or-opt fixes long-range
    misplacements, adjacent swaps fix local inversions, and random
    3-position perturbations diversify when stuck, all
    accept-only-if-better, until the per-workload deadline.

    Budget: ~30s per workload (hard per-workload deadline), keeping the
    three-workload total well inside the 360s limit. A cache keyed on the
    sequence tuple avoids recomputing repeated prefixes. A valid identity
    permutation is always available as a fallback before any search, and
    every beam extension appends a distinct unplaced transaction, so any
    completed state is a full permutation of range(n).
    """
    n = workload.num_txns
    t0 = time.time()
    deadline = t0 + 30.0
    sample_size = max(3, n // 3)
    beam_k = 12
    max_per_last = 2

    cache = {}

    def seq_cost(seq):
        key = tuple(seq)
        c = cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            cache[key] = c
        return c

    best_seq = list(range(n))
    best_cost = seq_cost(best_seq)

    # --- Beam search over partial permutations ---
    # State: (cost_of_prefix, prefix_list). Initialized with each single
    # start transaction (capped at beam_k starts for diversity).
    starts = list(range(min(beam_k, n)))
    beam = []
    for s in starts:
        beam.append((seq_cost([s]), [s]))

    depth = 1
    while beam and time.time() < deadline:
        successors = []
        for cost, seq in beam:
            placed = set(seq)
            # Sample unplaced transactions.
            unplaced = [x for x in range(n) if x not in placed]
            if len(unplaced) <= sample_size:
                cands = unplaced
            else:
                cands = random.sample(unplaced, sample_size)
            for t in cands:
                ext = seq + [t]
                c = seq_cost(ext)
                successors.append((c, ext))
            if time.time() > deadline:
                break
        if not successors:
            break
        # Sort by true cost; dedupe identical prefixes; keep beam_k best,
        # capping survivors sharing the same last transaction for diversity.
        successors.sort(key=lambda p: p[0])
        new_beam = []
        seen = set()
        last_count = {}
        for c, seq in successors:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            lt = seq[-1]
            if last_count.get(lt, 0) >= max_per_last:
                continue
            last_count[lt] = last_count.get(lt, 0) + 1
            new_beam.append((c, seq))
            if len(new_beam) >= beam_k:
                break
        # Harvest any completed permutations.
        for c, seq in new_beam:
            if len(seq) == n and c < best_cost:
                best_cost, best_seq = c, seq
        beam = [(c, seq) for c, seq in new_beam if len(seq) < n]
        depth += 1
        # Later depths have fewer states; widen the candidate sample.
        if len(beam) <= 4:
            sample_size = max(sample_size, n // 2)

    # --- Iterated local search refinement until deadline ---
    def local_search(seq, cost):
        # Or-opt: try reinserting each transaction at every other position.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                item = seq[i]
                rest = seq[:i] + seq[i + 1:]
                for j in range(n):
                    if j == i:
                        continue
                    cand = rest[:j] + [item] + rest[j:]
                    c = seq_cost(cand)
                    if c < cost:
                        seq, cost = cand, c
                        improved = True
                        break
                if improved:
                    break
        # Adjacent swaps: first-improvement.
        while time.time() < deadline:
            improved = False
            for i in range(n - 1):
                cand = list(seq)
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < cost:
                    seq, cost = cand, c
                    improved = True
                    if time.time() > deadline:
                        break
            if not improved:
                break
        return seq, cost

    best_seq, best_cost = local_search(best_seq, best_cost)
    cur_seq, cur_cost = list(best_seq), best_cost

    # Perturbation restarts: random 3-position shuffles, re-optimized.
    while time.time() < deadline:
        cand = list(cur_seq)
        for a in random.sample(range(n), 3):
            b = random.randrange(n)
            cand[a], cand[b] = cand[b], cand[a]
        c = seq_cost(cand)
        cand, c = local_search(cand, c)
        if c < cur_cost:
            cur_seq, cur_cost = cand, c
        if cur_cost < best_cost:
            best_cost, best_seq = cur_cost, list(cur_seq)

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
