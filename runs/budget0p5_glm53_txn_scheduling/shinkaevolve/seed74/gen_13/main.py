import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    GRASP-style search: beam search construction + randomized greedy restarts,
    refined by 2-opt and insertion local search with perturbation kicks.
    """
    n = workload.num_txns

    def greedy_construct(start_txn, alpha=0.3):
        txn_seq = [start_txn]
        remaining = [x for x in range(n) if x != start_txn]
        while remaining:
            test = txn_seq + [0]
            costs = []
            for t in remaining:
                test[-1] = t
                costs.append((workload.get_opt_seq_cost(test), t))
            costs.sort()
            # restricted candidate list
            cmin = costs[0][0]
            threshold = cmin + alpha * max(1e-9, costs[-1][0] - cmin)
            rcl = [t for c, t in costs if c <= threshold]
            t = random.choice(rcl)
            txn_seq.append(t)
            remaining.remove(t)
        return workload.get_opt_seq_cost(txn_seq), txn_seq

    def beam_construct(width=5):
        # beam search over partial sequences
        beam = [(workload.get_opt_seq_cost([t]), [t], {t}) for t in range(min(n, width))]
        while len(beam[0][1]) < n:
            candidates = []
            for cost, seq, used in beam:
                test = seq + [0]
                scored = []
                for t in range(n):
                    if t in used:
                        continue
                    test[-1] = t
                    scored.append((workload.get_opt_seq_cost(test), t))
                scored.sort()
                for c, t in scored[:3]:
                    ns = seq + [t]
                    candidates.append((c, ns, used | {t}))
            candidates.sort(key=lambda x: x[0])
            # dedupe by set of used txns, keep best per set
            seen = set()
            beam = []
            for c, ns, us in candidates:
                k = frozenset(us)
                if k not in seen:
                    seen.add(k)
                    beam.append((c, ns, us))
                if len(beam) >= width:
                    break
        best = min(beam, key=lambda x: workload.get_opt_seq_cost(x[1]))
        return workload.get_opt_seq_cost(best[1]), best[1]

    def local_search(seq, cost, rounds=8):
        improved = True
        rnd = 0
        while improved and rnd < rounds:
            improved = False
            rnd += 1
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    t = seq[i]
                    cand = seq[:i] + seq[i+1:]
                    cand.insert(j, t)
                    c = workload.get_opt_seq_cost(cand)
                    if c < cost:
                        cost, seq, improved = c, cand, True
            # 2-opt reversals
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < cost:
                        cost, seq, improved = c, cand, True
        return cost, seq

    def perturb(seq):
        # double-bridge kick
        s = seq[:]
        if n < 8:
            random.shuffle(s)
            return s
        a, b, c = sorted(random.sample(range(1, n), 3))
        return s[:a] + s[b:c] + s[a:b] + s[c:]

    best_cost, best_seq = float('inf'), None

    # beam search first (deterministic-ish strong start)
    c, sq = beam_construct()
    c, sq = local_search(sq, c)
    if c < best_cost:
        best_cost, best_seq = c, sq

    # randomized greedy restarts + local search, within time budget
    deadline = time.time() + 25
    restarts = 0
    while time.time() < deadline and restarts < 40:
        restarts += 1
        start = random.randrange(n)
        c, sq = greedy_construct(start)
        c, sq = local_search(sq, c)
        if c < best_cost:
            best_cost, best_seq = c, sq

    # iterated local search from incumbent
    while time.time() < deadline:
        sq = perturb(best_seq)
        c = workload.get_opt_seq_cost(sq)
        c, sq = local_search(sq, c)
        if c < best_cost:
            best_cost, best_seq = c, sq

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
